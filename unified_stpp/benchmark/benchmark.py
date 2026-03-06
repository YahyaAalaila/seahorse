"""
Benchmark — two-stage HPO + multi-seed evaluation across presets × datasets.

Stage 1 (optional):  tune_all()  — HPO per preset on a single dataset.
Stage 2:             run()       — grid of (preset × dataset × seed).
"""

from __future__ import annotations

from typing import Any, Optional

from unified_stpp.config import STPPConfig
from unified_stpp.runner import STPPRunner, RunResult
from .results import BenchmarkTable


# ---------------------------------------------------------------------------
# Module-level picklable worker (needed by joblib)
# ---------------------------------------------------------------------------

def _run_single(
    preset: str,
    dataset_id: str,
    seed: int,
    config: STPPConfig,
    train_seqs: list[dict],
    val_seqs: list[dict],
    test_seqs: Optional[list[dict]],
) -> RunResult:
    """Run one (preset, dataset, seed) trial and return a RunResult."""
    import copy

    cfg = copy.deepcopy(config)
    # Override seed
    cfg.data.seed = seed

    runner = STPPRunner(cfg)
    return runner.fit(train_seqs, val_seqs, test_seqs, dataset_id=dataset_id)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class Benchmark:
    """Run a full benchmark grid over presets × datasets × seeds.

    Parameters
    ----------
    presets:        List of preset names (e.g. ``["auto_stpp", "deep_stpp"]``).
    splits:         ``{dataset_id: (train_seqs, val_seqs, test_seqs)}``.
    seeds:          List of random seeds for multi-seed runs.
    base_overrides: Dict merged into every preset config before HPO/training.
    hpo_configs:    Pre-computed best configs from ``tune_all()``; if provided,
                    ``run()`` skips Stage 1 and uses these directly.

    Usage
    -----
    >>> bench = Benchmark(["auto_stpp"], {"toy": (tr, va, te)}, seeds=[42, 0])
    >>> table = bench.run()
    >>> table.report("/tmp/out/")
    """

    def __init__(
        self,
        presets: list[str],
        splits: dict[str, tuple[list, list, list]],
        seeds: list[int] = None,
        base_overrides: dict[str, Any] = None,
        hpo_configs: Optional[dict[str, STPPConfig]] = None,
    ):
        self.presets = presets
        self.splits = splits
        self.seeds = seeds if seeds is not None else [42]
        self.base_overrides = base_overrides or {}
        self.hpo_configs: dict[str, STPPConfig] = hpo_configs or {}

    # ------------------------------------------------------------------
    # Stage 1: HPO
    # ------------------------------------------------------------------

    def tune_all(
        self,
        n_trials: int = 50,
        algorithm: str = "asha",
        tune_dataset: Optional[str] = None,
        n_workers: int = 1,
    ) -> dict[str, STPPConfig]:
        """Run HPO for every preset.  Results are stored in ``self.hpo_configs``.

        Parameters
        ----------
        n_trials:      Number of trials per preset.
        algorithm:     ``"asha"`` | ``"bayesian"`` | ``"grid"``.
        tune_dataset:  Dataset to use for HPO (defaults to the first key in splits).
        n_workers:     Number of parallel Ray workers (passed to ``run_hpo``).
        """
        from unified_stpp.benchmark.hpo import run_hpo

        ds_key = tune_dataset or next(iter(self.splits))
        train_seqs, val_seqs, _ = self.splits[ds_key]

        for preset in self.presets:
            cfg = self._base_config(preset)
            best_cfg = run_hpo(
                config_dict=cfg.model_dump(mode="json"),
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                n_trials=n_trials,
                algorithm=algorithm,
            )
            self.hpo_configs[preset] = best_cfg

        return self.hpo_configs

    # ------------------------------------------------------------------
    # Stage 2: Multi-seed evaluation
    # ------------------------------------------------------------------

    def run(
        self,
        n_workers: int = 1,
        backend: str = "joblib",
    ) -> BenchmarkTable:
        """Evaluate all (preset × dataset × seed) combinations.

        Parameters
        ----------
        n_workers:  Degree of parallelism (``1`` = sequential, safe for debugging).
        backend:    ``"joblib"`` (default) or ``"sequential"``.

        Returns a :class:`BenchmarkTable` with all ``RunResult`` objects.
        """
        jobs = []
        for preset in self.presets:
            cfg = self.hpo_configs.get(preset) or self._base_config(preset)
            for dataset_id, (train_seqs, val_seqs, test_seqs) in self.splits.items():
                for seed in self.seeds:
                    jobs.append((preset, dataset_id, seed, cfg, train_seqs, val_seqs, test_seqs))

        if n_workers == 1 or backend == "sequential":
            results = [_run_single(*j) for j in jobs]
        else:
            results = self._run_parallel(jobs, n_workers, backend)

        return BenchmarkTable(runs=results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_config(self, preset: str) -> STPPConfig:
        """Build base STPPConfig for *preset*, applying ``base_overrides``."""
        cfg = STPPConfig.from_preset(preset)
        if self.base_overrides:
            import copy
            raw = cfg.model_dump()
            _deep_update(raw, self.base_overrides)
            cfg = STPPConfig(**raw)
        return cfg

    def _run_parallel(self, jobs, n_workers: int, backend: str) -> list[RunResult]:
        if backend == "ray":
            try:
                import ray
                from ray.remote_function import RemoteFunction as _
            except ImportError:
                raise ImportError("backend='ray' requires Ray: pip install 'unified-stpp[hpo]'")

            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            remote_fn = ray.remote(_run_single)
            futures = [remote_fn.remote(*j) for j in jobs]
            return ray.get(futures)

        else:  # joblib
            try:
                from joblib import Parallel, delayed
            except ImportError:
                raise ImportError("backend='joblib' requires joblib: pip install 'unified-stpp[runner]'")

            return Parallel(n_jobs=n_workers)(delayed(_run_single)(*j) for j in jobs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_update(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
