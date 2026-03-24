"""
Benchmark — two-stage HPO + multi-seed evaluation across presets × datasets.

Stage 1 (optional):  tune_all()  — HPO per preset on a single dataset.
Stage 2:             run()       — grid of (preset × dataset × seed).
"""

from __future__ import annotations

from typing import Any, Optional

from unified_stpp.config import STPPConfig
from unified_stpp.runner import STPPRunner, RunResult
from unified_stpp.utils import deep_update
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
    surface_viz=None,
) -> tuple:
    """Run one (preset, dataset, seed) trial and return a (RunResult, surfaces) tuple."""
    import copy

    cfg = copy.deepcopy(config)
    cfg.data.seed = seed

    runner = STPPRunner(cfg)
    result = runner.fit(train_seqs, val_seqs, test_seqs, dataset_id=dataset_id)

    surfaces = None
    if surface_viz is not None and getattr(surface_viz, "enabled", False):
        from unified_stpp.viz.workflow import SurfaceVisualizationWorkflow
        from unified_stpp.runner.artifacts import _extend_viz_manifest
        wf = SurfaceVisualizationWorkflow(surface_viz)
        artifacts = wf.run(runner, result.run_dir)
        _extend_viz_manifest(result.run_dir, artifacts)
        surfaces = wf.surfaces_

    return result, surfaces


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
    normalize:      Whether to z-score normalize time and space for all models.
                    Forced uniformly so that all presets see the same coordinate
                    system and NLL values are directly comparable.

    ML best-practices contract
    --------------------------
    The benchmark is the authority on data.  Splits are fixed before any model
    is initialised.  Every preset is forced to ``protocol="unified"`` so that
    no preset can re-split or re-format the data internally.  ``normalize`` is
    applied identically to all presets via the same training-set statistics.

    Attributes set after run()
    --------------------------
    runners_ : dict[str, STPPRunner] | None
        Fitted runners keyed by preset name (first seed per preset).
        Only available in sequential mode (``n_workers=1``); ``None`` in
        parallel mode.

    Usage
    -----
    >>> bench = Benchmark(["auto_stpp"], {"toy": (tr, va, te)}, seeds=[42, 0])
    >>> table = bench.run()
    >>> table.report("/tmp/out/")
    """

    runners_: Optional[dict] = None

    def __init__(
        self,
        presets: list[str],
        splits: dict[str, tuple[list, list, list]],
        seeds: list[int] = None,
        base_overrides: dict[str, Any] = None,
        hpo_configs: Optional[dict[str, STPPConfig]] = None,
        hpo_configs_dir: Optional[str] = None,
        normalize: bool = True,
    ):
        self.presets = presets
        self.splits = splits
        self.seeds = seeds if seeds is not None else [42]
        self.base_overrides = base_overrides or {}
        self.normalize = normalize
        self.hpo_configs: dict[str, STPPConfig] = hpo_configs or {}
        self.runners_: Optional[dict] = None

        # Load pre-saved HPO configs (from a previous tune_all run).
        # Any preset already in hpo_configs takes precedence.
        if hpo_configs_dir:
            from pathlib import Path
            hpo_dir = Path(hpo_configs_dir)
            for yaml_path in sorted(hpo_dir.glob("*_best.yaml")):
                preset_name = yaml_path.stem[: -len("_best")]
                if preset_name not in self.hpo_configs:
                    self.hpo_configs[preset_name] = STPPConfig.from_yaml(str(yaml_path), sanitize=False)
                    print(f"[bench] Loaded HPO config for '{preset_name}' from {yaml_path}")

    # ------------------------------------------------------------------
    # Stage 1: HPO
    # ------------------------------------------------------------------

    def tune_all(
        self,
        n_trials: int = 50,
        algorithm: str = "asha",
        tune_dataset: Optional[str] = None,
        n_workers: int = 1,
        out_dir: Optional[str] = None,
    ) -> dict[str, STPPConfig]:
        """Run HPO for every preset.  Results are stored in ``self.hpo_configs``.

        Parameters
        ----------
        n_trials:      Number of trials per preset.
        algorithm:     ``"asha"`` | ``"bayesian"`` | ``"grid"``.
        tune_dataset:  Dataset to use for HPO (defaults to the first key in splits).
        n_workers:     Number of parallel Ray workers (passed to ``run_hpo``).
        out_dir:       If set, saves each best config to ``{out_dir}/hpo/{preset}_best.yaml``
                       so it can be reloaded in a later run via ``hpo_configs_dir``.
        """
        from pathlib import Path
        from unified_stpp.benchmark.hpo import run_hpo

        ds_key = tune_dataset or next(iter(self.splits))
        train_seqs, val_seqs, _ = self.splits[ds_key]

        # Presets already loaded (e.g. from hpo_configs_dir) can be skipped.
        presets_to_tune = [p for p in self.presets if p not in self.hpo_configs]
        if len(presets_to_tune) < len(self.presets):
            skipped = [p for p in self.presets if p in self.hpo_configs]
            print(f"[bench] Skipping HPO for already-tuned presets: {skipped}")

        for preset in presets_to_tune:
            # Load raw YAML to preserve HPO search-space syntax (lists, {min/max}
            # dicts).  Using model_dump(mode="json") would convert tuples (e.g.
            # paper_split_ratio) to lists, which the HPO parser then mistakes for
            # discrete choice sets — causing Pydantic validation failures in trials.
            raw_dict = _load_raw_yaml(preset)
            # Force uniform data protocol: the benchmark owns the splits, so no
            # preset is allowed to re-split or change the time representation.
            raw_dict.setdefault("data", {})
            raw_dict["data"]["protocol"] = "unified"
            raw_dict["data"]["normalize"] = self.normalize
            if self.base_overrides:
                deep_update(raw_dict, self.base_overrides)

            best_cfg = run_hpo(
                config_dict=raw_dict,
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                n_trials=n_trials,
                algorithm=algorithm,
            )
            self.hpo_configs[preset] = best_cfg

            # Persist immediately so a partial run is recoverable.
            if out_dir:
                hpo_dir = Path(out_dir) / "hpo"
                hpo_dir.mkdir(parents=True, exist_ok=True)
                save_path = hpo_dir / f"{preset}_best.yaml"
                best_cfg.to_yaml(str(save_path))
                print(f"[bench] Saved best HPO config for '{preset}' to {save_path}")

        return self.hpo_configs

    # ------------------------------------------------------------------
    # Stage 2: Multi-seed evaluation
    # ------------------------------------------------------------------

    def run(
        self,
        n_workers: int = 1,
        backend: str = "joblib",
        surface_viz=None,
    ) -> BenchmarkTable:
        """Evaluate all (preset × dataset × seed) combinations.

        Parameters
        ----------
        n_workers:   Degree of parallelism (``1`` = sequential, safe for debugging).
        backend:     ``"joblib"`` (default) or ``"sequential"``.
        surface_viz: Optional :class:`~unified_stpp.viz.workflow.SurfaceVizConfig`.
                     When enabled, surface visualizations are generated per model
                     and benchmark-level comparison panels are added to the returned
                     :class:`BenchmarkTable`.  ``runners_`` is populated in
                     sequential mode only.

        Returns a :class:`BenchmarkTable` with all ``RunResult`` objects.
        """
        import copy

        jobs = []
        for preset in self.presets:
            cfg = self.hpo_configs.get(preset) or self._base_config(preset)
            # Apply the benchmark data contract to every config, including those
            # loaded from disk (hpo_configs_dir), which may carry stale protocol
            # settings from a previous run with different data handling.
            cfg = self._apply_data_contract(cfg)
            for dataset_id, (train_seqs, val_seqs, test_seqs) in self.splits.items():
                for seed in self.seeds:
                    jobs.append((preset, dataset_id, seed, cfg, train_seqs, val_seqs, test_seqs))

        if n_workers == 1 or backend == "sequential":
            # Inline sequential loop to capture fitted runners.
            self.runners_ = {}
            pairs = []
            for job in jobs:
                preset, dataset_id, seed, cfg, tr, va, te = job
                cfg_copy = copy.deepcopy(cfg)
                cfg_copy.data.seed = seed
                runner = STPPRunner(cfg_copy)
                result = runner.fit(tr, va, te, dataset_id=dataset_id)

                surfaces = None
                if surface_viz is not None and getattr(surface_viz, "enabled", False):
                    from unified_stpp.viz.workflow import SurfaceVisualizationWorkflow
                    from unified_stpp.runner.artifacts import _extend_viz_manifest
                    wf = SurfaceVisualizationWorkflow(surface_viz)
                    artifacts = wf.run(runner, result.run_dir)
                    _extend_viz_manifest(result.run_dir, artifacts)
                    surfaces = wf.surfaces_

                pairs.append((result, surfaces))
                if preset not in self.runners_:
                    self.runners_[preset] = runner
        else:
            self.runners_ = None
            pairs = self._run_parallel(jobs, n_workers, backend, surface_viz=surface_viz)

        # Aggregate surfaces (first seed per (preset, dataset) only)
        surfaces_by_dataset: dict = {}
        for (preset, dataset_id, *_), (result, surfaces) in zip(jobs, pairs):
            if surfaces is not None:
                surfaces_by_dataset.setdefault(dataset_id, {})
                if preset not in surfaces_by_dataset[dataset_id]:
                    surfaces_by_dataset[dataset_id][preset] = surfaces

        run_results = [r for r, _ in pairs]
        return BenchmarkTable(runs=run_results, surfaces_by_dataset=surfaces_by_dataset)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_data_contract(self, cfg: STPPConfig) -> STPPConfig:
        """Enforce the benchmark data contract on any STPPConfig.

        Forces ``protocol="unified"`` and ``normalize=self.normalize`` so that
        every preset — whether built fresh or loaded from a saved HPO file —
        uses the same coordinate system as the splits supplied to the benchmark.
        """
        raw = cfg.model_dump(mode="json")
        raw.setdefault("data", {})
        raw["data"]["protocol"] = "unified"
        raw["data"]["normalize"] = self.normalize
        return STPPConfig(**raw)

    def _base_config(self, preset: str) -> STPPConfig:
        """Build base STPPConfig for *preset*, applying ``base_overrides``."""
        cfg = STPPConfig.from_preset(preset)
        if self.base_overrides:
            raw = cfg.model_dump(mode="json")
            deep_update(raw, self.base_overrides)
            cfg = STPPConfig(**raw)
        return cfg

    def _run_parallel(self, jobs, n_workers: int, backend: str, surface_viz=None) -> list:
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
            futures = [remote_fn.remote(*j, surface_viz=surface_viz) for j in jobs]
            return ray.get(futures)

        else:  # joblib
            try:
                from joblib import Parallel, delayed
            except ImportError:
                raise ImportError("backend='joblib' requires joblib: pip install 'unified-stpp[runner]'")

            return Parallel(n_jobs=n_workers)(
                delayed(_run_single)(*j, surface_viz=surface_viz) for j in jobs
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_raw_yaml(preset: str) -> dict:
    """Return the raw (unsanitized) YAML dict for *preset*.

    Using the raw YAML preserves HPO search-space syntax (lists and {min/max}
    dicts) so the HPO parser can distinguish between tunable parameters and
    fixed config values.  Falls back to a minimal dict if no YAML file exists.
    """
    import yaml
    from pathlib import Path

    yaml_path = Path(__file__).parent.parent / "configs" / f"{preset}.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            return yaml.safe_load(f) or {}
    return {"model": {"preset": preset}}
