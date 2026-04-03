"""
Benchmark — optional HPO + multi-seed evaluation across presets × datasets.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

from unified_stpp.config import BenchmarkConfig, STPPConfig
from unified_stpp.runner import STPPRunner, RunResult
from unified_stpp.utils import deep_update

from .results import BenchmarkTable


def _run_single(
    preset: str,
    dataset_id: str,
    seed: int,
    config: STPPConfig,
    train_seqs: list[dict],
    val_seqs: list[dict],
    test_seqs: Optional[list[dict]],
) -> RunResult:
    """Run one (preset, dataset, seed) trial and return its RunResult."""
    cfg = copy.deepcopy(config)
    cfg.data.seed = seed

    runner = STPPRunner(cfg)
    return runner.fit(train_seqs, val_seqs, test_seqs, dataset_id=dataset_id)


class Benchmark:
    """Run a benchmark grid over presets × datasets × seeds.

    Procedure and policy live in ``BenchmarkConfig``. Invocation-specific inputs
    such as presets, resolved splits, output paths, and cross-preset overrides
    remain outside the config and are supplied directly to the benchmark.
    """

    def __init__(
        self,
        presets: list[str],
        splits: dict[str, tuple[list, list, list]],
        config: BenchmarkConfig,
        *,
        base_overrides: Optional[dict[str, Any]] = None,
        hpo_configs: Optional[dict[str, STPPConfig]] = None,
        hpo_configs_dir: Optional[str] = None,
        out_dir: Optional[str | Path] = None,
        # CLI-provenance fields written into bench_meta.json
        argv: Optional[list[str]] = None,
        splits_dir_str: Optional[str] = None,
        raw_overrides: Optional[list[str]] = None,
    ):
        self.presets = presets
        self.splits = splits
        self.config = config
        self.base_overrides = base_overrides or {}
        self.hpo_configs: dict[str, STPPConfig] = hpo_configs or {}
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self._hpo_configs_dir = hpo_configs_dir
        self._argv = argv
        self._splits_dir_str = splits_dir_str
        self._raw_overrides = raw_overrides or []

        if hpo_configs_dir:
            hpo_dir = Path(hpo_configs_dir)
            for yaml_path in sorted(hpo_dir.glob("*_best.yaml")):
                preset_name = yaml_path.stem[: -len("_best")]
                if preset_name not in self.hpo_configs:
                    self.hpo_configs[preset_name] = STPPConfig.from_yaml(
                        str(yaml_path), sanitize=False
                    )
                    print(f"[bench] Loaded HPO config for '{preset_name}' from {yaml_path}")

    def tune_all(self, *, out_dir: Optional[str | Path] = None) -> dict[str, STPPConfig]:
        """Run HPO for each preset not already covered by ``self.hpo_configs``."""
        from unified_stpp.benchmark.hpo import run_hpo

        tuning = self.config.resolved_tuning()
        ds_key = self.config.tune_dataset or next(iter(self.splits))
        if ds_key not in self.splits:
            raise ValueError(
                f"BenchmarkConfig.tune_dataset={ds_key!r} was not found in splits. "
                f"Available datasets: {sorted(self.splits)}"
            )
        train_seqs, val_seqs, _ = self.splits[ds_key]

        presets_to_tune = [p for p in self.presets if p not in self.hpo_configs]
        if len(presets_to_tune) < len(self.presets):
            skipped = [p for p in self.presets if p in self.hpo_configs]
            print(f"[bench] Skipping HPO for already-tuned presets: {skipped}")

        target_out = Path(out_dir) if out_dir is not None else self.out_dir
        for preset in presets_to_tune:
            raw_dict = STPPConfig.raw_source_dict(preset=preset)
            overrides = self._effective_base_overrides()
            if overrides:
                deep_update(raw_dict, overrides)
            # Enforce data contract on raw dict (HPO needs raw syntax preserved)
            raw_dict.setdefault("data", {})
            raw_dict["data"]["protocol"] = self.config.protocol
            raw_dict["data"]["normalize"] = self.config.normalize

            best_cfg = run_hpo(
                config_dict=raw_dict,
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                tuning=tuning,
            )
            self.hpo_configs[preset] = best_cfg

            if target_out is not None:
                hpo_dir = target_out / "hpo"
                hpo_dir.mkdir(parents=True, exist_ok=True)
                save_path = hpo_dir / f"{preset}_best.yaml"
                best_cfg.to_yaml(str(save_path))
                print(f"[bench] Saved best HPO config for '{preset}' to {save_path}")

        return self.hpo_configs

    def run(self) -> BenchmarkTable:
        """Run optional HPO followed by multi-seed evaluation."""
        self._write_meta()

        if self.config.run_hpo:
            self.tune_all()

        jobs = []
        for preset in self.presets:
            cfg = self.hpo_configs.get(preset) or self._base_config(preset)
            for dataset_id, (train_seqs, val_seqs, test_seqs) in self.splits.items():
                for seed in self.config.seeds:
                    jobs.append((preset, dataset_id, seed, cfg, train_seqs, val_seqs, test_seqs))

        if self.config.n_workers == 1 or self.config.backend == "sequential":
            results = []
            for preset, dataset_id, seed, cfg, tr, va, te in jobs:
                cfg_copy = copy.deepcopy(cfg)
                cfg_copy.data.seed = seed
                runner = STPPRunner(cfg_copy)
                results.append(runner.fit(tr, va, te, dataset_id=dataset_id))
        else:
            results = self._run_parallel(jobs, self.config.n_workers, self.config.backend)

        return BenchmarkTable(runs=results)

    def _write_meta(self) -> None:
        """Write bench_meta.json at run start (no-op when out_dir is not set)."""
        if self.out_dir is None or self._argv is None:
            return
        from unified_stpp.runner.artifacts import write_bench_meta
        write_bench_meta(
            out_dir=self.out_dir,
            bench_id=self.out_dir.name,
            argv=self._argv,
            splits_dir=self._splits_dir_str or str(self.out_dir),
            datasets=sorted(self.splits.keys()),
            presets=self.presets,
            benchmark_config=self.config,
            overrides=self._raw_overrides,
            hpo_configs_dir=self._hpo_configs_dir,
        )

    def _effective_base_overrides(self) -> dict[str, Any]:
        """Return cross-preset overrides plus the benchmark-local output dir."""
        overrides: dict[str, Any] = {}
        if self.out_dir is not None:
            overrides["logging"] = {"out_dir": str(self.out_dir)}
        if self.base_overrides:
            deep_update(overrides, self.base_overrides)
        return overrides

    def _base_config(self, preset: str) -> STPPConfig:
        """Build the effective STPPConfig for *preset* under benchmark policy."""
        cfg = STPPConfig.from_preset(preset)
        overrides = self._effective_base_overrides()
        if overrides:
            raw = cfg.model_dump(mode="json")
            deep_update(raw, overrides)
            cfg = STPPConfig(**raw)
        return self.config.apply_to_config(cfg)

    def _run_parallel(self, jobs, n_workers: int, backend: str) -> list[RunResult]:
        if backend == "ray":
            try:
                import ray
                from ray.remote_function import RemoteFunction as _
            except ImportError as exc:
                raise ImportError(
                    "backend='ray' requires Ray: pip install 'unified-stpp[hpo]'"
                ) from exc

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            remote_fn = ray.remote(_run_single)
            futures = [remote_fn.remote(*job) for job in jobs]
            return ray.get(futures)

        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError(
                "backend='joblib' requires joblib: pip install 'unified-stpp[runner]'"
            ) from exc

        return Parallel(n_jobs=n_workers)(delayed(_run_single)(*job) for job in jobs)
