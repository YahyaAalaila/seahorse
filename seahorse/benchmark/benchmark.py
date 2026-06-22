"""
Benchmark — optional HPO + multi-seed evaluation across presets × datasets.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from seahorse.config import BenchmarkConfig, STPPConfig
from seahorse.runner import STPPRunner, RunResult
from seahorse.utils import deep_update

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


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_tunable_spec(value: Any) -> bool:
    if isinstance(value, dict):
        return "min" in value and "max" in value
    if isinstance(value, list):
        return bool(value) and not isinstance(value[0], (dict, list))
    return False


def _nested_get(mapping: dict[str, Any], dotted_key: str) -> Any:
    node: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _nested_set(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = mapping
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


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
        data_manifest: Optional[dict[str, Any]] = None,
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
        self.data_manifest = copy.deepcopy(data_manifest) if data_manifest is not None else None
        self._hpo_configs_dir = hpo_configs_dir
        self._argv = argv
        self._splits_dir_str = splits_dir_str
        self._raw_overrides = raw_overrides or []
        self._locked_training_cache: dict[str, dict[str, Any]] = {}
        self._hpo_provenance: dict[str, dict[str, Any]] = {}
        self._data_manifest_path: Optional[Path] = None
        self._cell_index_path: Optional[Path] = None

        if hpo_configs_dir:
            hpo_dir = Path(hpo_configs_dir)
            for yaml_path in sorted(hpo_dir.glob("*_best.yaml")):
                preset_name = yaml_path.stem[: -len("_best")]
                if preset_name not in self.hpo_configs:
                    self.hpo_configs[preset_name] = STPPConfig.from_yaml(
                        str(yaml_path), sanitize=False
                    )
                    self._hpo_provenance[preset_name] = self._loaded_hpo_provenance(
                        preset_name,
                        yaml_path,
                    )
                    print(f"[bench] Loaded HPO config for '{preset_name}' from {yaml_path}")

    def tune_all(self, *, out_dir: Optional[str | Path] = None) -> dict[str, STPPConfig]:
        """Run HPO for each preset not already covered by ``self.hpo_configs``."""
        from seahorse.benchmark.hpo import (
            build_hpo_manifest,
            run_hpo,
            write_trial_history,
        )

        tuning = self.config.resolved_tuning()
        ds_key = self.config.tune_dataset
        if not ds_key:
            raise ValueError(
                "Benchmark HPO requires BenchmarkConfig.tune_dataset to be set explicitly."
            )
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
            self._assert_locked_fields_not_tunable(raw_dict, preset)
            self._apply_locked_policy_to_raw_dict(raw_dict, preset)

            hpo_result = run_hpo(
                config_dict=raw_dict,
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                tuning=tuning,
                dataset_id=ds_key,
                return_analysis=True,
            )
            if isinstance(hpo_result, tuple):
                best_cfg, analysis = hpo_result
            else:
                best_cfg, analysis = hpo_result, None
            self.hpo_configs[preset] = self._apply_benchmark_policy(preset, best_cfg)

            if target_out is not None:
                hpo_dir = target_out / "hpo"
                hpo_dir.mkdir(parents=True, exist_ok=True)
                save_path = hpo_dir / f"{preset}_best.yaml"
                self.hpo_configs[preset].to_yaml(str(save_path))
                trials_json = hpo_dir / f"{preset}_trials.json"
                trials_csv = hpo_dir / f"{preset}_trials.csv"
                if analysis is not None:
                    write_trial_history(analysis, json_path=trials_json, csv_path=trials_csv)
                provenance = build_hpo_manifest(
                    source="fresh_hpo",
                    preset=preset,
                    dataset_id=ds_key,
                    data_source_fingerprint=self._dataset_source_fingerprint(ds_key),
                    tuning=tuning,
                    best_config_path=save_path,
                    trials_json_path=trials_json,
                    trials_csv_path=trials_csv,
                    analysis=analysis,
                    argv=self._argv,
                    extra={
                        "search_space_fingerprint": _stable_json_sha256(raw_dict),
                    },
                )
                manifest_path = hpo_dir / f"{preset}_hpo_manifest.json"
                with open(manifest_path, "w") as f:
                    json.dump(provenance, f, indent=2, default=str)
                provenance["manifest_path"] = str(manifest_path.resolve())
                self._hpo_provenance[preset] = provenance
                print(f"[bench] Saved best HPO config for '{preset}' to {save_path}")
            else:
                self._hpo_provenance[preset] = {
                    "preset": preset,
                    "source": "fresh_hpo",
                    "dataset_id": ds_key,
                    "data_source_fingerprint": self._dataset_source_fingerprint(ds_key),
                    "tuning": tuning.model_dump(mode="json"),
                    "objective_metric": tuning.metric,
                    "search_space_fingerprint": _stable_json_sha256(raw_dict),
                }

        return self.hpo_configs

    def run(self) -> BenchmarkTable:
        """Run optional HPO followed by multi-seed evaluation."""
        self._write_meta()

        if self.config.run_hpo:
            self.tune_all()
        self._assert_uniform_hpo_provenance()

        jobs = []
        for preset in self.presets:
            cfg = self._effective_config_for_preset(preset)
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

        self._write_cell_index(results)
        self._write_meta()
        return BenchmarkTable(runs=results)

    def _write_meta(self) -> None:
        """Write bench_meta.json at run start (no-op when out_dir is not set)."""
        if self.out_dir is None or self._argv is None:
            return
        from seahorse.runner.artifacts import write_bench_meta
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
            data_manifest=self._write_data_manifest(),
            hpo_provenance=self._hpo_provenance or None,
            cell_index_path=None if self._cell_index_path is None else str(self._cell_index_path),
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
        return self._apply_benchmark_policy(preset, cfg)

    def _effective_config_for_preset(self, preset: str) -> STPPConfig:
        cfg = self.hpo_configs.get(preset)
        if cfg is None:
            cfg = self._base_config(preset)
        return self._apply_benchmark_policy(preset, cfg)

    def _locked_training_fields(self, preset: str) -> dict[str, Any]:
        cached = self._locked_training_cache.get(preset)
        if cached is not None:
            return cached
        cfg = STPPConfig.from_preset(preset)
        overrides = self._effective_base_overrides()
        if overrides:
            raw = cfg.model_dump(mode="json")
            deep_update(raw, overrides)
            cfg = STPPConfig(**raw)
        locked = {
            "n_epochs": cfg.training.n_epochs,
            "patience": cfg.training.patience,
        }
        self._locked_training_cache[preset] = locked
        return locked

    def _apply_benchmark_policy(self, preset: str, cfg: STPPConfig) -> STPPConfig:
        return self.config.apply_to_config(
            cfg,
            training_overrides=self._locked_training_fields(preset),
        )

    def _assert_locked_fields_not_tunable(self, raw_dict: dict[str, Any], preset: str) -> None:
        locked_paths = (
            "data.protocol",
            "data.normalize",
            "training.checkpoint_select",
            "training.test_nll_space",
            "training.predictive_test_nll_samples",
            "training.n_epochs",
            "training.patience",
        )
        for path in locked_paths:
            value = _nested_get(raw_dict, path)
            if _is_tunable_spec(value):
                raise ValueError(
                    f"Benchmark HPO does not allow tuning locked field '{path}' for preset {preset!r}."
                )

    def _apply_locked_policy_to_raw_dict(self, raw_dict: dict[str, Any], preset: str) -> None:
        locked_training = self._locked_training_fields(preset)
        locked_values = {
            "data.protocol": self.config.protocol,
            "data.normalize": self.config.normalize,
            "training.checkpoint_select": self.config.checkpoint_select,
            "training.test_nll_space": self.config.test_nll_space,
            "training.predictive_test_nll_samples": self.config.predictive_test_nll_samples,
            "training.n_epochs": locked_training["n_epochs"],
            "training.patience": locked_training["patience"],
        }
        for path, value in locked_values.items():
            _nested_set(raw_dict, path, value)

    def _loaded_hpo_provenance(self, preset: str, yaml_path: Path) -> dict[str, Any]:
        manifest_path = yaml_path.with_name(f"{preset}_hpo_manifest.json")
        trials_json = yaml_path.with_name(f"{preset}_trials.json")
        trials_csv = yaml_path.with_name(f"{preset}_trials.csv")
        provenance = {
            "preset": preset,
            "source": "pre_tuned_yaml",
            "best_config_path": str(yaml_path.resolve()),
            "best_config_sha256": _file_sha256(yaml_path),
            "manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else None,
            "trials_json_path": str(trials_json.resolve()) if trials_json.exists() else None,
            "trials_csv_path": str(trials_csv.resolve()) if trials_csv.exists() else None,
        }
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                provenance.setdefault("dataset_id", manifest.get("dataset_id"))
                provenance.setdefault(
                    "data_source_fingerprint",
                    manifest.get("data_source_fingerprint"),
                )
            except Exception:
                pass
        return provenance

    def _assert_uniform_hpo_provenance(self) -> None:
        sources = {
            (
                self._hpo_provenance.get(preset, {}).get("source")
                or ("base_yaml" if preset not in self.hpo_configs else "pre_tuned_yaml")
            )
            for preset in self.presets
        }
        if len(sources) > 1 and not self.config.allow_mixed_hpo_provenance:
            raise ValueError(
                "Benchmark run mixes HPO provenance sources across presets "
                f"({sorted(sources)}). Set BenchmarkConfig.allow_mixed_hpo_provenance=True "
                "only if you intend to compare mixed-provenance cells."
            )
        for preset in self.presets:
            self._hpo_provenance.setdefault(
                preset,
                {
                    "preset": preset,
                    "source": "base_yaml",
                },
            )

    def _write_data_manifest(self) -> dict[str, Any] | None:
        if self.data_manifest is None:
            return None
        if self.out_dir is None:
            return self.data_manifest
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "data_manifest.json"
        with open(path, "w") as f:
            json.dump(self.data_manifest, f, indent=2, default=str)
        self._data_manifest_path = path
        return self.data_manifest

    def _dataset_source_fingerprint(self, dataset_id: str) -> str | None:
        datasets = (self.data_manifest or {}).get("datasets", {})
        dataset_entry = datasets.get(dataset_id, {})
        return dataset_entry.get("source_fingerprint")

    def _write_cell_index(self, results: list[RunResult]) -> None:
        if self.out_dir is None:
            return
        rows = []
        for result in results:
            run_dir = None if result.run_dir is None else str(result.run_dir.resolve())
            rows.append(
                {
                    "preset": result.preset,
                    "dataset_id": result.dataset_id,
                    "seed": result.seed,
                    "run_dir": run_dir,
                    "run_result_path": None if run_dir is None else str((result.run_dir / "run_result.json").resolve()),
                    "artifacts_path": None if run_dir is None else str((result.run_dir / "artifacts.json").resolve()),
                    "resolved_config_path": None if run_dir is None else str((result.run_dir / "resolved_config.yaml").resolve()),
                    "checkpoint_path": None if result.checkpoint_path is None else str(result.checkpoint_path.resolve()),
                    "checkpoint_select": result.effective_config.get("training", {}).get("checkpoint_select"),
                    "nll_kind": result.nll_kind,
                    "nll_report_space": result.nll_report_space,
                    "test_nll_method": result.test_nll_method,
                    "test_nll_contexts": result.test_nll_contexts,
                    "test_nll_scored_contexts": result.test_nll_scored_contexts,
                    "test_nll_missing_contexts": result.test_nll_missing_contexts,
                    "native_test_nll": result.native_test_nll,
                    "hpo_source": self._hpo_provenance.get(result.preset, {}).get("source", "base_yaml"),
                    "hpo_manifest_path": self._hpo_provenance.get(result.preset, {}).get("manifest_path"),
                    "data_source_fingerprint": self._dataset_source_fingerprint(result.dataset_id),
                }
            )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._cell_index_path = self.out_dir / "cell_index.json"
        with open(self._cell_index_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)

    def _run_parallel(self, jobs, n_workers: int, backend: str) -> list[RunResult]:
        if backend == "ray":
            try:
                import ray
                from ray.remote_function import RemoteFunction as _
            except ImportError as exc:
                raise ImportError(
                    "backend='ray' requires Ray: pip install 'seahorse-stpp[hpo]'"
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
                "backend='joblib' requires joblib: pip install 'seahorse-stpp[runner]'"
            ) from exc

        return Parallel(n_jobs=n_workers)(delayed(_run_single)(*job) for job in jobs)
