"""Execute one frozen campaign cell, or inspect/advance its serial queue.

No command runs on import. `run`/`queue --execute` are the explicit launch
boundary. HPO opens train/validation only; evaluation is a separate stage and
never contributes to hyperparameter/checkpoint selection.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from seahorse.config import STPPConfig
from seahorse.data.prepared_protocol import _copy_unmarked, LazyPreparedSlidingWindowDataset
from seahorse.data.registry import DataBundle, DataRegistry
from seahorse.experiments.campaign import verify, write_json
from seahorse.experiments.coordinates import CoordinateTransform, transform_sequences
from seahorse.experiments.continuous_context import ContinuousPreparedContextDataset
from seahorse.models.configs import ConfigRegistry
from seahorse.runner.runner import STPPRunner, RunnerRegistry
from seahorse.training.data_module import STPPDataModule
from seahorse.utils import _canonicalize_sequence_record, deep_update
from seahorse.experiments.runtime import classify_exception, process_context, process_exit_context


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return clean_json(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_records(path, *, sha256=None, sample_history=None):
    """Read canonical JSONL while verifying its bytes; sample mode is smoke-only."""
    records, digest = [], hashlib.sha256()
    with Path(path).open("rb") as f:
        for line in f:
            digest.update(line)
            if line.strip():
                row = _canonicalize_sequence_record(json.loads(line))
                if sample_history is None:
                    records.append(row)
                elif len(row["times"]) > sample_history:
                    return [row]
    if sha256 is not None and digest.hexdigest() != sha256:
        raise ValueError(f"Dataset file hash differs from frozen manifest: {path}")
    return records


def select_indices(n, limit, seed):
    if n < 0 or limit < 1:
        raise ValueError("Invalid target population/budget.")
    return np.sort(np.random.default_rng(seed).choice(n, min(n, limit), replace=False)).tolist()


def _model_entry(manifest, job):
    return next(m for m in manifest["spec"]["models"] if m["id"] == job["model"])


def _coordinate_entry(manifest, job):
    route = manifest["datasets"][job["dataset"]]["dataset_id"]
    entry = manifest["coordinates"]["routes"][route]
    if (
        entry["train_split"]["sha256"]
        != manifest["datasets"][job["dataset"]]["split_sha256"]["train"]
    ):
        raise ValueError("Coordinate transform was fitted on a different training split.")
    return CoordinateTransform.from_dict(entry["transform"])


class TrainingAudit(pl.Callback):
    def __init__(self):
        self.targets = self.batches = 0
        self.started = None
        self.batch_seconds = []
        self.peak_allocated_gpu_bytes = 0

    def on_train_start(self, trainer, pl_module):
        self.started = time.monotonic()
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_start = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batches += 1
        self.targets += int((batch["lengths"] > 1).sum())
        self.batch_seconds.append(time.monotonic() - self._batch_start)
        if pl_module.device.type == "cuda":
            self.peak_allocated_gpu_bytes = max(
                self.peak_allocated_gpu_bytes,
                int(torch.cuda.max_memory_allocated(pl_module.device)),
            )

    def summary(self, runner):
        lm = runner._lightning_module
        return {
            "optimizer_steps": int(lm.global_step),
            "training_batches": self.batches,
            "target_presentations": self.targets,
            "median_train_batch_seconds": float(np.median(self.batch_seconds))
            if self.batch_seconds
            else None,
            # Lightning may move the module to CPU during fit teardown.
            "peak_allocated_gpu_bytes": self.peak_allocated_gpu_bytes,
        }


def make_config(manifest, job, out, *, lr_multiplier=1.0, smoke=False, device="cuda"):
    spec, model = manifest["spec"], _model_entry(manifest, job)
    cfg = STPPConfig.from_preset(model["preset"])
    cfg.data.protocol = "raw"
    cfg.data.normalize = False
    cfg.data.seed = int(job["seed"])
    cfg.data.batch_size = 2 if smoke else spec["fit"]["batch_size"]
    cfg.data.num_workers = 0
    cfg.model.n_marks = cfg.model.event_cov_dim = cfg.model.field_cov_dim = 0
    cfg.training.loss_target_scope = "last"
    cfg.training.lr *= float(lr_multiplier)
    cfg.training.device = device
    cfg.training.devices = 1
    cfg.training.precision = spec["fit"]["precision"]
    cfg.training.n_epochs = spec["fit"]["n_epochs"]
    cfg.training.max_steps = 1 if smoke else job["max_steps"]
    cfg.training.max_time_seconds = 120 if smoke else job["max_seconds"]
    cfg.training.val_check_interval_steps = 1 if smoke else spec["fit"]["validation_interval_steps"]
    cfg.training.patience = spec["fit"]["patience"]
    cfg.training.checkpoint_select = "best"
    cfg.training.test_nll_space = "raw"
    cfg.training.batch_size = cfg.data.batch_size
    cfg.logging.out_dir = str(out / "fit")
    cfg.logging.save_checkpoints = True
    overrides = model.get("config_overrides", {})
    values = cfg.model_dump(mode="python")
    deep_update(values, overrides)
    if model["preset"] in {"nsmpp", "dnsk"}:
        deep_update(
            values,
            {
                "model": {
                    "support_space_min": [0.0, 0.0],
                    "support_space_max": [1.0, 1.0],
                    "support_t0": 0.0,
                }
            },
        )
    cfg = STPPConfig.model_validate(values)
    return cfg


class IndependentContextDataset(LazyPreparedSlidingWindowDataset):
    """Synthetic realizations have independent initial conditions and never join."""

    def __init__(self, records, lookback):
        super().__init__(records, lookback=lookback)
        self.sequences = self.stat_sequences
        self._global_starts = np.cumsum([0, *[len(r["times"]) for r in records[:-1]]])

    def __getitem__(self, index):
        item = super().__getitem__(index)
        metadata = item["observation_metadata"]
        source = metadata["parent_source_sequence_index"]
        metadata["target_source_sequence_index"] = source
        metadata["target_global_event_index"] = (
            int(self._global_starts[source]) + metadata["target_parent_event_index"]
        )
        metadata["conditioning_reset"] = "independent simulated realization"
        return item


def context_dataset(records, manifest, job, coord, *, split):
    history = manifest["spec"]["protocol"]["history_events"]
    if manifest["datasets"][job["dataset"]].get("independent_windows"):
        return IndependentContextDataset(records, lookback=history)
    return ContinuousPreparedContextDataset(records, lookback=history, transform=coord, split=split)


def make_data(manifest, job, cfg, *, smoke=False):
    spec = manifest["spec"]
    route = manifest["datasets"][job["dataset"]]
    history = spec["protocol"]["history_events"]
    coord = _coordinate_entry(manifest, job)
    records = {}
    for split in ("train", "val"):
        records[split] = _copy_unmarked(
            list(
                transform_sequences(
                    read_records(
                        route["splits"][split],
                        sha256=route["split_sha256"][split],
                        sample_history=history if smoke else None,
                    ),
                    coord,
                )
            )
        )
    train = context_dataset(records["train"], manifest, job, coord, split="train")
    val_all = context_dataset(records["val"], manifest, job, coord, split="val")
    # Model-owned transforms must see each original train event once.
    train.sequences = train.stat_sequences
    val_all.sequences = val_all.stat_sequences
    if len(train) == 0 or len(val_all) == 0:
        raise ValueError("No eligible train/validation targets under the frozen history policy.")
    val_ids = select_indices(
        len(val_all),
        2 if smoke else spec["protocol"]["validation_targets"],
        spec["protocol"]["validation_target_seed"],
    )
    val = Subset(val_all, val_ids)
    if smoke:
        # Keep the complete sampled train source for transform fitting, while
        # bounding optimizer work to one batch of actual H+1 windows.
        train.indices_for_smoke = select_indices(len(train), 2, 0)
        sampler = [train.indices_for_smoke]
    else:
        sampler = None  # equal context-count batches; no model-specific event sampler
    collate = DataRegistry.get_collate(ConfigRegistry.get_collate_key(cfg.model.preset))
    dm = STPPDataModule(
        DataBundle(train, val, None, collate, sampler),
        batch_size=cfg.data.batch_size,
        num_workers=0,
        seed=job["seed"],
    )
    return dm, records, coord, val_ids


def selected_lr(manifest, job, campaign_dir):
    candidates = []
    for job_id in job["dependencies"]:
        path = campaign_dir / "jobs" / job_id / "status.json"
        if not path.exists():
            raise ValueError(f"HPO trial has not been attempted: {job_id}")
        status = json.loads(path.read_text())
        if status.get("state") not in {"completed", "failed", "timeout"}:
            raise ValueError(f"HPO trial is unfinished: {job_id}")
        value = status.get("validation_objective")
        if status["state"] == "completed" and value is not None and math.isfinite(value):
            trial = next(j for j in manifest["jobs"] if j["id"] == job_id)
            candidates.append((value, trial["trial"], trial["lr_multiplier"], job_id))
    if not candidates:
        raise ValueError("No successful HPO trial; record failed refit cell.")
    best = min(candidates)
    return best[2], best[3]


def fit_one(
    manifest, job, campaign_dir, *, smoke=False, device="cuda", retry=False, retry_reason=None
):
    if smoke:
        return _fit_one(manifest, job, campaign_dir, smoke=True, device=device)
    from seahorse.experiments.attempts import claim_job_phase

    out = campaign_dir / "jobs" / job["id"]
    with claim_job_phase(
        out, phase="fit", device=device, retry=retry, retry_reason=retry_reason
    ) as claim:
        result = _fit_one(manifest, job, campaign_dir, device=device, claim=claim)
    return result


def _fit_one(manifest, job, campaign_dir, *, smoke=False, device="cuda", claim=None):
    if ConfigRegistry.runner_kind(_model_entry(manifest, job)["preset"]) == "statstpp":
        from seahorse.experiments.campaign_statistical import fit_statistical

        return fit_statistical(manifest, job, campaign_dir, smoke=smoke, claim=claim)
    out = campaign_dir / ("preflight" if smoke else "jobs") / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    if not smoke and status_path.exists():
        raise FileExistsError(
            "Job already has an attempt. Preserve it; use explicit versioned retry policy."
        )
    status = {
        "job_id": job["id"],
        "state": "running",
        "smoke_only": smoke,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest["sha256"],
        "runner": "nnstpp",
        "process": process_context(device=device),
        "attempt": claim,
        "stdout_path": str(out / "stdout.log"),
        "traceback_path": str(out / "traceback.txt"),
    }
    if claim and claim.get("retry_of"):
        status["retry_of"] = claim["retry_of"]
    write_json(status_path, status)
    start = time.monotonic()
    try:
        chosen_trial = None
        multiplier = job.get("lr_multiplier", 1.0)
        if job["action"] == "refit" and not smoke:
            multiplier, chosen_trial = selected_lr(manifest, job, campaign_dir)
        cfg = make_config(manifest, job, out, lr_multiplier=multiplier, smoke=smoke, device=device)
        if claim and claim.get("retry_of"):
            checkpoints = sorted((out / "fit").rglob("last.ckpt"), key=lambda p: p.stat().st_mtime)
            if checkpoints:
                cfg.training.resume_from_checkpoint = str(checkpoints[-1])
                status["resume_checkpoint"] = str(checkpoints[-1])
                status["resume_policy"] = (
                    "Lightning optimizer/step/callback state; shuffled mid-epoch data order is not guaranteed bitwise identical"
                )
        dm, records, coord, val_ids = make_data(manifest, job, cfg, smoke=smoke)
        write_json(
            out / "input_contract.json",
            {
                "coordinate_transform": coord.to_dict(),
                "validation_target_indices": val_ids,
                "train_target_population": len(dm.train_dataset),
                "filter_counts": dm.train_dataset.sequence_filter_counts,
                "validation_target_policy": manifest["spec"]["protocol"],
                "sample_only": smoke,
            },
        )
        runner = RunnerRegistry.create(cfg)
        audit = TrainingAudit()
        result = runner.fit(
            records["train"],
            records["val"],
            None,
            data_module=dm,
            dataset_id=job["dataset"],
            extra_callbacks=[audit],
        )
        if not math.isfinite(result.val_objective):
            raise ValueError(
                "No finite validation checkpoint before stopping; this fit cannot be selected."
            )
        saved = runner.save(out / "model")
        # Check that the final artifact can actually be reconstructed.
        restored = STPPRunner.load(saved)
        for key, value in runner.model.state_dict().items():
            if not torch.equal(value.cpu(), restored.model.state_dict()[key].cpu()):
                raise ValueError(f"Checkpoint reload mismatch: {key}")
        if restored.norm_stats != runner.norm_stats:
            raise ValueError("Checkpoint lost runner preprocessing settings.")
        # A state_dict comparison cannot detect lost non-buffer transform or
        # configuration values. Compare an actual common-context score too.
        parity_batch = next(iter(dm.val_dataloader()))
        parity_inputs = {k: parity_batch[k].to(device) for k in ("times", "locations", "lengths")}
        parity = []
        for candidate in (runner, restored):
            candidate.model.to(device).eval()
            cuda_devices = [torch.cuda.current_device()] if str(device).startswith("cuda") else []
            with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                torch.manual_seed(20260910)
                parity.append(
                    candidate.model.eval_forward(**parity_inputs, target_scope="last")["nll"]
                    .detach()
                    .cpu()
                )
        torch.testing.assert_close(parity[0], parity[1], rtol=1e-5, atol=1e-5)
        elapsed = time.monotonic() - start
        status.update(
            state="completed",
            validation_objective=result.val_objective,
            validation_metric=result.val_metric_key,
            chosen_hpo_trial=chosen_trial,
            lr_multiplier=multiplier,
            checkpoint=str(saved),
            elapsed_seconds=elapsed,
            training_audit=audit.summary(runner),
            checkpoint_semantic_parity="seeded_validation_objective_and_preprocessing",
            budget_limited=(
                cfg.training.max_time_seconds is not None
                and elapsed >= 0.95 * cfg.training.max_time_seconds
            )
            or (
                cfg.training.max_steps > 0
                and audit.summary(runner)["optimizer_steps"] >= cfg.training.max_steps
            ),
            process=process_context(device=device),
        )
    except Exception as exc:
        status.update(
            state="timeout" if isinstance(exc, TimeoutError) else "failed",
            **classify_exception(exc, origin="fit"),
            elapsed_seconds=time.monotonic() - start,
            process=process_context(device=device),
        )
        (out / "traceback.txt").write_text(traceback.format_exc())
    write_json(status_path, clean_json(status))
    return status


def evaluate_one(manifest, job, campaign_dir, *, device="cuda"):
    if ConfigRegistry.runner_kind(_model_entry(manifest, job)["preset"]) == "statstpp":
        from seahorse.experiments.campaign_statistical import evaluate_statistical

        return evaluate_statistical(manifest, job, campaign_dir)
    from seahorse.evaluation.likelihood import _reporting_correction
    from seahorse.runner.results import resolve_loss_result_reporting

    out = campaign_dir / "jobs" / job["id"]
    status = json.loads((out / "status.json").read_text())
    if status["state"] != "completed" or job["action"] != "refit":
        raise ValueError(
            "Test evaluation requires a completed final-seed refit, never an HPO trial."
        )
    if (out / "evaluation.json").exists():
        raise FileExistsError("Test evaluation already recorded.")
    runner = STPPRunner.load(status["checkpoint"])
    runner.model.to(device).eval()
    model_entry = _model_entry(manifest, job)
    route = manifest["datasets"][job["dataset"]]
    coord = _coordinate_entry(manifest, job)
    protocol = manifest["spec"]["protocol"]
    from seahorse.experiments.campaign_progress import NLLProgress, checkpoint_fingerprint
    from seahorse.experiments.campaign import digest_file

    binding = {
        "manifest_sha256": manifest["sha256"],
        "job_id": job["id"],
        "checkpoint_sha256": checkpoint_fingerprint(status["checkpoint"]),
        "test_split_sha256": route["split_sha256"]["test"],
        "batch_size": protocol.get("evaluation_batch_size", manifest["spec"]["fit"]["batch_size"]),
        "seed": protocol.get("evaluation_seed", 20260916),
    }
    cached_nll = (
        json.loads((out / "nll_status.json").read_text())
        if (out / "nll_status.json").exists()
        else None
    )
    if cached_nll and (
        cached_nll.get("evaluation_binding") != binding
        or cached_nll.get("artifact_sha256") != digest_file(out / "target_nll.npz")
    ):
        raise ValueError("Saved NLL receipt is not bound to this checkpoint and target artifact.")
    comparable_nll = model_entry["nll_path"] not in {"none", "bound", "surrogate"}
    collate = DataRegistry.get_collate(ConfigRegistry.get_collate_key(runner.config.model.preset))
    numerical = {
        "status": "not_applicable",
        "reason": "closed-form likelihood or sampling-only family",
    }
    if cached_nll:
        numerical = cached_nll["numerical_validation"]
    if not cached_nll and comparable_nll and model_entry["nll_path"] != "closed_form":
        torch.manual_seed(binding["seed"])
        from seahorse.experiments.numerical_checks import run_numerical_nll_checks

        settings = protocol["numerical_validation"]
        val_records = _copy_unmarked(
            list(
                transform_sequences(
                    read_records(route["splits"]["val"], sha256=route["split_sha256"]["val"]), coord
                )
            )
        )
        val_data = context_dataset(val_records, manifest, job, coord, split="val")
        ids = select_indices(len(val_data), settings["targets"], settings["seed"])
        checks = []
        for first in range(0, len(ids), settings["batch_size"]):
            batch = collate([val_data[i] for i in ids[first : first + settings["batch_size"]]])
            checks.append(
                run_numerical_nll_checks(
                    runner,
                    batch,
                    {
                        "absolute_tolerance": settings["absolute_tolerance"],
                        "relative_tolerance": settings["relative_tolerance"],
                    },
                )
            )
        numerical = {
            "status": "pass" if checks and all(c["status"] == "pass" for c in checks) else "fail",
            "validation_target_indices": ids,
            "checks": checks,
            "scope": "fixed validation subset; empirical numerical accuracy check, not a global error bound",
        }
        write_json(out / "numerical_validation.json", numerical)
        del val_records, val_data
    score_nll = comparable_nll and numerical["status"] in {"pass", "not_applicable"}
    records = _copy_unmarked(
        list(
            transform_sequences(
                read_records(route["splits"]["test"], sha256=route["split_sha256"]["test"]), coord
            )
        )
    )
    dataset = context_dataset(records, manifest, job, coord, split="test")
    if cached_nll:
        from seahorse.experiments.campaign_forecast import evaluate_forecasts

        metrics = dict(cached_nll)
        metrics.update(
            evaluate_forecasts(runner, dataset, coord, manifest, job, out, device=device)
        )
        write_json(out / "evaluation.json", clean_json(metrics))
        return metrics
    progress = (
        NLLProgress(out / "nll_progress", binding=binding, size=len(dataset)) if score_nll else None
    )
    values = progress.arrays["nll"] if progress else np.empty(0, np.float64)
    sequence_ids = progress.arrays["sequence_index"] if progress else np.empty(0, np.int64)
    event_ids = progress.arrays["target_event_index"] if progress else np.empty(0, np.int64)
    errors = progress.errors if progress else []
    position = progress.position if progress else 0
    loader = DataLoader(
        Subset(dataset, range(position, len(dataset))),
        batch_size=binding["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )
    batches_since_commit = 0
    for batch in loader if score_nll else []:
        size = len(batch["lengths"])
        metadata = batch["observation_metadata"]
        sequence_ids[position : position + size] = [
            m["target_source_sequence_index"] for m in metadata
        ]
        event_ids[position : position + size] = [m["target_global_event_index"] for m in metadata]
        if score_nll:
            try:
                kwargs = {k: batch[k].to(device) for k in ("times", "locations", "lengths")}
                cuda_devices = (
                    [torch.cuda.current_device()] if str(device).startswith("cuda") else []
                )
                with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                    torch.manual_seed(binding["seed"] + position)
                    output = runner.model.eval_forward(**kwargs, target_scope="last")
                    result = runner.model.compute_loss(output)
                reported, _, _, _, space = resolve_loss_result_reporting(
                    result, requested_space="raw"
                )
                correction = _reporting_correction(result, reported, space)
                # Families with no internal transform already score common input
                # coordinates. Other families must expose their exact Jacobian.
                state_transform = getattr(runner.model.state_model, "_input_transform_spec", {})
                if space != "raw" and state_transform and state_transform.get("type") != "identity":
                    raise ValueError(
                        "Missing family coordinate correction for raw-space comparison."
                    )
                per_target = output["nll_per_event"].detach().cpu().numpy().reshape(-1)
                if len(per_target) != size:
                    raise ValueError("Per-target score count differs from common context count.")
                values[position : position + size] = (
                    per_target + correction + coord.log_raw_volume_per_normalized_volume
                )
            except (TimeoutError, MemoryError, torch.cuda.OutOfMemoryError):
                raise
            except Exception as exc:
                errors.append(
                    {
                        "start_target": position,
                        "count": size,
                        "error": str(exc),
                        "type": type(exc).__name__,
                    }
                )
        position += size
        batches_since_commit += 1
        if batches_since_commit >= 32:
            progress.commit(position)
            batches_since_commit = 0
    if progress:
        progress.commit(position)
        artifact_hash = progress.finalize(out / "target_nll.npz")
    else:
        np.savez_compressed(
            out / "target_nll.npz",
            nll=values,
            sequence_index=sequence_ids,
            target_event_index=event_ids,
        )
        artifact_hash = digest_file(out / "target_nll.npz")
    finite = np.isfinite(values)
    metrics = {
        "nll_path": model_entry["nll_path"],
        "target_population": len(dataset),
        "finite_targets": int(finite.sum()) if comparable_nll else None,
        "failed_targets": int(len(dataset) - finite.sum()) if comparable_nll else None,
        "raw_next_event_nll": float(values.mean())
        if score_nll and len(values) and finite.all()
        else None,
        "nll_status": (
            "complete"
            if score_nll and len(values) and finite.all()
            else "numerical_validation_failed"
            if comparable_nll and not score_nll
            else "failed"
            if comparable_nll
            else "unavailable_by_design"
        ),
        "failure_batches": errors,
        "filter_counts": dataset.sequence_filter_counts,
        "numerical_validation": numerical,
    }
    write_json(
        out / "nll_status.json",
        clean_json(
            {
                **metrics,
                "evaluation_binding": binding,
                "artifact_sha256": artifact_hash,
                "manifest_sha256": manifest["sha256"],
                "job_id": job["id"],
            }
        ),
    )
    from seahorse.experiments.campaign_forecast import evaluate_forecasts

    metrics.update(evaluate_forecasts(runner, dataset, coord, manifest, job, out, device=device))
    write_json(out / "evaluation.json", clean_json(metrics))
    return metrics


def evaluate_attempt(manifest, job, campaign_dir, *, device="cuda", retry=False, retry_reason=None):
    from seahorse.experiments.attempts import claim_job_phase

    out = campaign_dir / "jobs" / job["id"]
    with claim_job_phase(
        out, phase="evaluate", device=device, retry=retry, retry_reason=retry_reason
    ) as claim:
        return _evaluate_attempt(manifest, job, campaign_dir, device=device, claim=claim)


def _evaluate_attempt(manifest, job, campaign_dir, *, device="cuda", claim=None):
    out = campaign_dir / "jobs" / job["id"]
    path = out / "evaluation_status.json"
    if path.exists():
        raise FileExistsError(
            "Evaluation was already attempted; retain it and use a versioned retry."
        )
    started = time.monotonic()
    status = {
        "job_id": job["id"],
        "state": "running",
        "manifest_sha256": manifest["sha256"],
        "runner": job.get("runner", "nnstpp"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": claim,
        "process": process_context(device=device),
        "stdout_path": str(out / "evaluation_stdout.log"),
        "traceback_path": str(out / "evaluation_traceback.txt"),
    }
    if claim and claim.get("retry_of"):
        status["retry_of"] = claim["retry_of"]
    write_json(path, status)
    try:
        result = evaluate_one(manifest, job, campaign_dir, device=device)
        status.update(state="completed", nll_status=result["nll_status"])
    except Exception as exc:
        status.update(
            state="timeout" if isinstance(exc, TimeoutError) else "failed",
            **classify_exception(exc, origin="evaluate"),
        )
        (out / "evaluation_traceback.txt").write_text(traceback.format_exc())
    status["elapsed_seconds"] = time.monotonic() - started
    status["process"] = process_context(device=device)
    write_json(path, status)
    return status


def _launch_evidence(manifest, directory, *, runner="all"):
    from seahorse.experiments.campaign import canonical_hash
    from seahorse.experiments.campaign_preflight import check_runtime

    path = directory / "launch_readiness.json"
    if not path.exists():
        raise ValueError("Missing source-bound launch_readiness.json from integration checks.")
    import fcntl

    with (directory / "launch_readiness.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        ready = json.loads(path.read_text())
    if ready.get("sha256") != canonical_hash({k: v for k, v in ready.items() if k != "sha256"}):
        raise ValueError("Launch evidence is incomplete or has changed.")
    check_runtime(ready, runner=runner)
    return ready


def _dataset_gate(readiness, dataset, runner, model=None):
    evidence = readiness.get("datasets", {}).get(dataset, {})
    probes = evidence.get("runner_smokes")
    if probes is not None:
        probe = probes.get(runner, {})
        if model is not None and "model_checks" in probe:
            return probe["model_checks"].get(model, {}).get("status") == "pass"
        return probe.get("status") == "pass"
    # Compatibility with old neural-only evidence; it cannot authorize stat.
    return runner == "nnstpp" and evidence.get("status") == "pass"


def _queue_usage(directory, runner="all"):
    # Count actual queue subprocess wall time, including data loading and
    # evaluation. Manually executed cells also count, without double counting.
    seconds = 0.0
    for path in (directory / "jobs").glob("*/status.json"):
        status = json.loads(path.read_text())
        if runner == "all" or status.get("runner", "nnstpp") == runner:
            seconds += float(status.get("elapsed_seconds", 0.0))
    for path in (directory / "jobs").glob("*/evaluation_status.json"):
        status = json.loads(path.read_text())
        if runner == "all" or status.get("runner", "nnstpp") == runner:
            seconds += float(status.get("elapsed_seconds", 0.0))
    return seconds / 3600


def _queue_model_selection(models):
    """Return a canonical explicit model selection, or ``None`` for all."""
    if models is None:
        return None
    if isinstance(models, str):
        raise TypeError("models must be an iterable of model identifiers, not a string.")
    supplied = tuple(models)
    if not supplied or any(not isinstance(model, str) or not model for model in supplied):
        raise ValueError("--models requires one or more nonempty model identifiers.")
    return tuple(sorted(set(supplied)))


def _queue_selection_suffix(models) -> str:
    """Stable filename suffix for a model subset; preserve legacy all-model names."""
    selected = _queue_model_selection(models)
    if selected is None:
        return ""
    digest = hashlib.sha256("\x1f".join(selected).encode()).hexdigest()[:12]
    return f"_models_{digest}"


def _queue_model_lock_name(model: str) -> str:
    # IDs in published specs are already simple, but retain a readable safe
    # prefix and a full digest so a future ID cannot escape the campaign dir.
    safe = (
        "".join(char if char.isalnum() or char in "_.-" else "_" for char in model)[:48] or "model"
    )
    return f"queue_model_{safe}_{hashlib.sha256(model.encode()).hexdigest()}.lock"


def queue(
    manifest_path,
    *,
    execute=False,
    stage="core",
    max_jobs=None,
    phase="all",
    runner="all",
    models=None,
):
    if runner not in {"all", "nnstpp", "statstpp"}:
        raise ValueError("runner must be all, nnstpp or statstpp.")
    manifest = verify(manifest_path)
    directory = Path(manifest_path).resolve().parent
    selected_models = _queue_model_selection(models)
    known_models = {entry["id"] for entry in manifest["spec"]["models"]}
    unknown = set(selected_models or ()) - known_models
    if unknown:
        raise ValueError(f"Unknown campaign model selection: {sorted(unknown)}")
    suffix = _queue_selection_suffix(selected_models)
    jobs = [
        j
        for j in manifest["jobs"]
        if (stage == "all" or j["stage"] == stage)
        and (runner == "all" or j.get("runner", "nnstpp") == runner)
        and (selected_models is None or j["model"] in selected_models)
    ]
    pending = []
    for job in jobs:
        out = directory / "jobs" / job["id"]
        if phase in {"all", "fit"} and not (out / "status.json").exists():
            pending.append((job, "run", job["max_seconds"]))
        if (
            phase in {"all", "evaluate"}
            and job["action"] == "refit"
            and not (out / "evaluation_status.json").exists()
            and not (out / "evaluation.json").exists()
        ):
            pending.append((job, "evaluate", manifest["spec"]["compute"]["evaluation_max_seconds"]))
    if not execute:
        return {
            "state": "dry_run",
            "phase": phase,
            "runner": runner,
            "models": list(selected_models) if selected_models else None,
            "pending_actions": len(pending),
            "fit_actions": sum(a == "run" for _, a, _ in pending),
            "evaluation_actions": sum(a == "evaluate" for _, a, _ in pending),
            "next_jobs": [{"id": j["id"], "action": a} for j, a, _ in pending[:10]],
            "evaluation_dependency": "completed refit; HPO checkpoints are never test-evaluated",
        }
    from seahorse.experiments.campaign_preflight import model_gate

    ready = _launch_evidence(manifest, directory, runner=runner)
    executed = 0
    compute = manifest["spec"]["compute"]
    blocked = []
    reason = "queue_exhausted"

    def record(state, **extra):
        value = {
            "state": state,
            "utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "manifest_sha256": manifest["sha256"],
            "executed": executed,
            "blocked_cells": blocked,
            "runner": runner,
            "models": list(selected_models) if selected_models else None,
            "accounted_gpu_process_hours": _queue_usage(directory, "nnstpp"),
            "accounted_cpu_process_hours": _queue_usage(directory, "statstpp"),
            **extra,
        }
        write_json(directory / f"queue_state_{runner}{suffix}.json", value)
        with (directory / f"queue_events_{runner}{suffix}.jsonl").open("a") as f:
            f.write(json.dumps(value, sort_keys=True) + "\n")
        return value

    # flock prevents two queue invocations from competing for the same H100.
    import fcntl

    with contextlib.ExitStack() as locks:
        readiness_lock = locks.enter_context((directory / "launch_readiness.lock").open("a"))
        fcntl.flock(readiness_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        # Neural jobs retain one shared queue lock because the campaign uses
        # one shared GPU. Statistical subsets are isolated by model locks
        # below so independent CPU groups can advance concurrently.
        selected_kinds = {job.get("runner", "nnstpp") for job in jobs}
        if "nnstpp" in selected_kinds:
            lock = locks.enter_context((directory / "queue_nnstpp.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A subset queue owns every selected model.  Disjoint statistical
        # subsets can proceed concurrently; overlap is refused immediately.
        # The runner lock above remains the shared-GPU gate for neural jobs.
        for model in selected_models or tuple(sorted({job["model"] for job in jobs})):
            lock = locks.enter_context((directory / _queue_model_lock_name(model)).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job, action, max_seconds in pending:
            if max_jobs is not None and executed >= max_jobs:
                reason = "requested_action_limit"
                break
            if compute.get("stop_before_utc") and datetime.now(
                timezone.utc
            ) >= datetime.fromisoformat(compute["stop_before_utc"]):
                reason = "deadline_reached"
                break
            kind = job.get("runner", "nnstpp")
            limit = (
                compute["max_gpu_hours"] if kind == "nnstpp" else compute.get("max_cpu_hours", 48)
            )
            watchdog_seconds = (
                None
                if max_seconds is None
                else max_seconds
                + (compute.get("fit_overhead_seconds", 900) if action == "run" else 0)
            )
            if (
                limit is not None
                and _queue_usage(directory, kind) + (watchdog_seconds or 0) / 3600 > limit
            ):
                reason = "compute_budget_exhausted"
                break
            gate = model_gate(manifest, ready, job["model"])
            if not _dataset_gate(ready, job["dataset"], kind, job["model"]):
                gate = "dataset_integration_check_missing"
            if gate:
                blocked.append({"job": job["id"], "action": action, "reason": gate})
                continue
            if action == "evaluate":
                path = directory / "jobs" / job["id"] / "status.json"
                if not path.exists() or json.loads(path.read_text()).get("state") != "completed":
                    blocked.append(
                        {"job": job["id"], "action": action, "reason": "refit_not_completed"}
                    )
                    continue
            if kind == "nnstpp":
                free = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    text=True,
                )
                if int(free.strip().splitlines()[0]) < compute["minimum_free_gpu_gib"] * 1024:
                    return record(
                        "waiting_for_shared_gpu_memory",
                        free_gpu_mib=int(free.strip().splitlines()[0]),
                    )
            job_out = directory / "jobs" / job["id"]
            job_out.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "seahorse",
                "campaign",
                action,
                "--manifest",
                str(manifest_path),
                "--job",
                job["id"],
                "--device",
                "cuda" if kind == "nnstpp" else "cpu",
            ]
            record("running", job_id=job["id"], action=action)
            started = time.monotonic()
            status_path = job_out / ("status.json" if action == "run" else "evaluation_status.json")
            with (job_out / ("stdout.log" if action == "run" else "evaluation_stdout.log")).open(
                "w"
            ) as log:
                try:
                    # Fit timeout includes a bounded checkpoint/data-loading
                    # allowance. Actual elapsed process time counts in budget.
                    proc = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=watchdog_seconds,
                        check=False,
                    )
                    status = (
                        json.loads(status_path.read_text())
                        if status_path.exists()
                        else {
                            "job_id": job["id"],
                            "state": "failed",
                            "error": "Process exited without a status artifact.",
                        }
                    )
                    if status.get("state") == "running":
                        status.update(
                            state="failed", error="Process exited before finalizing its status."
                        )
                    status.update(
                        process_returncode=proc.returncode,
                        returncode=proc.returncode,
                        elapsed_seconds=time.monotonic() - started,
                        runner=kind,
                    )
                    if proc.returncode and not status.get("failure_class"):
                        status.update(process_exit_context(proc.returncode))
                    write_json(status_path, status)
                except subprocess.TimeoutExpired:
                    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
                    write_json(
                        status_path,
                        {
                            **previous,
                            "job_id": job["id"],
                            "state": "timeout",
                            "runner": kind,
                            "failure_class": "timeout",
                            "failure_origin": "queue_watchdog",
                            "wall_limit_seconds": watchdog_seconds,
                            "error": "Outer process watchdog expired.",
                            "manifest_sha256": manifest["sha256"],
                            "elapsed_seconds": time.monotonic() - started,
                        },
                    )
            executed += 1
        return record(reason)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("queue", "run", "evaluate", "smoke"))
    p.add_argument("--manifest", required=True)
    p.add_argument("--job")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--stage", choices=("core", "expansion", "all"), default="core")
    p.add_argument("--phase", choices=("fit", "evaluate", "all"), default="all")
    p.add_argument("--runner", choices=("all", "nnstpp", "statstpp"), default="all")
    p.add_argument("--max-jobs", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--retry", action="store_true")
    p.add_argument("--retry-reason")
    args = p.parse_args(argv)
    if args.command == "queue":
        result = queue(
            args.manifest,
            execute=args.execute,
            stage=args.stage,
            max_jobs=args.max_jobs,
            phase=args.phase,
            runner=args.runner,
            models=args.models,
        )
    else:
        if not args.job:
            p.error("--job is required")
        manifest = verify(args.manifest)
        job = next(j for j in manifest["jobs"] if j["id"] == args.job)
        torch.set_num_threads(manifest["spec"]["compute"]["cpu_threads_per_job"])
        directory = Path(args.manifest).resolve().parent
        if args.command != "smoke":
            from seahorse.experiments.campaign_preflight import model_gate

            readiness = _launch_evidence(manifest, directory, runner=job.get("runner", "nnstpp"))
            problem = model_gate(manifest, readiness, job["model"])
            if problem:
                raise ValueError(problem)
            if not _dataset_gate(
                readiness, job["dataset"], job.get("runner", "nnstpp"), job["model"]
            ):
                raise ValueError("Dataset integration check is missing.")
        from seahorse.experiments.runtime import wall_time_limit

        seconds = (
            manifest["spec"]["compute"]["evaluation_max_seconds"]
            if args.command == "evaluate"
            else (120 if args.command == "smoke" else job["max_seconds"])
        )
        if seconds is not None and args.command != "evaluate":
            seconds += manifest["spec"]["compute"].get("fit_overhead_seconds", 900)
        with wall_time_limit(seconds):
            options = {
                "device": args.device,
                "retry": args.retry,
                "retry_reason": args.retry_reason,
            }
            result = (
                evaluate_attempt(manifest, job, directory, **options)
                if args.command == "evaluate"
                else fit_one(manifest, job, directory, smoke=args.command == "smoke", **options)
            )
    print(json.dumps(clean_json(result), indent=2))
    if result.get("state") in {"failed", "timeout"}:
        sys.exit(1)


if __name__ == "__main__":
    main()
