"""Full-data, full-batch validation rehearsal through the public runner.

These bounded fits are diagnostic artifacts, never HPO candidates or paper
results.  They read training/validation only and retain the production history,
batch size, precision, model dimensions and numerical tolerances.
"""

from __future__ import annotations

import gc
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from seahorse.experiments.campaign import verify, write_json
from seahorse.experiments.runtime import classify_exception, process_context


def rehearse(manifest_path, *, out, dataset, models, steps=3, device="cuda"):
    from seahorse.experiments.campaign_run import make_config, make_data, TrainingAudit, clean_json
    from seahorse.runner.runner import RunnerRegistry, STPPRunner
    from seahorse.evaluation.forecast import forecast_next, forecast_horizon
    from seahorse.experiments.campaign_forecast import _input_history, _count_horizon
    from seahorse.experiments.campaign_statistical import raw_context

    if steps < 1:
        raise ValueError("Rehearsal steps must be positive.")
    manifest = verify(manifest_path)
    out = Path(out)
    if (out / "rehearsal.json").exists():
        raise FileExistsError("Preserve the previous rehearsal; choose a new output directory.")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(manifest["spec"]["compute"]["cpu_threads_per_job"])
    rows = []
    for model_id in models:
        job = next(
            j
            for j in manifest["jobs"]
            if j["dataset"] == dataset
            and j["model"] == model_id
            and j["action"] == "hpo"
            and j["trial"] == 0
        )
        if job.get("runner", "nnstpp") != "nnstpp":
            raise ValueError(
                "The full-batch rehearsal applies to neural models; statistical preflight uses native fitting."
            )
        root = out / model_id
        row = {"model": model_id, "dataset": dataset, "status": "running", "diagnostic_only": True}
        started = time.monotonic()
        try:
            cfg = make_config(manifest, job, root, device=device)
            cfg.training.max_steps = steps
            cfg.training.max_time_seconds = None
            cfg.training.val_check_interval_steps = steps
            before_data = time.monotonic()
            dm, records, coord, val_ids = make_data(manifest, job, cfg)
            row.update(
                data_load_seconds=time.monotonic() - before_data,
                train_targets=len(dm.train_dataset),
                validation_targets=len(val_ids),
                batch_size=cfg.data.batch_size,
                history_events=manifest["spec"]["protocol"]["history_events"],
            )
            runner = RunnerRegistry.create(cfg)
            audit = TrainingAudit()
            fitted = runner.fit(
                records["train"],
                records["val"],
                None,
                data_module=dm,
                dataset_id=dataset,
                extra_callbacks=[audit],
            )
            if not np.isfinite(fitted.val_objective):
                raise FloatingPointError(
                    "Rehearsal did not produce a finite validation checkpoint."
                )
            checkpoint = runner.save(root / "model")
            row.update(
                checkpoint=str(checkpoint),
                training_audit=audit.summary(runner),
                validation_objective=fitted.val_objective,
            )
            restored = STPPRunner.load(checkpoint)
            restored.model.to(device).eval()
            validation_loader = dm.val_dataloader()
            batch = next(iter(validation_loader))
            inputs = {k: batch[k].to(device) for k in ("times", "locations", "lengths")}
            elapsed = []
            with torch.no_grad():
                for _ in range(3):
                    start = time.perf_counter()
                    result = restored.model.eval_forward(**inputs, target_scope="last")
                    loss = restored.model.compute_loss(result).loss
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize()
                    elapsed.append(time.perf_counter() - start)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite fitted validation objective.")
            row["fitted_validation_batch_seconds"] = elapsed
            validation = validation_loader.dataset.dataset
            index = val_ids[0]
            context = raw_context(validation, index, coord)
            history = _input_history(context, validation, index, coord, False)
            start = time.monotonic()
            next_draw = forecast_next(restored, history, n_samples=64, seed=20260916, device=device)
            row["next_forecast"] = {
                "seconds": time.monotonic() - start,
                "status": next_draw["status"],
                "valid_draws": int(next_draw["valid"].sum()) if "valid" in next_draw else None,
                "draws": 64,
            }
            if next_draw["status"] != "unsupported":
                horizon = (
                    _count_horizon(coord, manifest["spec"]["protocol"]["count_forecast"])
                    / coord.time_scale
                )
                start = time.monotonic()
                forecast = forecast_horizon(
                    restored,
                    history,
                    t_end=float(np.asarray(history["times"])[-1]) + horizon,
                    n_samples=4,
                    seed=20260916,
                    device=device,
                    history_limit=manifest["spec"]["protocol"]["history_events"],
                    event_cap=1000,
                )
                row["count_forecast"] = {
                    "seconds": time.monotonic() - start,
                    "status": forecast["status"],
                    "draws": 4,
                    "path_statuses": list(forecast.get("sample_status", [])),
                }
            row["status"] = "pass"
            del restored, runner, dm, records, batch, inputs, result, loss, validation
        except Exception as exc:
            row.update(
                status="fail",
                **classify_exception(exc, origin="rehearsal"),
                traceback=traceback.format_exc(),
            )
        row.update(
            elapsed_seconds=time.monotonic() - started, process=process_context(device=device)
        )
        rows.append(row)
        write_json(
            out / "rehearsal.json",
            clean_json(
                {
                    "manifest_sha256": manifest["sha256"],
                    "source_sha256": manifest["source"]["sha256"],
                    "scope": "bounded training and validation only; no test metrics or convergence claims",
                    "rows": rows,
                }
            ),
        )
        print(clean_json(row), flush=True)
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return {
        "pass": all(r["status"] == "pass" for r in rows),
        "rows": rows,
        "path": str(out / "rehearsal.json"),
    }


def add_subparser(commands):
    p = commands.add_parser(
        "rehearse", help="Full-data, production-batch, validation-only rehearsal"
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--device", default="cuda")
