from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from unified_stpp.config import STPPConfig
from unified_stpp.runner.results import RunResult


SAMPLE_SEQUENCES = [
    {
        "times": [0.10, 0.32, 0.58, 0.91, 1.23, 1.51, 1.88],
        "locations": [
            [0.10, 0.20],
            [0.22, 0.18],
            [0.35, 0.44],
            [0.52, 0.49],
            [0.68, 0.61],
            [0.74, 0.77],
            [0.88, 0.91],
        ],
    }
]


MINI_OVERRIDES = {
    "auto_stpp": [
        "model.hidden_dim=16",
        "model.decoder.lookback=3",
        "model.decoder.max_history=3",
        "model.decoder.n_prodnet=1",
        "model.decoder.hidden_size=16",
        "model.decoder.num_layers=1",
    ],
    "deep_stpp": [
        "model.hidden_dim=16",
        "model.encoder.num_heads=1",
        "model.encoder.num_layers=1",
        "model.updater.num_heads=1",
        "model.decoder.seq_len=3",
        "model.decoder.num_points=4",
        "model.decoder.n_layers=1",
    ],
    "smash": [
        "model.hidden_dim=16",
        "model.encoder.d_model=16",
        "model.encoder.d_rnn=16",
        "model.encoder.d_inner=16",
        "model.encoder.n_layers=1",
        "model.encoder.n_head=1",
        "model.encoder.d_k=8",
        "model.encoder.d_v=8",
        "model.decoder.num_noise=2",
        "model.decoder.samplingsteps=2",
        "model.decoder.n_samples=2",
    ],
    "diffusion_stpp": [
        "model.hidden_dim=16",
        "model.encoder.d_model=16",
        "model.encoder.d_rnn=16",
        "model.encoder.d_inner=16",
        "model.encoder.n_layers=1",
        "model.encoder.n_head=1",
        "model.encoder.d_k=8",
        "model.encoder.d_v=8",
        "model.decoder.hidden_units=16",
        "model.decoder.timesteps=2",
        "model.decoder.sampling_timesteps=1",
    ],
    "neural_cond_gmm": [
        "model.hidden_dim=8",
        "model.backbone.tpp_hidden_dims=8-8",
        "model.decoder.spatial.hidden_dims=8-8",
        "model.decoder.spatial.n_mixtures=2",
    ],
}


def write_history_jsonl(path: Path, sequences: list[dict] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = sequences or SAMPLE_SEQUENCES
    with open(path, "w") as f:
        for seq in payload:
            f.write(json.dumps(seq) + "\n")
    return path


def make_saved_run(
    root: Path,
    *,
    preset: str,
    label: str | None = None,
    override_list: list[str] | None = None,
    preset_status: str | None = None,
    nll_kind: str | None = None,
    nll_report_space: str | None = None,
) -> Path:
    run_dir = Path(root) / (label or preset)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = STPPConfig.from_source(
        preset=preset,
        override_list=list(override_list or MINI_OVERRIDES.get(preset, [])),
    )
    cfg.to_yaml(run_dir / "config.yaml")
    cfg.to_yaml(run_dir / "resolved_config.yaml")

    torch.manual_seed(0)
    model = cfg.model.build_model()
    model.eval()
    torch.save(model.state_dict(), run_dir / "model.ckpt")

    status = preset_status
    if status is None:
        status = "provisional" if preset.startswith("neural_") else "canonical"
    kind = nll_kind
    if kind is None:
        kind = "approx" if preset in {"smash", "diffusion_stpp"} else "exact"
    report_space = nll_report_space
    if report_space is None:
        report_space = "raw" if preset in {"auto_stpp", "deep_stpp"} else "native"

    norm_stats = {
        "normalize": bool(getattr(cfg.data, "normalize", False)),
        "time_mean": 0.0,
        "time_std": 1.0,
        "loc_mean": [0.0, 0.0],
        "loc_std": [1.0, 1.0],
    }
    result = RunResult(
        preset=preset,
        dataset_id="toy_history",
        seed=0,
        val_objective=0.0,
        test_nll=0.0,
        train_time_sec=0.0,
        n_params=sum(p.numel() for p in model.parameters()),
        effective_config=cfg.model_dump(mode="json"),
        preset_status=status,
        checkpoint_path=run_dir / "model.ckpt",
        norm_stats=norm_stats,
        run_dir=run_dir,
        training_objective="nll",
        val_metric_key="nll",
        objective_description="test fixture",
        nll_kind=kind,
        nll_description=f"{kind} fixture nll",
        nll_report_space=report_space,
    )
    result.to_json(run_dir / "run_result.json")

    with open(run_dir / "artifacts.json", "w") as f:
        json.dump(
            {
                "config": "config.yaml",
                "resolved_config": "resolved_config.yaml",
                "run_result": "run_result.json",
                "checkpoint": "model.ckpt",
            },
            f,
            indent=2,
        )
    return run_dir


def assert_finite_array(test_case, value) -> None:
    arr = np.asarray(value)
    test_case.assertTrue(np.isfinite(arr).all(), f"Expected finite values, got {arr}")
