import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from seahorse.baselines.etas_author import ETASAuthorAdapter, ETASContract
from seahorse.baselines.tritpp import TriTPPAdapter, TriTPPContract
from seahorse.baselines.wsm_author import WSMAuthorAdapter, WSMContract


def test_tritpp_fixed_horizon_contract_rejects_boundaries_and_non_strict_times():
    adapter = TriTPPAdapter(TriTPPContract(t_max=1.0, spatial_head_name="gmm"))
    with pytest.raises(ValueError, match="0 < event time"):
        adapter._pack([{"times": [0.0, 0.2]}])
    with pytest.raises(ValueError, match="strictly increasing"):
        adapter._pack([{"times": [0.2, 0.2]}])


def test_etas_catalog_contract_requires_author_columns_and_ordered_windows(tmp_path):
    c = ETASContract(
        "2000-01-01", "2001-01-01", "2002-01-01", "2003-01-01", 2.5, 0.1, [[0, 0], [0, 1], [1, 0]]
    )
    adapter = ETASAuthorAdapter(c)
    frame = pd.DataFrame(
        {
            "id": ["a"],
            "latitude": [1.0],
            "longitude": [2.0],
            "time": ["2001-02-01"],
            "magnitude": [3.0],
        }
    )
    metadata = adapter.catalog_metadata(frame, data_path=tmp_path)
    assert metadata["mc"] == 2.5
    assert (tmp_path / "catalog_input.csv").exists()
    with pytest.raises(ValueError, match="required columns"):
        adapter.catalog_metadata(frame.drop(columns="magnitude"), data_path=tmp_path)


def test_blocked_author_spec_cannot_start_fitting(tmp_path, monkeypatch):
    import json
    import sys
    from seahorse.baselines.run import main

    spec = tmp_path / "blocked.json"
    spec.write_text(
        json.dumps(
            {
                "kind": "etas",
                "execution_gate": "blocked_author_score_contract",
                "execution_gate_reason": "score contract requires correction",
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["baseline", "--spec", str(spec), "--out", str(tmp_path / "out"), "--stage", "fit"],
    )
    with pytest.raises(ValueError, match="score contract requires correction"):
        main()


def test_wsm_prepares_train_only_normalization_and_matched_commands(tmp_path):
    a = WSMAuthorAdapter(WSMContract(1.0, (0.0, 0.0), (5.0, 10.0)), tmp_path)
    s = {
        k: [{"times": [0.1, 0.4], "locations": [[2.0, 4.0], [4.0, 8.0]]}]
        for k in ("train", "val", "test")
    }
    out = a.prepare(s, tmp_path / "prepared")
    assert (out / "train_only_transform.npz").exists()
    stats = np.load(out / "train_only_transform.npz")
    np.testing.assert_array_equal(stats["loc_max"], [5.0, 10.0])
    assert "wsm" in a.command(out, estimator="wsm", seed=1, epochs=1)
    assert "mle" in a.command(out, estimator="mle", seed=1, epochs=1)


@pytest.mark.parametrize("estimator", ["wsm", "mle"])
def test_wsm_pinned_checkpoint_reloads_and_scores_same_tiny_batch(tmp_path, estimator):
    """Exercises the author loss/model path when its pinned checkout is present."""
    author_root = Path(os.environ.get("SEAHORSE_WSM_AUTHOR_ROOT", "/private/tmp/wsm_stpp"))
    if not (author_root / "model" / "SM_model.py").exists():
        pytest.skip("pinned WSM author checkout is optional")
    contract = WSMContract(
        1.0,
        (0.0, 0.0),
        (1.0, 1.0),
        cond_dim=4,
        num_units=4,
        n_layers=1,
        grid_t=2,
        grid_s=2,
        eval_grid_t=2,
        eval_grid_s=2,
    )
    splits = {
        name: [{"times": [0.1, 0.4, 0.7], "locations": [[0.1, 0.2], [0.4, 0.3], [0.7, 0.8]]}]
        for name in ("train", "val", "test")
    }
    adapter = WSMAuthorAdapter(contract, author_root)
    model, metrics = adapter.fit(
        {"train": splits["train"], "val": splits["val"]},
        estimator=estimator,
        epochs=1,
        work_dir=tmp_path / "prepared",
    )
    stats = np.load(metrics["transform"])
    before = adapter.score(
        model, splits["test"], {"loc_min": stats["loc_min"], "loc_max": stats["loc_max"]}
    )
    checkpoint = adapter.save(model, metrics, tmp_path / "checkpoint", estimator=estimator)
    loaded_adapter, loaded_model, payload = WSMAuthorAdapter.load(checkpoint)
    assert loaded_adapter.score(loaded_model, splits["test"], payload) == pytest.approx(before)


def test_wsm_source_score_restores_space_and_time_jacobians(tmp_path):
    class FixedScore:
        def __call__(self, batch, mode):
            windows = batch[0].shape[0]
            return torch.tensor(-3.0 * windows), torch.tensor(0.0), windows

    adapter = WSMAuthorAdapter(WSMContract(1.0, (0.0, 0.0), (2.0, 3.0)), tmp_path)
    rows = [
        {
            "times": [0.1, 0.4],
            "locations": [[0.1, 0.2], [1.0, 2.0]],
            "time_log_jacobian": np.log(2.0),
        },
        {"times": [0.2], "locations": [[1.0, 1.0]], "time_log_jacobian": np.log(3.0)},
    ]
    score = adapter.score(
        FixedScore(), rows, {"loc_min": np.array([0.0, 0.0]), "loc_max": np.array([2.0, 3.0])}
    )
    assert score["windows"] == 2 and score["events"] == 3
    assert score["mean_loglik_per_window_source"] == pytest.approx(
        -3.0 - (3 * np.log(6.0) + 2 * np.log(2.0) + np.log(3.0)) / 2
    )


def test_wsm_restores_best_epoch_not_last(tmp_path, monkeypatch):
    class OptimizerFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch, mode):
            return self.theta, 1

    adapter = WSMAuthorAdapter(WSMContract(1.0, (0.0, 0.0), (1.0, 1.0)), tmp_path)
    monkeypatch.setattr(adapter, "_model", lambda _: OptimizerFixture())
    monkeypatch.setattr(
        adapter,
        "_score_with_transform",
        lambda model, *_: {
            "mean_loglik_per_window_normalized": -float((model.theta.detach() + 0.1).square())
        },
    )
    rows = [{"times": [0.1, 0.4], "locations": [[0.1, 0.2], [0.4, 0.7]]}]
    model, metrics = adapter.fit(
        {"train": rows, "val": rows},
        estimator="wsm",
        epochs=10,
        lr=0.1,
        max_steps=3,
        work_dir=tmp_path / "prepared",
    )
    assert metrics["optimizer_steps"] == 3
    assert metrics["budget_limited"]
    assert float(model.theta.detach()) == pytest.approx(-0.1, abs=1e-6)
