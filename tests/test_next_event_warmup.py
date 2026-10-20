"""Next-event scores must exclude history-only events, with explicit alignment."""

from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch

from seahorse.data.transforms import PaperAffineTransformArtifact
from seahorse.evaluation.likelihood import (
    _build_single_seq_batch,
    _extract_eventwise_next_event_nlls,
    _prefix_difference_next_event_nlls,
    _prefix_difference_next_event_nlls_unbatched,
    compute_next_event_test_nll,
)
from seahorse.registry import build_model
from seahorse.models.unified_model import LossResult
from seahorse.runner.results import resolve_loss_result_reporting


CPU = torch.device("cpu")
PRESETS = ("auto_stpp", "deep_stpp")


def _sequence(n):
    rng = np.random.default_rng(5)
    return {
        "times": np.cumsum(rng.uniform(0.04, 0.1, n)),
        "locations": rng.uniform(0.1, 0.9, (n, 2)).astype(np.float32),
        "marks": (np.arange(n) % 3).astype(np.int64),
    }


def _runner(preset, space="native", history=20):
    torch.manual_seed(7)
    decoder = (
        {"lookback": history, "n_prodnet": 2, "hidden_size": 8, "num_layers": 1}
        if preset == "auto_stpp"
        else {"seq_len": history, "num_points": 2, "n_layers": 1}
    )
    model = build_model(
        config={
            "encoder": {"num_layers": 1},
            "decoder": decoder,
            "paper_dt_range": 1.5,
            "paper_loc_range": [2.0, 3.0],
            "input_transform": PaperAffineTransformArtifact(
                paper_dt_range=1.5, paper_loc_range=(2.0, 3.0),
            ).serialize(),
        },
        preset=preset,
        spatial_dim=2,
        hidden_dim=8,
    ).eval()
    return SimpleNamespace(
        model=model,
        norm_stats={"normalize": False},
        config=SimpleNamespace(
            training=SimpleNamespace(test_nll_space=space),
            data=SimpleNamespace(seed=0),
        ),
    )


def _expected(runner, seq):
    with torch.no_grad():
        output = runner.model.eval_forward(
            **_build_single_seq_batch(seq, runner.norm_stats, CPU)
        )
        result = runner.model.compute_loss(output)
    reported, *_ = resolve_loss_result_reporting(
        result, requested_space=runner.config.training.test_nll_space
    )
    correction = reported - float(result.nll)
    return output, output["nll_matrix"][output["mask"] > 0].numpy() + correction


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("space", ("native", "raw"))
def test_history_is_not_counted_as_zero_loss(preset, space):
    runner = _runner(preset, space)
    seq = _sequence(25)
    output, expected = _expected(runner, seq)
    summary = compute_next_event_test_nll(runner, [seq], device=CPU)

    assert expected.shape == (5,)
    assert summary["report_space"] == space
    assert summary["n_contexts"] == 24
    assert summary["n_scored_contexts"] == 5
    assert summary["n_missing_contexts"] == 19
    assert np.isnan(summary["per_context_nll"][:19]).all()
    np.testing.assert_allclose(summary["per_context_nll"][19:], expected, rtol=1e-5)
    assert summary["mean_nll"] == pytest.approx(float(expected.mean()), rel=1e-5)
    assert summary["method"] == "exact_next_event_from_eventwise_terms"
    torch.testing.assert_close(output["eventwise_target_indices"], torch.arange(20, 25)[None])


@pytest.mark.parametrize("preset", PRESETS)
def test_short_sequences_have_no_scored_predictions(preset):
    runner = _runner(preset)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary = compute_next_event_test_nll(runner, [_sequence(20)], device=CPU)
    assert summary["n_contexts"] == 19
    assert summary["n_scored_contexts"] == 0
    assert np.isnan(summary["per_context_nll"]).all()
    assert np.isnan(summary["mean_nll"])


@pytest.mark.parametrize("preset", PRESETS)
def test_uneven_lengths_weight_each_scored_prediction_once(preset):
    runner = _runner(preset, "raw")
    seqs = [_sequence(n) for n in (20, 21, 25)]
    expected = np.concatenate([_expected(runner, seq)[1] for seq in seqs])
    summary = compute_next_event_test_nll(runner, seqs, device=CPU)
    assert summary["n_scored_contexts"] == 6
    assert summary["mean_nll"] == pytest.approx(float(expected.mean()), rel=1e-5)

    batch = _build_single_seq_batch(seqs[-1], runner.norm_stats, CPU)
    batch = {key: value.expand(3, *value.shape[1:]).clone() for key, value in batch.items()}
    batch["lengths"] = torch.tensor([20, 21, 25])
    with torch.no_grad():
        output = runner.model.eval_forward(**batch)
    expected_indices = torch.tensor([
        [-1, -1, -1, -1, -1], [20, -1, -1, -1, -1], [20, 21, 22, 23, 24]
    ])
    torch.testing.assert_close(output["eventwise_target_indices"], expected_indices)


@pytest.mark.parametrize("preset", PRESETS)
def test_independent_prefix_recomputation_agrees_after_warmup(preset):
    runner = _runner(preset, "raw", history=3)
    seq = _sequence(7)
    _, expected = _expected(runner, seq)
    with torch.no_grad():
        for recompute in (
            _prefix_difference_next_event_nlls,
            _prefix_difference_next_event_nlls_unbatched,
        ):
            actual = recompute(runner, seq, device=CPU)
            assert np.isnan(actual[:2]).all()
            np.testing.assert_allclose(actual[2:], expected, rtol=1e-4, atol=1e-4)


def test_explicit_target_alignment_preserves_real_zero_loss():
    output = {
        "nll_matrix": torch.tensor([[0.0, 4.0, 999.0]]),
        "mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "eventwise_target_indices": torch.tensor([[3, 5, -1]]),
    }
    actual = _extract_eventwise_next_event_nlls(output, seq_len=6, correction=0.0)
    np.testing.assert_allclose(actual, [np.nan, np.nan, 0.0, np.nan, 4.0], equal_nan=True)


def test_full_event_matrix_respects_partial_next_event_mask():
    output = {
        "nll_matrix": torch.tensor([[100.0, 100.0, 100.0, 0.0, 4.0]]),
        "mask": torch.ones(1, 5),
        "next_event_mask": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]]),
    }
    actual = _extract_eventwise_next_event_nlls(output, seq_len=5, correction=0.0)
    np.testing.assert_allclose(actual, [np.nan, np.nan, 0.0, 4.0], equal_nan=True)


@pytest.mark.parametrize("indices", ([3, 3], [4, 3], [-1, 3], [3, 6]))
def test_invalid_explicit_targets_fail_loudly(indices):
    output = {
        "nll_matrix": torch.ones(1, 2),
        "mask": torch.ones(1, 2),
        "eventwise_target_indices": torch.tensor([indices]),
    }
    with pytest.raises(ValueError, match="eventwise_target_indices"):
        _extract_eventwise_next_event_nlls(output, seq_len=6, correction=0.0)


class _AdditivePrefixModel:
    """A known additive loss sensitive to both time and location normalization."""

    event_model = SimpleNamespace(capabilities=SimpleNamespace(nll_kind="exact"))

    def eval(self):
        return self

    def eval_forward(self, *, times, locations, lengths, **kwargs):
        idx = torch.arange(times.shape[1])[None]
        mask = (idx < lengths[:, None]) & (idx >= 2)
        values = times.double() + locations.double().sum(-1)
        counts = mask.sum(1)
        return {
            "mask": mask,
            "nll_per_event": (values * mask).sum(1) / counts.clamp(min=1),
        }

    def compute_loss(self, output):
        counts = output["mask"].sum(1)
        total = counts.sum()
        nll = (output["nll_per_event"] * counts).sum() / total.clamp(min=1)
        return LossResult(
            loss=nll, nll=nll, total_events=total, kl=None, aux_terms={},
            extra_metrics={
                "raw_space_nll": float(nll) + 0.3, "raw_reporting_correction": 0.3,
            },
        )


def test_prefix_paths_normalize_once_and_apply_raw_correction_once():
    runner = SimpleNamespace(
        model=_AdditivePrefixModel(),
        norm_stats={
            "normalize": True, "time_mean": 100.0, "time_std": 4.0,
            "loc_mean": [10.0, 20.0], "loc_std": [2.0, 5.0],
        },
        config=SimpleNamespace(training=SimpleNamespace(test_nll_space="raw")),
    )
    seq = {
        "times": np.array([100.0, 104.0, 108.0, 112.0]),
        "locations": np.array([[10, 20], [11, 21], [12, 22], [13, 23]], dtype=np.float32),
    }
    expected = np.array([np.nan, 3.7, 5.4])
    for recompute in (
        _prefix_difference_next_event_nlls,
        _prefix_difference_next_event_nlls_unbatched,
    ):
        np.testing.assert_allclose(recompute(runner, seq, device=CPU), expected, atol=1e-6)
    summary = compute_next_event_test_nll(runner, [seq], device=CPU)
    np.testing.assert_allclose(summary["per_context_nll"], expected, atol=1e-6)
    assert summary["n_scored_contexts"] == 2


@pytest.mark.parametrize("nonfinite", (float("inf"), float("nan")))
@pytest.mark.parametrize("space", ("native", "raw"))
def test_nonfinite_scored_loss_is_not_dropped_as_warmup(nonfinite, space):
    class Model(_AdditivePrefixModel):
        def eval_forward(self, **kwargs):
            return {
                "nll_matrix": torch.tensor([[1.0, nonfinite]]),
                "mask": torch.ones(1, 2),
                "eventwise_target_indices": torch.tensor([[2, 4]]),
                "nll_per_event": torch.tensor([(1.0 + nonfinite) / 2]),
            }

    runner = SimpleNamespace(
        model=Model(), norm_stats={"normalize": False},
        config=SimpleNamespace(training=SimpleNamespace(test_nll_space=space)),
    )
    summary = compute_next_event_test_nll(runner, [_sequence(5)], device=CPU)
    assert summary["n_contexts"] == 4
    assert summary["n_scored_contexts"] == 2
    assert summary["n_missing_contexts"] == 2
    assert summary["n_unscored_contexts"] == 2
    assert summary["n_nonfinite_scored_contexts"] == 1
    assert summary["n_failed_contexts"] == int(np.isnan(nonfinite))
    np.testing.assert_array_equal(summary["per_context_scored_mask"], [False, True, False, True])
    assert np.isnan(summary["mean_nll"]) if np.isnan(nonfinite) else np.isposinf(summary["mean_nll"])
    assert summary["per_context_nll"][1] == pytest.approx(1.3 if space == "raw" else 1.0)


def test_raw_correction_cannot_be_inferred_from_infinite_aggregates():
    from seahorse.evaluation.likelihood import _reporting_correction

    result = SimpleNamespace(nll=float("inf"), extra_metrics={"raw_space_nll": float("inf")})
    with pytest.raises(ValueError, match="raw_reporting_correction"):
        _reporting_correction(result, float("inf"), "raw")
