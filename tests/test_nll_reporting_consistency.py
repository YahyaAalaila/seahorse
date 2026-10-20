"""Cross-preset NLL reporting consistency guard.

Every exact-likelihood preset must report the *same statistical quantity*, so that
their benchmark numbers are directly comparable. Concretely, two invariants must
hold for every family with ``nll_kind == "exact"``:

I1. ``nll_matrix`` averages to the scalar NLL.
    The eventwise matrix the benchmark scores from must be a decomposition of the
    scalar the model reports, not a different statistic:

        (nll_matrix * mask).sum() / mask.sum()  ==  result.nll

I2. ``nll_matrix`` agrees with an independent prefix-difference recomputation.
    Scoring event j from the eventwise terms must equal the increase in the
    sequence's total NLL when event j is appended to its own prefix:

        nll_matrix[j]  ==  total_nll(events[:j+1]) - total_nll(events[:j])

    These two routes share no code: the first reads the model's per-event matrix,
    the second re-evaluates the model on N truncated prefixes and differences the
    sequence totals. Agreement means the per-event terms are genuinely causal and
    complete.

Both invariants are parameter-independent — they are identities about how the
likelihood is decomposed, not about how well a model fits — so this test needs no
training and stays fast enough to run in CI.

Regression history
------------------
These invariants were added after two defects that made presets silently
incomparable, in opposite directions:

* ``nsmpp`` built ``nll_matrix`` from ``-log lambda`` alone, dropping the
  compensator that was nonetheless folded into its scalar NLL. Reported NLL was
  roughly one nat/event too low (at a likelihood optimum the compensator over the
  observation window is approximately the event count).
* ``thp_gmm`` / ``rmtpp_gmm`` silently substituted a zero matrix for the per-event
  temporal term, because ``RMTPPTemporalProcess`` / ``THPTemporalProcess`` did not
  implement ``event_logprob_matrix``. Their reported NLL was the *spatial term
  alone*, several nats too high, and nearly independent of the temporal model.

I1 catches both at their source; I2 catches any future variant of either.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from seahorse.config import STPPConfig
from seahorse.evaluation.likelihood import (
    _extract_eventwise_next_event_nlls,
    _prefix_difference_next_event_nlls,
)
from seahorse.runner.results import resolve_loss_result_reporting


#: Every preset in the exact-likelihood cohort compared by the HawkesNest suites.
EXACT_PRESETS_DECLARED = (
    "static_hawkes",
    "nsmpp",
    "thp_gmm",
    "rmtpp_gmm",
)


def _registered_presets() -> frozenset[str]:
    """The presets this checkout can actually build, from the canonical registry."""
    from seahorse.models.configs import ConfigRegistry

    return frozenset(ConfigRegistry.preset_names())


_AVAILABLE = _registered_presets()
#: The subset actually exercised here. A preset absent from this checkout is
#: dropped rather than failing the suite — but NEVER silently: SKIPPED_PRESETS is
#: asserted on, so a shrinking cohort shows up as a named list instead of as a
#: green run that quietly covers less than it claims.
EXACT_PRESETS = tuple(p for p in EXACT_PRESETS_DECLARED if p in _AVAILABLE)
SKIPPED_PRESETS = tuple(p for p in EXACT_PRESETS_DECLARED if p not in EXACT_PRESETS)


class TestCohortCoverage(unittest.TestCase):
    """Fail loudly if the exact-likelihood cohort silently shrinks."""

    #: Presets known to be unavailable in this checkout, with the reason. Anything
    #: dropping out that is NOT listed here is a regression, not an expectation.
    EXPECTED_MISSING: dict[str, str] = {}

    def test_only_expected_presets_are_missing(self):
        unexpected = sorted(set(SKIPPED_PRESETS) - set(self.EXPECTED_MISSING))
        self.assertEqual(
            unexpected, [],
            f"exact-likelihood presets vanished from the registry: {unexpected}. "
            f"Either they regressed, or EXPECTED_MISSING needs updating.",
        )
        self.assertTrue(EXACT_PRESETS, "no exact presets left to check at all")

#: Shrink the heavier families so the whole suite stays CI-fast on CPU.
MINI_OVERRIDES: dict[str, list[str]] = {
    "nsmpp": [
        "model.decoder.n_basis=2",
        "model.decoder.basis_dim=4",
        "model.decoder.nn_width=6",
        "model.decoder.int_res=5",
        "model.decoder.compensator_chunk_size=16",
        # A background rate of the order the data implies, rather than the
        # near-zero default. With mu ~ 0 the compensator is worth ~0.001 nat/event
        # and a missing-compensator regression would slip under the tolerance; a
        # realistic rate makes it worth ~0.5 nat/event, as on real data.
        "model.decoder.mu=6.0",
    ],
    "thp_gmm": [
        "model.hidden_size=16",
        "model.n_heads=2",
        "model.n_layers=1",
        "model.dropout=0.0",
    ],
    "rmtpp_gmm": [
        "model.hidden_size=16",
    ],
}

#: Two short sequences inside the unit support box (nsmpp integrates over [0,1]^3).
FIXTURE_SEQUENCES = [
    {
        "times": np.array(
            [0.05, 0.13, 0.21, 0.34, 0.46, 0.58, 0.67, 0.79, 0.88], dtype=np.float64
        ),
        "locations": np.array(
            [
                [0.12, 0.20], [0.24, 0.17], [0.31, 0.42], [0.48, 0.51], [0.55, 0.39],
                [0.62, 0.66], [0.71, 0.58], [0.83, 0.74], [0.91, 0.62],
            ],
            dtype=np.float32,
        ),
    },
    {
        "times": np.array([0.08, 0.19, 0.28, 0.41, 0.55, 0.70, 0.86], dtype=np.float64),
        "locations": np.array(
            [
                [0.22, 0.31], [0.30, 0.28], [0.44, 0.47], [0.51, 0.35],
                [0.60, 0.61], [0.72, 0.49], [0.88, 0.77],
            ],
            dtype=np.float32,
        ),
    },
]


class _StubRunner:
    """Minimal duck-typed stand-in for ``STPPRunner``.

    ``seahorse.evaluation.likelihood`` only needs ``model``, ``norm_stats`` and
    ``config`` from a runner. Building the model directly from the config keeps this
    test free of Lightning setup and training, which is what makes it CI-fast.
    """

    def __init__(self, model, config, norm_stats: dict):
        self.model = model
        self.config = config
        self.norm_stats = norm_stats


def _build_runner(
    preset: str,
    *,
    overrides: list[str] | None = None,
    cli_values: dict | None = None,
) -> _StubRunner:
    override_list = list(MINI_OVERRIDES.get(preset, []))
    override_list.extend(overrides or [])

    config = STPPConfig.from_source(
        preset=preset, override_list=override_list, cli_values=cli_values
    )
    torch.manual_seed(0)
    model = config.model.build_model()
    model.eval()
    return _StubRunner(model, config, {"normalize": False})


def _forward(runner: _StubRunner, seq: dict, device: torch.device):
    times = torch.tensor(seq["times"], dtype=torch.float32, device=device).unsqueeze(0)
    locs = torch.tensor(seq["locations"], dtype=torch.float32, device=device).unsqueeze(0)
    lengths = torch.tensor([times.shape[1]], dtype=torch.long, device=device)
    output = runner.model.eval_forward(times=times, locations=locs, lengths=lengths)
    return output, runner.model.compute_loss(output)


class TestEventwiseMatrixAveragesToScalarNLL(unittest.TestCase):
    """I1: the eventwise matrix is a decomposition of the reported scalar NLL."""

    def test_all_exact_presets(self):
        device = torch.device("cpu")
        failures = []
        for preset in EXACT_PRESETS:
            runner = _build_runner(preset)
            caps = runner.model.event_model.capabilities
            self.assertEqual(
                caps.nll_kind, "exact", f"{preset} is not an exact-likelihood preset"
            )
            with torch.no_grad():
                output, result = _forward(runner, FIXTURE_SEQUENCES[0], device)

            matrix = output.get("nll_matrix")
            mask = output.get("mask")
            self.assertIsInstance(
                matrix,
                torch.Tensor,
                f"{preset} declares exact NLL but exposes no nll_matrix",
            )
            weights = mask.to(matrix.dtype)
            matrix_mean = float((matrix * weights).sum() / weights.sum().clamp(min=1.0))
            scalar = float(result.nll)
            gap = abs(matrix_mean - scalar)
            if gap >= 1e-4:
                failures.append(
                    f"{preset}: mean(nll_matrix)={matrix_mean:.6f} "
                    f"vs result.nll={scalar:.6f} (gap {gap:.6f})"
                )
        self.assertEqual(
            failures,
            [],
            "eventwise matrix does not average to the reported scalar NLL:\n"
            + "\n".join(failures),
        )


class TestEventwiseMatrixMatchesPrefixDifferences(unittest.TestCase):
    """I2: eventwise terms equal an independent prefix-difference recomputation."""

    TOLERANCE = 1e-3

    def _compare_for_preset(self, preset: str, runner: _StubRunner) -> list[str]:
        device = torch.device("cpu")
        problems = []
        for index, seq in enumerate(FIXTURE_SEQUENCES):
            seq_len = int(seq["times"].shape[0])
            with torch.no_grad():
                output, result = _forward(runner, seq, device)
                reported, _t, _s, _extra, _space = resolve_loss_result_reporting(
                    result, requested_space=runner.config.training.test_nll_space
                )
                correction = float(reported - float(result.nll))
                eventwise = _extract_eventwise_next_event_nlls(
                    output, seq_len=seq_len, correction=correction
                )
                prefix = _prefix_difference_next_event_nlls(runner, seq, device=device)

            self.assertIsNotNone(
                eventwise, f"{preset}: no eventwise next-event terms available"
            )
            self.assertEqual(
                eventwise.shape,
                prefix.shape,
                f"{preset}: eventwise/prefix target counts disagree",
            )
            max_gap = float(np.max(np.abs(eventwise - prefix)))
            if not (max_gap < self.TOLERANCE):
                problems.append(
                    f"{preset} seq{index}: max |eventwise - prefix_difference| "
                    f"= {max_gap:.6f} (mean eventwise={eventwise.mean():.4f}, "
                    f"mean prefix={prefix.mean():.4f})"
                )
        return problems

    def test_all_exact_presets(self):
        failures = []
        for preset in EXACT_PRESETS:
            failures.extend(self._compare_for_preset(preset, _build_runner(preset)))
        self.assertEqual(
            failures,
            [],
            "eventwise NLL disagrees with independent prefix differences:\n"
            + "\n".join(failures),
        )


class TestRawSpaceCorrectionAppliedConsistently(unittest.TestCase):
    """A nonzero raw-space Jacobian must shift both scoring routes identically.

    Guards against the correction being applied to one route only, twice, or with
    the wrong sign — all of which would reintroduce a space mismatch between
    ``nll_matrix`` and ``result.nll``.
    """

    def test_zscore_transform_shifts_both_routes_equally(self):
        device = torch.device("cpu")
        # A z-score input transform with non-unit scales gives a nonzero log-Jacobian.
        cli_values = {
            "model": {
                "input_transform": {
                    "type": "zscore",
                    "normalize_time": True,
                    "normalize_space": True,
                    "time_mean": 0.0,
                    "time_std": 2.5,
                    "loc_mean": [0.0, 0.0],
                    "loc_std": [1.5, 2.0],
                }
            }
        }
        runner = _build_runner("thp_gmm", cli_values=cli_values)
        seq = FIXTURE_SEQUENCES[0]
        seq_len = int(seq["times"].shape[0])

        with torch.no_grad():
            output, result = _forward(runner, seq, device)
            reported, _t, _s, _extra, space = resolve_loss_result_reporting(
                result, requested_space="raw"
            )
            correction = float(reported - float(result.nll))
            eventwise = _extract_eventwise_next_event_nlls(
                output, seq_len=seq_len, correction=correction
            )
            prefix = _prefix_difference_next_event_nlls(runner, seq, device=device)

        self.assertEqual(space, "raw", "expected raw-space reporting to engage")
        self.assertGreater(
            abs(correction), 1e-6, "expected a nonzero raw-space Jacobian correction"
        )
        self.assertLess(float(np.max(np.abs(eventwise - prefix))), 1e-3)


class TestFactorizedRefusesToInventTemporalTerms(unittest.TestCase):
    """A temporal process without per-event terms must not yield a bogus matrix.

    Previously a missing ``event_logprob_matrix`` was replaced by zeros, turning
    ``nll_matrix`` into a spatial-only quantity that the benchmark then reported as
    a joint NLL. The model must now decline to expose eventwise terms instead, so
    consumers fall back to the correct prefix-difference route or fail loudly.
    """

    def test_missing_event_logprob_matrix_suppresses_eventwise_terms(self):
        from seahorse.models.event_models.factorized import FactorizedEventModel
        from seahorse.models.spatial_models.gaussian_mixture import (
            GaussianMixtureSpatialModel,
        )
        from seahorse.models.temporal_models.parametric_processes import HawkesProcess

        temporal = HawkesProcess()
        event_model = FactorizedEventModel(
            temporal_model=temporal,
            spatial_model=GaussianMixtureSpatialModel(),
        )
        self.assertTrue(event_model.capabilities.exposes_eventwise_terms)

        # Hide the hook the way a sequence-level-only temporal process would.
        temporal.event_logprob_matrix = None
        self.assertFalse(event_model.capabilities.exposes_eventwise_terms)


class TestAllTemporalProcessesExposePerEventTerms(unittest.TestCase):
    """Every registered factorized temporal process must supply per-event terms.

    Without this, ``FactorizedEventModel`` has no temporal contribution to report
    and its preset drops out of the eventwise-comparable cohort.
    """

    def test_registered_temporal_models(self):
        from seahorse.models.temporal_models.neural_temporal import (
            RMTPPTemporalProcess,
            THPTemporalProcess,
        )
        from seahorse.models.temporal_models.parametric_processes import (
            HawkesProcess,
            HomogeneousPoissonProcess,
            SelfCorrectingProcess,
        )

        classes = [
            HomogeneousPoissonProcess,
            HawkesProcess,
            SelfCorrectingProcess,
            RMTPPTemporalProcess,
            THPTemporalProcess,
        ]
        for cls in classes:
            self.assertTrue(
                callable(getattr(cls, "event_logprob_matrix", None)),
                f"{cls.__name__} does not implement event_logprob_matrix",
            )

    def test_per_event_terms_sum_to_sequence_logprob(self):
        """``sum_i event_logprob_matrix[i] == logprob`` under the last-event window."""
        from seahorse.models.temporal_models.neural_temporal import (
            RMTPPTemporalProcess,
            THPTemporalProcess,
        )
        from seahorse.models.temporal_models.parametric_processes import (
            HawkesProcess,
            HomogeneousPoissonProcess,
            SelfCorrectingProcess,
        )

        torch.manual_seed(0)
        batch, length = 3, 9
        times = torch.cumsum(torch.rand(batch, length) * 0.2, dim=1)
        times = times - times[:, :1]
        locations = torch.rand(batch, length, 2)
        mask = torch.ones(batch, length)
        mask[2, -3:] = 0.0
        lengths = mask.sum(dim=1).long()
        t0 = torch.zeros(batch)
        # Last-event convention: the tail compensator vanishes, so the per-event
        # terms must account for the whole sequence log-prob.
        t1 = times[torch.arange(batch), lengths - 1]

        models = [
            HomogeneousPoissonProcess(),
            HawkesProcess(),
            SelfCorrectingProcess(),
            RMTPPTemporalProcess(hidden_size=16),
            THPTemporalProcess(hidden_size=16, n_heads=2, n_layers=1, dropout=0.0),
        ]
        for model in models:
            model.eval()
            with torch.no_grad():
                sequence_logprob = model.logprob(times, locations, mask, t0, t1)
                matrix = model.event_logprob_matrix(times, locations, mask, t0, t1)
            gap = float((matrix.sum(dim=1) - sequence_logprob).abs().max())
            self.assertLess(
                gap,
                1e-4,
                f"{type(model).__name__}: sum(event_logprob_matrix) - logprob = {gap:.3e}",
            )


class TestNSMPPIntervalCompensator(unittest.TestCase):
    """The NSMPP per-interval compensator must reconstruct the sequence total.

    The per-event attribution is only legitimate if the increments are a genuine
    decomposition of the compensator over ``(support_t0, t_N]`` — not an invented
    split of a scalar.
    """

    def _model(self):
        from seahorse.models.event_models.nsmpp_deepbasis import (
            NSMPPDeepBasisEventModel,
        )

        torch.manual_seed(0)
        return NSMPPDeepBasisEventModel(
            spatial_dim=2,
            mu=0.5,
            n_basis=2,
            basis_dim=4,
            nn_width=6,
            int_res=10,
            support_t0=0.0,
            support_t1=1.0,
            support_space_min=(0.0, 0.0),
            support_space_max=(1.0, 1.0),
            interval_int_res=2,
        )

    def _events(self):
        seq = FIXTURE_SEQUENCES[0]
        times = torch.tensor(seq["times"], dtype=torch.float32).unsqueeze(0)
        locs = torch.tensor(seq["locations"], dtype=torch.float32).unsqueeze(0)
        events = torch.cat([times.unsqueeze(-1), locs], dim=-1)
        valid = torch.ones(events.shape[:2], dtype=torch.bool)
        return events, valid

    def test_prefix_kernel_sum_matches_explicit_masked_sum(self):
        model = self._model()
        events, valid = self._events()
        length = events.shape[1]
        causal = torch.tril(
            torch.ones(length, length, dtype=torch.bool), diagonal=-1
        ).unsqueeze(0)
        history_mask = valid.unsqueeze(1) & causal
        counts = torch.arange(length).unsqueeze(0).expand(1, length)
        with torch.no_grad():
            explicit = model.kernel.sum_over_history(events, events, history_mask)
            prefixed = model.kernel.sum_over_prefix_history(events, counts, events, valid)
        self.assertLess(float((explicit - prefixed).abs().max()), 1e-5)

    def test_increments_sum_to_independent_reintegration(self):
        """Sum of per-interval increments == a brute-force integral of the same region.

        The reference re-integration is written independently of the model's own
        quadrature (finer, uniform grid over ``(support_t0, t_N]``), so agreement
        confirms the increments really are the compensator and not a rescaled proxy.
        """
        import torch.nn.functional as F

        model = self._model()
        events, valid = self._events()
        with torch.no_grad():
            increments = model._interval_compensators(events, valid)

            times = events[0, :, 0]
            end = float(times[-1])
            resolution = 32
            axis = (torch.arange(resolution, dtype=torch.float32) + 0.5) / resolution
            grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
            grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
            cell = (1.0 / resolution) ** 2
            n_nodes = 240
            step = end / n_nodes
            total = 0.0
            for node in range(n_nodes):
                moment = (node + 0.5) * step
                n_history = int((times < moment).sum())
                query = torch.cat(
                    [
                        torch.full((1, grid.shape[0], 1), moment),
                        grid.unsqueeze(0),
                    ],
                    dim=-1,
                )
                kernel_sum = model.kernel.sum_over_prefix_history(
                    query,
                    torch.full((1, grid.shape[0]), n_history, dtype=torch.long),
                    events,
                    valid,
                )
                rate = F.softplus(kernel_sum + model.raw_mu) + model.intensity_eps
                total += float(rate.sum()) * cell * step

        self.assertAlmostEqual(float(increments.sum()), total, delta=5e-3)


if __name__ == "__main__":
    unittest.main()
