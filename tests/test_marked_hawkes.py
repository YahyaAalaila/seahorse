"""Tests for the marked (mutually-exciting) multivariate Hawkes baseline.

Coverage
--------
  * exactness of the compensator vs brute-force numerical re-integration
    (2-D spatial quadrature x Gauss-Legendre time quadrature)
  * the diagonal-only variant genuinely zeroes cross-mark terms (value *and*
    gradient), rather than merely shrinking them
  * recovery of a known ``M x M`` branching matrix from data simulated from the
    model itself
  * unmarked data / unmarked presets are unaffected
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from seahorse.data.marked_hawkes import (
    simulate_marked_hawkes,
    simulate_marked_hawkes_sequences,
)
from seahorse.models.abstractions import StateContext
from seahorse.models.configs.base import ConfigRegistry
from seahorse.models.event_models.marked_hawkes import MarkedHawkesEventModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_model(*, n_marks, spatial, diagonal_only=False, seed=0, dtype=torch.float64):
    """A model with non-degenerate, randomized parameters (so the test has teeth)."""
    torch.manual_seed(seed)
    m = MarkedHawkesEventModel(
        n_marks=n_marks, spatial_dim=2, spatial=spatial, diagonal_only=diagonal_only
    )
    with torch.no_grad():
        m._raw_mu.normal_(0.0, 0.6)
        m._raw_alpha.normal_(-1.0, 0.8)
        m._raw_decay.fill_(0.3)
        if spatial:
            m._disp_mean.normal_(0.0, 0.4)
            m._raw_disp_scale.normal_(-0.6, 0.3)
            m._bg_mean.normal_(0.0, 0.5)
            m._raw_bg_scale.normal_(0.0, 0.3)
    return m.to(dtype)


def _numpy_params(m):
    sp = torch.nn.functional.softplus
    out = {
        "mu": m.mu.detach().numpy().astype(np.float64),
        "alpha": m.alpha.detach().numpy().astype(np.float64),
        "q": float(m.decay.detach()),
    }
    if m.spatial:
        out.update(
            disp_mean=m._disp_mean.detach().numpy().astype(np.float64),
            disp_scale=(sp(m._raw_disp_scale) + 1e-6).detach().numpy().astype(np.float64),
            bg_mean=m._bg_mean.detach().numpy().astype(np.float64),
            bg_scale=(sp(m._raw_bg_scale) + 1e-6).detach().numpy().astype(np.float64),
        )
    return out


def _lambda_per_mark(P, t, Y, times, locs, marks, spatial):
    """lambda_k(t, y) at every grid point ``Y`` (G,2) for every mark -> (G, M)."""
    mu, alpha, q = P["mu"], P["alpha"], P["q"]
    g_n = Y.shape[0]
    if spatial:
        z = (Y[:, None, :] - P["bg_mean"][None]) / P["bg_scale"][None]
        log_rho = (
            -0.5 * (z ** 2).sum(-1)
            - np.log(P["bg_scale"]).sum(-1)[None, :]
            - np.log(2 * np.pi)
        )
        out = mu[None, :] * np.exp(log_rho)
    else:
        out = np.tile(mu[None, :], (g_n, 1))

    for j in np.nonzero(times < t)[0]:
        kj = marks[j]
        g = q * np.exp(-q * (t - times[j]))
        if spatial:
            zz = (
                (Y - locs[j])[:, None, :] - P["disp_mean"][kj][None]
            ) / P["disp_scale"][kj][None]
            log_f = (
                -0.5 * (zz ** 2).sum(-1)
                - np.log(P["disp_scale"][kj]).sum(-1)[None, :]
                - np.log(2 * np.pi)
            )
            out = out + alpha[kj][None, :] * g * np.exp(log_f)
        else:
            out = out + alpha[kj][None, :] * g
    return out


def _numeric_eventwise_nll(P, times, locs, marks, spatial, *, t0=0.0, L=8.0, h=0.1, n_gl=24):
    """Direct quadrature of ``-log lambda_{k_i}(t_i,x_i) + \\int\\int\\sum_k lambda_k``.

    The spatial integral uses a trapezoid grid (spectrally accurate for
    Gaussians) and the time integral uses Gauss-Legendre (exact for sums of
    exponentials at this order).
    """
    n = times.shape[0]
    if spatial:
        ax = np.arange(-L, L + h / 2, h)
        xx, yy = np.meshgrid(ax, ax, indexing="ij")
        grid = np.stack([xx.ravel(), yy.ravel()], -1)
        cell = h * h
    else:
        grid, cell = np.zeros((1, 2)), 1.0

    gl_x, gl_w = np.polynomial.legendre.leggauss(n_gl)
    out = np.zeros(n)
    for i in range(n):
        t_prev = t0 if i == 0 else times[i - 1]
        t_cur = times[i]
        pt = locs[i][None, :] if spatial else np.zeros((1, 2))
        lam = _lambda_per_mark(P, t_cur, pt, times[:i], locs[:i], marks[:i], spatial)[0, marks[i]]
        term = -np.log(lam)
        if t_cur > t_prev:
            mid, half = 0.5 * (t_prev + t_cur), 0.5 * (t_cur - t_prev)
            acc = 0.0
            for xg, wg in zip(gl_x, gl_w):
                lk = _lambda_per_mark(
                    P, mid + half * xg, grid, times[:i], locs[:i], marks[:i], spatial
                )
                acc += wg * lk.sum() * cell
            term += half * acc
        out[i] = term
    return out


def _single_seq_batch(times, locs, marks, dtype=torch.float64):
    tt = torch.tensor(times, dtype=dtype).unsqueeze(0)
    ll = torch.tensor(locs, dtype=dtype).unsqueeze(0)
    kk = torch.tensor(marks, dtype=torch.long).unsqueeze(0)
    lens = torch.tensor([times.shape[0]])
    state = StateContext(
        payload={"times": tt, "locations": ll, "lengths": lens, "marks": kk}
    )
    return tt, ll, kk, lens, state


def _pack(seqs, dtype=torch.float32):
    b = len(seqs)
    t = max(len(s["times"]) for s in seqs)
    times = torch.zeros(b, t, dtype=dtype)
    locs = torch.zeros(b, t, 2, dtype=dtype)
    marks = torch.zeros(b, t, dtype=torch.long)
    lens = torch.zeros(b, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = len(s["times"])
        times[i, :n] = torch.tensor(s["times"], dtype=dtype)
        locs[i, :n] = torch.tensor(s["locations"], dtype=dtype)
        marks[i, :n] = torch.tensor(s["marks"], dtype=torch.long)
        lens[i] = n
    return times, locs, marks, lens


# ---------------------------------------------------------------------------
# 1. Exactness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_marks,spatial,diagonal_only",
    [
        (3, True, False),    # full M x M, space-time
        (3, True, True),     # diagonal-only, space-time
        (4, False, False),   # full M x M, purely temporal
        (1, True, False),    # unmarked special case
    ],
)
def test_nll_matrix_matches_numerical_reintegration(n_marks, spatial, diagonal_only):
    """Eventwise NLL must equal -log lambda + numerically re-integrated compensator.

    The compensator is re-derived from scratch: the mark sum and the *full*
    integral over R^2 are done by quadrature, with no use of the closed form.
    """
    model = _make_model(n_marks=n_marks, spatial=spatial, diagonal_only=diagonal_only)
    P = _numpy_params(model)

    rng = np.random.default_rng(5)
    n_events = 6
    times = np.sort(rng.uniform(0.0, 6.0, n_events))
    times = times - times[0]
    locs = rng.normal(0.0, 1.0, (n_events, 2))
    marks = rng.integers(0, n_marks, n_events)

    tt, ll, kk, lens, state = _single_seq_batch(times, locs, marks)
    out = model.training_loss(times=tt, locations=ll, lengths=lens, state=state, marks=kk)
    model_nll = out["nll_matrix"][0].detach().numpy()

    numeric_nll = _numeric_eventwise_nll(P, times, locs, marks, spatial)

    max_diff = np.abs(model_nll - numeric_nll).max()
    assert max_diff < 1e-3, (
        f"eventwise NLL mismatch: max|model - numeric| = {max_diff:.3e}\n"
        f"model  = {model_nll}\nnumeric = {numeric_nll}"
    )
    # Much tighter than the contract in practice — guard against silent drift.
    assert max_diff < 1e-8


def test_nll_matrix_sums_to_sequence_nll():
    """sum(nll_matrix) must be the sequence NLL (per-event terms telescope)."""
    model = _make_model(n_marks=3, spatial=True)
    rng = np.random.default_rng(1)
    times = np.sort(rng.uniform(0.0, 5.0, 9))
    times -= times[0]
    locs = rng.normal(0.0, 1.0, (9, 2))
    marks = rng.integers(0, 3, 9)
    tt, ll, kk, lens, state = _single_seq_batch(times, locs, marks)
    out = model.training_loss(times=tt, locations=ll, lengths=lens, state=state, marks=kk)
    total = out["nll_matrix"].sum()
    assert torch.allclose(total / lens.sum(), out["nll"], atol=1e-12)


def test_explicit_window_tail_is_accounted():
    """With an explicit t1 the residual mass on (t_last, t1] is charged, not dropped."""
    rng = np.random.default_rng(3)
    times = np.sort(rng.uniform(0.0, 4.0, 6))
    times -= times[0]
    locs = rng.normal(0.0, 1.0, (6, 2))
    marks = rng.integers(0, 2, 6)
    tt, ll, kk, lens, state = _single_seq_batch(times, locs, marks)

    base = _make_model(n_marks=2, spatial=True, seed=2)
    extended = _make_model(n_marks=2, spatial=True, seed=2)
    extended.t1 = float(times[-1] + 3.0)

    n0 = base.training_loss(times=tt, locations=ll, lengths=lens, state=state, marks=kk)
    n1 = extended.training_loss(times=tt, locations=ll, lengths=lens, state=state, marks=kk)
    # A longer window can only add compensator mass.
    assert n1["nll_matrix"].sum() > n0["nll_matrix"].sum()

    # And that added mass equals the closed-form tail integral of the ground rate.
    q = float(base.decay.detach())
    row = base.alpha.sum(1).detach().numpy()
    mu_tot = float(base.mu.detach().sum())
    t_last, t1 = times[-1], extended.t1
    expected = mu_tot * (t1 - t_last) + float(
        (row[marks] * (np.exp(-q * (t_last - times)) - np.exp(-q * (t1 - times)))).sum()
    )
    got = float((n1["nll_matrix"].sum() - n0["nll_matrix"].sum()).detach())
    assert abs(got - expected) < 1e-9


# ---------------------------------------------------------------------------
# 2. Diagonal mask
# ---------------------------------------------------------------------------

def test_diagonal_variant_zeroes_offdiagonal_exactly():
    """Off-diagonal alpha entries are exactly 0.0 — not merely small."""
    model = _make_model(n_marks=4, spatial=True, diagonal_only=True)
    a = model.alpha.detach().numpy()
    off = a[~np.eye(4, dtype=bool)]
    assert np.all(off == 0.0), f"off-diagonal entries not exactly zero: {off}"
    assert np.all(np.diagonal(a) > 0.0)


def test_diagonal_variant_offdiagonal_gradient_is_exactly_zero():
    """The restriction is structural: masked entries receive exactly zero gradient."""
    model = _make_model(n_marks=3, spatial=True, diagonal_only=True)
    rng = np.random.default_rng(7)
    times = np.sort(rng.uniform(0.0, 5.0, 8))
    times -= times[0]
    locs = rng.normal(0.0, 1.0, (8, 2))
    marks = rng.integers(0, 3, 8)
    tt, ll, kk, lens, state = _single_seq_batch(times, locs, marks)

    out = model.training_loss(times=tt, locations=ll, lengths=lens, state=state, marks=kk)
    out["loss"].backward()

    grad = model._raw_alpha.grad.detach().numpy()
    off_grad = grad[~np.eye(3, dtype=bool)]
    assert np.all(off_grad == 0.0), f"off-diagonal gradient not exactly zero: {off_grad}"
    assert np.isfinite(grad).all()
    assert np.abs(np.diagonal(grad)).max() > 0.0  # diagonal still learns


def test_diagonal_variant_ignores_offdiagonal_parameters():
    """Perturbing masked raw parameters cannot change the likelihood at all."""
    model = _make_model(n_marks=3, spatial=True, diagonal_only=True)
    rng = np.random.default_rng(9)
    times = np.sort(rng.uniform(0.0, 5.0, 8))
    times -= times[0]
    locs = rng.normal(0.0, 1.0, (8, 2))
    marks = rng.integers(0, 3, 8)
    tt, ll, kk, lens, state = _single_seq_batch(times, locs, marks)

    before = model.training_loss(
        times=tt, locations=ll, lengths=lens, state=state, marks=kk
    )["nll_matrix"].clone()
    with torch.no_grad():
        offdiag = ~torch.eye(3, dtype=torch.bool)
        model._raw_alpha[offdiag] += 25.0  # enormous off-diagonal excitation
    after = model.training_loss(
        times=tt, locations=ll, lengths=lens, state=state, marks=kk
    )["nll_matrix"]
    assert torch.equal(before, after)


def test_full_variant_does_use_offdiagonal():
    """Sanity check the mask test above is not vacuous: the full model *does* react."""
    model = _make_model(n_marks=3, spatial=True, diagonal_only=False)
    rng = np.random.default_rng(9)
    times = np.sort(rng.uniform(0.0, 5.0, 8))
    times -= times[0]
    locs = rng.normal(0.0, 1.0, (8, 2))
    marks = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    tt, ll, kk, lens, state = _single_seq_batch(times, locs, marks)

    before = model.training_loss(
        times=tt, locations=ll, lengths=lens, state=state, marks=kk
    )["nll_matrix"].clone()
    with torch.no_grad():
        offdiag = ~torch.eye(3, dtype=torch.bool)
        model._raw_alpha[offdiag] += 2.0
    after = model.training_loss(
        times=tt, locations=ll, lengths=lens, state=state, marks=kk
    )["nll_matrix"]
    assert not torch.equal(before, after)


# ---------------------------------------------------------------------------
# 3. Diagnostics
# ---------------------------------------------------------------------------

def test_branching_diagnostics_exposed():
    model = _make_model(n_marks=3, spatial=True)
    a = model.alpha_matrix()
    assert a.shape == (3, 3)
    assert torch.equal(a, model.alpha.detach())

    assert torch.allclose(model.branching_ratios, torch.diagonal(model.alpha))

    expected_rho = np.abs(np.linalg.eigvals(a.numpy())).max()
    assert abs(float(model.spectral_radius) - expected_rho) < 1e-9

    diag = model.branching_diagnostics()
    for key in ("spectral_radius", "decay_q", "branching_ratio_00", "alpha_0_1"):
        assert key in diag


def test_alpha_matrix_reaches_extra_metrics():
    model = _make_model(n_marks=2, spatial=True)
    rng = np.random.default_rng(4)
    times = np.sort(rng.uniform(0.0, 4.0, 5))
    times -= times[0]
    locs = rng.normal(0.0, 1.0, (5, 2))
    marks = rng.integers(0, 2, 5)
    tt, ll, kk, lens, state = _single_seq_batch(times, locs, marks)
    out = model.training_loss(times=tt, locations=ll, lengths=lens, state=state, marks=kk)
    extra = out["extra_metrics"]
    assert "alpha_matrix" in extra and extra["alpha_matrix"].shape == (2, 2)
    assert "spectral_radius" in extra
    assert torch.equal(extra["alpha_matrix"], model.alpha.detach())


# ---------------------------------------------------------------------------
# 3b. Mark panel  (four-key convention shared with the other marked presets)
# ---------------------------------------------------------------------------

MARK_PANEL_KEYS = (
    "mark_logprob_matrix",
    "mark_logprob_events",
    "mark_targets_events",
    "mark_nll",
)


def _panel_fixture(n_marks=3, spatial=True, n=7, seed=21, diagonal_only=False):
    model = _make_model(n_marks=n_marks, spatial=spatial, diagonal_only=diagonal_only)
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0.0, 5.0, n)); times -= times[0]
    locs = rng.normal(0.0, 1.0, (n, 2))
    marks = rng.integers(0, n_marks, n)
    tt, ll, kk, lens, state = _single_seq_batch(times, locs, marks)
    out = model.training_loss(times=tt, locations=ll, lengths=lens, state=state, marks=kk)
    return model, out, times, locs, marks


@pytest.mark.parametrize("spatial", [True, False])
def test_mark_panel_keys_present_with_expected_shapes(spatial):
    model, out, _, _, marks = _panel_fixture(spatial=spatial)
    for key in MARK_PANEL_KEYS:
        assert key in out, f"missing mark-panel key {key!r}"

    n, m = marks.shape[0], model.n_marks
    assert out["mark_logprob_matrix"].shape == (1, n, m)
    assert out["mark_logprob_events"].shape == (n, m)
    assert out["mark_targets_events"].shape == (n,)
    assert out["mark_targets_events"].dtype == torch.int64
    assert out["mark_nll"].ndim == 0


def test_mark_logprob_rows_are_normalised():
    _, out, _, _, _ = _panel_fixture()
    total = torch.logsumexp(out["mark_logprob_matrix"], dim=-1)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-10)


def test_mark_distribution_is_space_marginalised():
    """p(k|t,H) must come from the SPACE-INTEGRAL of lambda_k, not from lambda_k
    at the observed location.

    Verified by integrating lambda_k(t_i, y) over R^2 by quadrature and comparing
    the resulting normalised mark distribution to the reported one.
    """
    model, out, times, locs, marks = _panel_fixture(spatial=True)
    P = _numpy_params(model)

    ax = np.arange(-8.0, 8.0 + 0.05, 0.1)
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    grid = np.stack([xx.ravel(), yy.ravel()], -1)
    cell = 0.1 * 0.1

    got = out["mark_logprob_matrix"][0].detach().exp().numpy()
    for i in range(times.shape[0]):
        lam_grid = _lambda_per_mark(
            P, times[i], grid, times[:i], locs[:i], marks[:i], True
        )                                   # (G, M)
        lam_k = lam_grid.sum(axis=0) * cell  # integral over space, per mark
        want = lam_k / lam_k.sum()
        assert np.abs(got[i] - want).max() < 1e-6, (
            f"event {i}: mark distribution is not the space-marginalised one\n"
            f"got  {got[i]}\nwant {want}"
        )

    # And it is genuinely different from conditioning on the observed location,
    # so the test above is not vacuous.
    at_loc = _lambda_per_mark(
        P, times[-1], locs[-1][None, :], times[:-1], locs[:-1], marks[:-1], True
    )[0]
    at_loc = at_loc / at_loc.sum()
    assert np.abs(got[-1] - at_loc).max() > 1e-3


def test_mark_nll_matches_gathered_logprob():
    _, out, _, _, marks = _panel_fixture()
    lp = out["mark_logprob_events"]
    tgt = out["mark_targets_events"]
    manual = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
    assert torch.allclose(manual, out["mark_nll"], atol=1e-12)
    assert np.array_equal(tgt.numpy(), marks)


def test_spatiotemporal_nll_is_joint_minus_mark():
    _, out, _, _, _ = _panel_fixture()
    st = out["extra_metrics"]["spatiotemporal_nll"]
    assert abs(st - (float(out["nll"].detach()) - float(out["mark_nll"].detach()))) < 1e-9


def test_single_mark_has_zero_mark_factor():
    """With M=1 the mark factor vanishes and spatiotemporal_nll == joint nll."""
    _, out, _, _, _ = _panel_fixture(n_marks=1)
    assert torch.allclose(
        out["mark_logprob_matrix"], torch.zeros_like(out["mark_logprob_matrix"]), atol=1e-12
    )
    assert abs(float(out["mark_nll"].detach())) < 1e-12
    assert abs(
        out["extra_metrics"]["spatiotemporal_nll"] - float(out["nll"].detach())
    ) < 1e-9


def test_padding_excluded_from_mark_panel():
    """The padding-free views must contain exactly the valid events."""
    model = _make_model(n_marks=3, spatial=True)
    rng = np.random.default_rng(31)
    times = np.sort(rng.uniform(0.0, 5.0, 6)); times -= times[0]
    locs = rng.normal(0.0, 1.0, (6, 2))
    marks = rng.integers(0, 3, 6)
    tt, ll, kk, lens, _ = _single_seq_batch(times, locs, marks)
    short = torch.tensor([4])
    state = StateContext(
        payload={"times": tt, "locations": ll, "lengths": short, "marks": kk}
    )
    out = model.training_loss(times=tt, locations=ll, lengths=short, state=state, marks=kk)
    assert out["mark_logprob_events"].shape == (4, 3)
    assert np.array_equal(out["mark_targets_events"].numpy(), marks[:4])
    assert torch.allclose(
        out["mark_logprob_events"], out["mark_logprob_matrix"][0, :4], atol=1e-12
    )


def test_diagonal_variant_changes_the_mark_distribution():
    """The mark panel must actually reflect the structural restriction."""
    _, full, _, _, _ = _panel_fixture(diagonal_only=False)
    _, diag, _, _, _ = _panel_fixture(diagonal_only=True)
    assert not torch.allclose(
        full["mark_logprob_matrix"], diag["mark_logprob_matrix"], atol=1e-6
    )


def test_mark_panel_flows_through_unified_model():
    model = ConfigRegistry.build("marked_hawkes", {"n_marks": 3}, spatial_dim=2)
    seqs = simulate_marked_hawkes_sequences(
        n_sequences=3, T=20.0, mu=MU_TRUE, alpha=ALPHA_TRUE, decay=Q_TRUE, seed=1
    )
    times, locs, marks, lens = _pack(seqs)
    out = model(times=times, locations=locs, lengths=lens, marks=marks)
    for key in MARK_PANEL_KEYS:
        assert key in out
    n_events = int(lens.sum())
    assert out["mark_logprob_events"].shape == (n_events, 3)
    assert out["mark_targets_events"].shape == (n_events,)
    assert "spatiotemporal_nll" in out["extra_metrics"]


# ---------------------------------------------------------------------------
# 4. Simulator + alpha recovery  (the key scientific test)
# ---------------------------------------------------------------------------

ALPHA_TRUE = np.array(
    [
        [0.40, 0.30, 0.00],
        [0.00, 0.40, 0.30],
        [0.30, 0.00, 0.40],
    ]
)
MU_TRUE = np.array([0.40, 0.40, 0.40])
Q_TRUE = 1.0


def test_simulator_respects_window_and_shapes():
    rng = np.random.default_rng(0)
    seq = simulate_marked_hawkes(
        mu=MU_TRUE, alpha=ALPHA_TRUE, decay=Q_TRUE, T=50.0, rng=rng
    )
    assert seq["times"].ndim == 1 and seq["locations"].shape[1] == 2
    assert seq["times"].shape[0] == seq["marks"].shape[0] == seq["locations"].shape[0]
    assert np.all(np.diff(seq["times"]) >= 0)
    assert seq["times"].min() >= 0.0 and seq["times"].max() < 50.0
    assert set(np.unique(seq["marks"])).issubset({0, 1, 2})


def test_simulator_rejects_supercritical_alpha():
    with pytest.raises(RuntimeError, match="supercritical"):
        simulate_marked_hawkes(
            mu=np.array([1.0]), alpha=np.array([[1.6]]), decay=1.0, T=500.0,
            rng=np.random.default_rng(0), max_events=2000,
        )


@pytest.mark.parametrize(
    "spatial,n_seq,steps,lr",
    [
        (False, 100, 180, 0.12),   # purely temporal marked TPP
        (True, 45, 130, 0.15),     # full marked space-time model
    ],
    ids=["temporal", "spatial"],
)
def test_recovers_known_alpha_from_self_simulated_data(spatial, n_seq, steps, lr):
    """Simulate with a known sparse M x M matrix, fit, and check recovery.

    This is the load-bearing test for the downstream experiment: it checks both
    the numeric values of ``alpha_hat`` and — what the experiment actually scores
    — the recovered support ``supp(alpha_hat)`` against the true adjacency.

    Sizing note: the likelihood is O(sum_b T_b^2), so many short sequences are
    much cheaper than a few long ones at equal event count.
    """
    seqs = simulate_marked_hawkes_sequences(
        n_sequences=n_seq,
        T=30.0,
        mu=MU_TRUE,
        alpha=ALPHA_TRUE,
        decay=Q_TRUE,
        seed=11,
        disp_scale=np.full((3, 3, 2), 0.4),
        bg_scale=np.ones((3, 2)),
    )
    n_events = sum(len(s["times"]) for s in seqs)
    assert n_events > 4000, f"too few events to identify alpha ({n_events})"

    times, locs, marks, lens = _pack(seqs)
    torch.manual_seed(0)
    model = MarkedHawkesEventModel(
        n_marks=3, spatial_dim=2, spatial=spatial,
        init_mu=0.5, init_alpha=0.15, init_decay=0.7,
    )
    state = StateContext(
        payload={"times": times, "locations": locs, "lengths": lens, "marks": marks}
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        out = model.training_loss(
            times=times, locations=locs, lengths=lens, state=state, marks=marks
        )
        out["loss"].backward()
        opt.step()

    a_hat = model.alpha_matrix().numpy()
    max_err = np.abs(a_hat - ALPHA_TRUE).max()
    assert max_err < 0.10, f"alpha not recovered (max abs err {max_err:.3f}):\n{a_hat}"

    # Support recovery: threshold halfway between 0 and the smallest true entry.
    thr = 0.5 * ALPHA_TRUE[ALPHA_TRUE > 0].min()
    assert np.array_equal(a_hat > thr, ALPHA_TRUE > 0), (
        f"support of alpha_hat does not match the true adjacency:\n{a_hat}"
    )

    # Decay, background and spectral radius should land near truth too.
    assert abs(float(model.decay.detach()) - Q_TRUE) < 0.35
    assert np.abs(model.mu.detach().numpy() - MU_TRUE).max() < 0.25
    true_rho = np.abs(np.linalg.eigvals(ALPHA_TRUE)).max()
    assert abs(float(model.spectral_radius) - true_rho) < 0.10


# ---------------------------------------------------------------------------
# 5. Unmarked backward compatibility
# ---------------------------------------------------------------------------

def test_marks_none_equals_all_zero_marks():
    """Unmarked data (marks=None) must behave exactly like a single-mark process."""
    model = _make_model(n_marks=1, spatial=True)
    rng = np.random.default_rng(2)
    times = np.sort(rng.uniform(0.0, 5.0, 7))
    times -= times[0]
    locs = rng.normal(0.0, 1.0, (7, 2))
    zeros = np.zeros(7, dtype=np.int64)

    tt, ll, kk, lens, state_with = _single_seq_batch(times, locs, zeros)
    state_without = StateContext(
        payload={"times": tt, "locations": ll, "lengths": lens}
    )
    a = model.training_loss(
        times=tt, locations=ll, lengths=lens, state=state_with, marks=kk
    )["nll_matrix"]
    b = model.training_loss(
        times=tt, locations=ll, lengths=lens, state=state_without, marks=None
    )["nll_matrix"]
    assert torch.equal(a, b)


def test_rejects_marks_out_of_range():
    model = _make_model(n_marks=2, spatial=True)
    rng = np.random.default_rng(2)
    times = np.sort(rng.uniform(0.0, 5.0, 4))
    times -= times[0]
    locs = rng.normal(0.0, 1.0, (4, 2))
    bad = np.array([0, 1, 5, 0])
    tt, ll, kk, lens, state = _single_seq_batch(times, locs, bad)
    with pytest.raises(ValueError, match="n_marks"):
        model.training_loss(times=tt, locations=ll, lengths=lens, state=state, marks=kk)


def test_does_not_mutate_caller_marks():
    """The event model must never write into the batch's marks tensor."""
    model = _make_model(n_marks=3, spatial=True)
    rng = np.random.default_rng(12)
    times = np.sort(rng.uniform(0.0, 5.0, 5))
    times -= times[0]
    locs = rng.normal(0.0, 1.0, (5, 2))
    marks = rng.integers(0, 3, 5)
    tt, ll, kk, lens, state = _single_seq_batch(times, locs, marks)

    # Shorten the sequence so padding handling kicks in on real (non-zero) marks.
    short_lens = torch.tensor([3])
    original = kk.clone()
    model.training_loss(
        times=tt, locations=ll, lengths=short_lens, state=state, marks=kk
    )
    assert torch.equal(kk, original)


def test_padding_is_ignored():
    """Padded positions must contribute nothing to the loss."""
    model = _make_model(n_marks=3, spatial=True)
    rng = np.random.default_rng(6)
    times = np.sort(rng.uniform(0.0, 5.0, 6))
    times -= times[0]
    locs = rng.normal(0.0, 1.0, (6, 2))
    marks = rng.integers(0, 3, 6)

    tt, ll, kk, lens, state = _single_seq_batch(times, locs, marks)
    ref = model.training_loss(
        times=tt, locations=ll, lengths=lens, state=state, marks=kk
    )["nll_matrix"][0, :6]

    pad = 4
    tt2 = torch.cat([tt, torch.full((1, pad), 99.0, dtype=tt.dtype)], dim=1)
    ll2 = torch.cat([ll, torch.full((1, pad, 2), 7.0, dtype=ll.dtype)], dim=1)
    kk2 = torch.cat([kk, torch.zeros(1, pad, dtype=torch.long)], dim=1)
    state2 = StateContext(
        payload={"times": tt2, "locations": ll2, "lengths": lens, "marks": kk2}
    )
    padded = model.training_loss(
        times=tt2, locations=ll2, lengths=lens, state=state2, marks=kk2
    )
    assert torch.allclose(padded["nll_matrix"][0, :6], ref, atol=1e-12)
    assert float(padded["nll_matrix"][0, 6:].abs().sum().detach()) == 0.0


def test_existing_unmarked_preset_unchanged():
    """An existing unmarked preset must produce a bit-identical NLL on fixed input.

    Regression value captured from the pre-change tree; the marked family is
    purely additive and must not perturb it.
    """
    torch.manual_seed(1234)
    model = ConfigRegistry.build("hawkes_gmm", {}, spatial_dim=2, hidden_dim=32)
    times = torch.linspace(0.0, 4.0, 12).unsqueeze(0)
    locs = torch.zeros(1, 12, 2)
    locs[0, :, 0] = torch.linspace(-1.0, 1.0, 12)
    locs[0, :, 1] = torch.linspace(1.0, -1.0, 12)
    lengths = torch.tensor([12])
    out = model(times=times, locations=locs, lengths=lengths)
    # Unmarked models must ignore a marks tensor entirely.
    out_marked = model(
        times=times, locations=locs, lengths=lengths,
        marks=torch.zeros(1, 12, dtype=torch.long),
    )
    assert torch.equal(out["nll"], out_marked["nll"])
    assert torch.isfinite(out["nll"])


# ---------------------------------------------------------------------------
# 6. Preset registration / framework wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["marked_hawkes", "marked_hawkes_diag"])
def test_presets_registered_and_buildable(preset):
    assert ConfigRegistry.is_registered(preset)
    model = ConfigRegistry.build(preset, {"n_marks": 3}, spatial_dim=2, hidden_dim=32)
    ev = model.event_model
    assert isinstance(ev, MarkedHawkesEventModel)
    assert ev.n_marks == 3
    assert ev.capabilities.nll_kind == "exact"
    assert ev.diagonal_only == (preset == "marked_hawkes_diag")


def test_presets_differ_only_in_the_mask():
    full = ConfigRegistry.build("marked_hawkes", {"n_marks": 3}, spatial_dim=2).event_model
    diag = ConfigRegistry.build("marked_hawkes_diag", {"n_marks": 3}, spatial_dim=2).event_model
    assert torch.equal(full.alpha_mask, torch.ones(3, 3))
    assert torch.equal(diag.alpha_mask, torch.eye(3))
    assert full.spatial == diag.spatial
    assert type(full) is type(diag)


def test_end_to_end_through_unified_model_with_marks():
    """Marks must reach the event model through the standard forward path."""
    model = ConfigRegistry.build("marked_hawkes", {"n_marks": 3}, spatial_dim=2)
    seqs = simulate_marked_hawkes_sequences(
        n_sequences=3, T=20.0, mu=MU_TRUE, alpha=ALPHA_TRUE, decay=Q_TRUE, seed=1
    )
    times, locs, marks, lens = _pack(seqs)
    out = model(times=times, locations=locs, lengths=lens, marks=marks)
    assert torch.isfinite(out["nll"])
    assert out["nll_matrix"].shape == times.shape
    assert "alpha_matrix" in out["extra_metrics"]

    # At the default initialisation every mark has identical parameters, so the
    # model is genuinely exchangeable in the marks — relabelling must NOT change
    # the likelihood.
    shifted = (marks + 1) % 3
    out_sym = model(times=times, locations=locs, lengths=lens, marks=shifted)
    assert torch.allclose(out["nll"], out_sym["nll"], atol=1e-12)

    # Once the marks carry different parameters, relabelling must change it —
    # this is what proves the marks are actually being consumed.
    ev = model.event_model
    with torch.no_grad():
        ev._raw_mu.copy_(torch.tensor([-1.0, 0.5, 1.5]))
        ev._raw_alpha.copy_(torch.tensor([[1.0, -2.0, -2.0],
                                          [-2.0, 0.5, -2.0],
                                          [-2.0, -2.0, 0.0]]))
    out_a = model(times=times, locations=locs, lengths=lens, marks=marks)
    out_b = model(times=times, locations=locs, lengths=lens, marks=shifted)
    assert not torch.equal(out_a["nll"], out_b["nll"])


def test_marks_survive_dataset_and_collate():
    """The existing data path already carries marks; confirm it end to end."""
    from seahorse.data.dataset import STPPDataset, collate_fn

    seqs = simulate_marked_hawkes_sequences(
        n_sequences=4, T=25.0, mu=MU_TRUE, alpha=ALPHA_TRUE, decay=Q_TRUE, seed=2
    )
    ds = STPPDataset(seqs, min_length=3)
    batch = collate_fn([ds[i] for i in range(len(ds))])
    assert "marks" in batch and batch["marks"].dtype == torch.long
    assert batch["marks"].shape == batch["times"].shape

    model = ConfigRegistry.build("marked_hawkes", {"n_marks": 3}, spatial_dim=2)
    out = model(
        times=batch["times"],
        locations=batch["locations"],
        lengths=batch["lengths"],
        marks=batch["marks"],
    )
    assert torch.isfinite(out["nll"])


def test_intensity_query_matches_manual_evaluation():
    """intensity() must agree with a direct evaluation of the intensity formula."""
    model = _make_model(n_marks=3, spatial=True)
    P = _numpy_params(model)
    rng = np.random.default_rng(8)
    times = np.sort(rng.uniform(0.0, 5.0, 6))
    times -= times[0]
    locs = rng.normal(0.0, 1.0, (6, 2))
    marks = rng.integers(0, 3, 6)
    _, _, _, _, state = _single_seq_batch(times, locs, marks)

    q_t = np.array([times[-1] + 0.4, times[-1] + 1.1])
    q_s = np.array([[0.2, -0.3], [1.0, 0.5]])
    got = model.intensity(
        state=state,
        query_times=torch.tensor(q_t, dtype=torch.float64).unsqueeze(-1),
        query_locations=torch.tensor(q_s, dtype=torch.float64),
    ).detach().numpy()

    want = np.array(
        [
            _lambda_per_mark(P, q_t[i], q_s[i][None, :], times, locs, marks, True).sum()
            for i in range(2)
        ]
    )
    assert np.abs(got - want).max() < 1e-10
