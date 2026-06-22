"""
Jump-ODE intensity process — continuous-time temporal point process engine.

Self-contained temporal point process owning:
  - learnable initial hidden state _init_state  (no encoder)
  - time-independent ODE: dh/dt = tanh(net(h)), zero-init last layer
  - configurable jump update on spatial location only:
      split  — update lower half of h from (h_lo, s), keep upper half as memory
      simple — tanh(Linear(cat(h, s)))
      gru    — GRUCell(input=s, hidden=h)
  - intensity: sigmoid(mlp(h) − 2.0) × 50   (2-layer MLP)
  - joint ODE: d/dt [h, Λ] = [f(h), λ(h(t))]
  - main API: sequence_nll_and_states(events, lengths) → (B,L), (B,L,h)

Used by neural_stpp_jump_sc and neural_stpp_attn_sc presets via NeuralSTPPStateModel.
The pre-jump hidden states h_seq_pre are fed directly to
JumpCNFSpatial / SelfAttentiveCNFSpatial as z_seq.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import os

try:
    from torchdiffeq import odeint as _odeint_std
    from torchdiffeq import odeint_adjoint as _odeint_adj

    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


def euler_solve(func, z0, t_span, n_steps: int = 50):
    """Simple Euler solver fallback when torchdiffeq is unavailable."""
    dt = (t_span[-1] - t_span[0]) / n_steps
    z = z0
    t = t_span[0]
    trajectory = [z0]
    target_times = t_span[1:]  # skip initial
    target_idx = 0
    for _ in range(n_steps):
        z = z + dt * func(t, z)
        t = t + dt
        while target_idx < len(target_times) and t >= target_times[target_idx] - 1e-6:
            trajectory.append(z)
            target_idx += 1
    while len(trajectory) < len(t_span):
        trajectory.append(z)
    return torch.stack(trajectory, dim=0)  # (T, B, h)


class _OdeDriftFn(nn.Module):
    """
    Time-independent hidden state ODE: dh/dt = tanh(net(h)).

    Faithful to original NeuralPointProcess dstate_net:
      - no time input (autonomous system)
      - tanh output bound prevents blow-up
      - zero-init last linear so dh/dt ≈ 0 at initialisation

    Args:
        hidden_dim:      input/output dimension of the hidden state.
        hidden_dims:     widths of the intermediate layers (default: [hidden_dim*2]).
                         Maps to ``--tpp_hdims`` in the original repo (e.g. [32, 32]).
        hidden_actfn:    activation applied between layers — ``"softplus"`` (paper
                         default via ``--tpp_actfn softplus``) or ``"tanh"``.
    """

    def __init__(
        self,
        hidden_dim: int,
        hidden_dims: list | tuple | int | str | None = None,
        hidden_actfn: str = "softplus",
    ):
        super().__init__()
        hidden_dims_list = self._normalize_hidden_dims(hidden_dims, hidden_dim)

        _ACTFNS = {"softplus": nn.Softplus, "tanh": nn.Tanh}
        if hidden_actfn not in _ACTFNS:
            raise ValueError(f"hidden_actfn must be one of {list(_ACTFNS)}; got {hidden_actfn!r}")
        act_cls = _ACTFNS[hidden_actfn]

        dims = [hidden_dim] + hidden_dims_list + [hidden_dim]
        layers: list = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())

        # Zero-init last layer so dh/dt ≈ 0 at init
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _normalize_hidden_dims(
        hidden_dims: list | tuple | int | str | None,
        hidden_dim: int,
    ) -> list[int]:
        """
        Accept multiple config shapes for ``--tpp_hdims`` style values.

        Supported forms:
          - None        -> [hidden_dim * 2] (compatibility default)
          - int         -> [int]
          - list/tuple  -> list(int(...))
          - str         -> "32-32", "32,32", or "[32, 32]"
        """
        if hidden_dims is None:
            return [hidden_dim * 2]

        if isinstance(hidden_dims, int):
            dims = [hidden_dims]
        elif isinstance(hidden_dims, str):
            raw = hidden_dims.strip().strip("[]()")
            if not raw:
                raise ValueError("hidden_dims string cannot be empty")
            if "-" in raw:
                parts = [p.strip() for p in raw.split("-")]
            elif "," in raw:
                parts = [p.strip() for p in raw.split(",")]
            else:
                parts = [raw]
            dims = [int(p) for p in parts if p]
        elif isinstance(hidden_dims, Iterable):
            dims = [int(d) for d in hidden_dims]
        else:
            raise TypeError(
                "hidden_dims must be None, int, str, or iterable of ints; "
                f"got {type(hidden_dims).__name__}"
            )

        if not dims:
            raise ValueError("hidden_dims must contain at least one layer width")
        if any(d <= 0 for d in dims):
            raise ValueError(f"hidden_dims must be positive; got {dims}")
        return dims

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        # t is accepted but ignored — ODE is time-independent
        return torch.tanh(self.net(h))


class _AugRHS(nn.Module):
    """
    Augmented ODE RHS: d/dt [h, Λ, E] = [f(h), λ(h(t)), ‖[dh, λ]‖²/dim].

    State layout: [h (hidden_dim) | Λ (1) | E (1)]
        Λ: compensator accumulator ∫ λ dt
        E: kinetic energy accumulator ∫ (‖dh‖² + λ²)/(h+1) dt
           Matches TimeVariableODE energy_regularization term from original repo.
           When energy_regularization=0 on the process, E is tracked but the
           loss contribution is zero (multiplied out in sequence_nll_and_states).

    Wrapped as nn.Module so that odeint_adjoint can traverse its parameters
    when use_adjoint=True.

    Intensity formula (faithful to IntensityODEFunc):
        λ(h) = sigmoid(intensity_net(h) − 2.0) × 50
    """

    def __init__(self, ode_func: _OdeDriftFn, intensity_net: nn.Module):
        super().__init__()
        self.ode_func = ode_func
        self.intensity_net = intensity_net

    def forward(self, t: Tensor, h_lam_e: Tensor) -> Tensor:
        """
        Args:
            t:       scalar — current time (passed by ODE solver, ignored)
            h_lam_e: (B, hidden_dim + 2) — [h | Λ | E]
        Returns:
            (B, hidden_dim + 2) — [dh/dt | λ(h) | d_energy/dt]
        """
        h = h_lam_e[:, :-2]                                              # (B, h)
        dh = self.ode_func(t, h)                                         # (B, h)
        lam = torch.sigmoid(self.intensity_net(h).squeeze(-1) - 2.0) * 50.0  # (B,)

        # Kinetic energy rate per sample: (‖dh‖² + λ²) / (h_dim + 1)
        # Detached so energy tracking does not create extra gradient paths
        # through the ODE parameters (consistent with original implementation).
        d_energy = (
            dh.detach().pow(2).sum(-1) + lam.detach().pow(2)
        ) / (h.shape[-1] + 1)                                           # (B,)

        return torch.cat([dh, lam.unsqueeze(-1), d_energy.unsqueeze(-1)], dim=-1)


class HiddenStateUpdate(nn.Module):
    """
    Configurable jump update for the temporal point process.

    Three variants matching the original NeuralPointProcess update_state family:

    'split'  — split h into two halves; update lower half from (h_lo, s),
               keep upper half as persistent memory:
                   h_lo' = tanh(W [h_lo; s] + b)
                   h'    = [h_lo' ; h_hi]

    'simple' — single linear+tanh on full concat(h, s):
                   h' = tanh(W [h; s] + b)

    'gru'    — GRUCell with spatial location as input, h as hidden state:
                   h' = GRUCell(input=s, hidden=h)
               (original "gru" variant of get_update_state)
    """

    def __init__(self, hidden_dim: int, spatial_dim: int, update_type: str = "gru"):
        super().__init__()
        self.update_type = update_type
        self.hidden_dim = hidden_dim

        if update_type == "split":
            half = hidden_dim // 2
            self.net = nn.Linear(half + spatial_dim, half)
        elif update_type == "simple":
            self.net = nn.Linear(hidden_dim + spatial_dim, hidden_dim)
        elif update_type == "gru":
            self.cell = nn.GRUCell(spatial_dim, hidden_dim)
        else:
            raise ValueError(
                f"Unknown update_type: {update_type!r}. "
                "Use 'split', 'simple', or 'gru'."
            )

    def forward(self, h: Tensor, s: Tensor) -> Tensor:
        """
        Args:
            h: (B, hidden_dim) — pre-jump hidden state
            s: (B, spatial_dim) — event location (spatial-only, no time)
        Returns:
            h_post: (B, hidden_dim)
        """
        if self.update_type == "split":
            half = self.hidden_dim // 2
            h_lo, h_hi = h[:, :half], h[:, half:]
            h_lo_new = torch.tanh(self.net(torch.cat([h_lo, s], dim=-1)))
            return torch.cat([h_lo_new, h_hi], dim=-1)
        elif self.update_type == "simple":
            return torch.tanh(self.net(torch.cat([h, s], dim=-1)))
        else:  # gru
            return self.cell(s, h)


class JumpOdeIntensityProcess(nn.Module):
    """
    Jump-ODE intensity process: continuous-time temporal point process engine.

    Owns the complete temporal generative model:
        h(0)       = _init_state                      (learnable, no encoder)
        dh/dt      = tanh(net(h))                     (time-independent ODE)
        λ(t)       = sigmoid(mlp(h(t)) − 2.0)×50    (2-layer MLP)
        Λ(a, b)    = ∫_a^b λ(h(s)) ds               (augmented ODE)
        h(t_i^+)   = update(h(t_i^-), s_i)           (spatial-only jump)

    Main API
    --------
    sequence_nll_and_states(events, lengths)
        → temporal_nll (B, L),  h_seq_pre (B, L, hidden_dim)

    h_seq_pre (pre-jump hidden states) is passed directly as z_seq to
    JumpCNFSpatial.sequence_nll / SelfAttentiveCNFSpatial.sequence_nll.

    Parameters
    ----------
    hidden_dim   : latent state dimension
    spatial_dim  : spatial coordinate dimension
    solver       : ODE solver (torchdiffeq method string, e.g. 'dopri5')
    atol / rtol  : ODE tolerances
    update_type  : 'split' | 'simple' | 'gru'
    use_adjoint  : if True, use odeint_adjoint for O(1)-memory backprop
                   (expensive in a sequential loop — default False)
    """

    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int,
        solver: str = "dopri5",
        atol: float = 1e-4,
        rtol: float = 1e-4,
        update_type: str = "gru",
        use_adjoint: bool = False,
        energy_regularization: float = 0.0,
        tpp_hidden_dims: list | None = None,
        tpp_actfn: str = "softplus",
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.use_adjoint = use_adjoint
        self.energy_regularization = energy_regularization

        # ── Learnable initial hidden state (no encoder) ──────────────────── #
        self._init_state = nn.Parameter(torch.zeros(hidden_dim))

        # ── Hidden state ODE: dh/dt = tanh(net(h)), time-independent ─────── #
        # tpp_hidden_dims: intermediate layer widths (--tpp_hdims in original repo).
        # tpp_actfn: activation for intermediate layers (--tpp_actfn, default softplus).
        self.ode_func = _OdeDriftFn(
            hidden_dim,
            hidden_dims=tpp_hidden_dims,
            hidden_actfn=tpp_actfn,
        )

        # ── Configurable jump update (spatial-only) ───────────────────────── #
        self.update = HiddenStateUpdate(hidden_dim, spatial_dim, update_type)

        # ── Intensity: sigmoid(mlp(h) − 2.0) × 50 ───────────────────────── #
        # 2-layer MLP matching original IntensityODEFunc:
        #   Linear(h → 4h) → Softplus → Linear(4h → 1)
        # Zero-init last layer: mlp(0) = 0 → sigmoid(−2)×50 ≈ 6.1 at init.
        _int_last = nn.Linear(hidden_dim * 4, 1)
        nn.init.zeros_(_int_last.weight)
        nn.init.zeros_(_int_last.bias)
        self.intensity_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Softplus(),
            _int_last,
        )

        # ── Augmented RHS for the joint [h, Λ] ODE ────────────────────────── #
        self.aug_func = _AugRHS(self.ode_func, self.intensity_net)
        # NOTE: _integrate uses a per-sample ODE loop so each batch element
        # gets its own [t_prev[b], t_curr[b]] span — equivalent to the original
        # TimeVariableODE reparameterisation without the [0,1] dummy variable.
        self._debug_neural_tpp = os.getenv("SEAHORSE_DEBUG_NSTPP", "0") == "1"
        self._debug_neural_tpp_max_calls = max(
            1, int(os.getenv("SEAHORSE_DEBUG_NSTPP_MAX_CALLS", "10"))
        )
        self._debug_neural_tpp_max_steps = max(
            1, int(os.getenv("SEAHORSE_DEBUG_NSTPP_MAX_STEPS", "3"))
        )
        self._debug_neural_tpp_calls = 0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _intensity(self, h: Tensor) -> Tensor:
        """λ(h) = sigmoid(linear(h) − 2.0) × 50.  Returns (B,)."""
        return torch.sigmoid(self.intensity_net(h).squeeze(-1) - 2.0) * 50.0

    def intensity_at(self, h: Tensor) -> Tensor:
        """Public API: λ(h) = sigmoid(intensity_net(h) − 2.0) × 50.  Returns (B,)."""
        return self._intensity(h)

    def _integrate(
        self, h_prev: Tensor, t_prev: Tensor, t_curr: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Solve d/dt [h, Λ, E] = [f(h), λ(h(t)), ‖[dh,λ]‖²/dim] per sample.

        Each batch element gets its own [t_prev[b], t_curr[b]] interval
        (equivalent to the original TimeVariableODE per-element ratio scaling).

        Args:
            h_prev: (B, hidden_dim) — post-jump state at the start of interval
            t_prev: (B, 1)          — interval start times
            t_curr: (B, 1)          — interval end times (next event)
        Returns:
            h_pre:  (B, hidden_dim) — pre-jump state at t_curr
            Lambda: (B,)            — compensator ∫_{t_prev}^{t_curr} λ(h(s)) ds
            energy: (B,)            — kinetic energy ∫ (‖dh‖²+λ²)/(h+1) dt
        """
        B = h_prev.shape[0]
        if B == 0:
            z = h_prev.new_zeros((0,))
            return h_prev, z, z

        odeint_fn = _odeint_adj if self.use_adjoint else _odeint_std
        h_pre_out = []
        lambda_out = []
        energy_out = []

        for b in range(B):
            h_prev_b = h_prev[b : b + 1]  # (1, h)
            zeros_b  = torch.zeros(1, 1, device=h_prev.device, dtype=h_prev.dtype)
            # State: [h | Λ | E],  all start at 0 for Λ and E
            h_aug_0_b = torch.cat([h_prev_b, zeros_b, zeros_b], dim=-1)  # (1, h+2)

            t0 = t_prev[b].reshape(1)  # (1,)
            t1 = t_curr[b].reshape(1)  # (1,)
            t1 = torch.where(t1 > t0, t1, t0 + 1e-6)
            t_span_b = torch.cat([t0, t1])  # (2,)

            if HAS_TORCHDIFFEQ:
                traj_b = odeint_fn(
                    self.aug_func,
                    h_aug_0_b,
                    t_span_b,
                    method=self.solver,
                    atol=self.atol,
                    rtol=self.rtol,
                    options={"dtype": h_aug_0_b.dtype},
                )
            else:
                traj_b = euler_solve(self.aug_func, h_aug_0_b, t_span_b, n_steps=50)

            h_aug_final_b = traj_b[-1]                   # (1, h+2)
            h_pre_out.append(h_aug_final_b[:, :-2])      # (1, h)
            lambda_out.append(h_aug_final_b[:, -2])      # (1,)
            energy_out.append(h_aug_final_b[:, -1])      # (1,)

        h_pre  = torch.cat(h_pre_out,  dim=0)   # (B, h)
        Lambda = torch.cat(lambda_out, dim=0)   # (B,)
        energy = torch.cat(energy_out, dim=0)   # (B,)
        return h_pre, Lambda, energy

    # ------------------------------------------------------------------ #
    # Main API                                                             #
    # ------------------------------------------------------------------ #

    def sequence_nll_and_states(
        self, events: Tensor, lengths: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full forward pass over a batch of sequences.

        Args:
            events:  (B, N, 1+d)  — cat([t.unsqueeze(-1), s], dim=-1)
            lengths: (B,)          — true sequence lengths
        Returns:
            temporal_nll: (B, L)   — per-position −log f*(t_i | H_i)
            h_seq_pre:    (B, L, hidden) — pre-jump hidden states at t_1..t_L
                          Fed directly as z_seq to JumpCNF / SelfAttentiveCNF.
            energy_reg:   scalar   — energy_regularization × mean kinetic energy
                          Matches --tpp_otreg_strength from original repo.
                          Zero when energy_regularization=0.0 (default).
        """
        B, N, _ = events.shape
        times = events[:, :, :1]   # (B, N, 1)
        locs = events[:, :, 1:]    # (B, N, d)

        L = int(lengths.max().item()) - 1  # prediction positions

        # Broadcast learnable init state — no encoder call needed
        h = self._init_state.unsqueeze(0).expand(B, -1).contiguous()  # (B, h)

        h_seq_pre_list = []
        nll_list = []
        energy_list = []
        debug_this_call = (
            self._debug_neural_tpp
            and self._debug_neural_tpp_calls < self._debug_neural_tpp_max_calls
        )

        if debug_this_call:
            print(
                f"[NSTPP-DEBUG][temporal] "
                f"B={B} N={N} L={L} lengths={lengths.detach().cpu().tolist()}"
            )

        for i in range(L):
            t_prev_i = times[:, i, :]      # (B, 1)
            t_curr_i = times[:, i + 1, :]  # (B, 1)
            s_curr_i = locs[:, i + 1, :]   # (B, d)

            active = lengths > (i + 1)  # event i+1 exists for this sequence
            # Joint ODE: evolve over per-sample intervals; inactive samples use
            # a zero-length interval (t_curr=t_prev) and are masked out below.
            t_curr_eff = torch.where(active.unsqueeze(-1), t_curr_i, t_prev_i)
            h_pre, Lambda, energy_i = self._integrate(h, t_prev_i, t_curr_eff)
            energy_list.append(energy_i * active.float())

            # Temporal NLL: −log λ(h(t_i^−)) + Λ(t_{i−1}, t_i)
            log_lam = torch.log(self._intensity(h_pre) + 1e-8)  # (B,)
            nll_i = -log_lam + Lambda                            # (B,)
            nll_i = torch.where(active, nll_i, torch.zeros_like(nll_i))

            if debug_this_call and i < self._debug_neural_tpp_max_steps:
                dt_all = (t_curr_i - t_prev_i).squeeze(-1).detach()
                dt = dt_all[active]
                dt0 = dt[0]
                max_dt_delta = (dt - dt0).abs().max().item()
                lmb = Lambda[active]
                ll = log_lam[active]
                ni = nll_i[active]
                print(
                    f"[NSTPP-DEBUG][temporal][step={i}] "
                    f"active={int(active.sum().item())} "
                    f"dt0={dt0.item():.6f} dt_mean={dt.mean().item():.6f} "
                    f"dt_max_abs_diff_from_dt0={max_dt_delta:.6f} "
                    f"Lambda(min/mean/max)="
                    f"{lmb.min().item():.6f}/{lmb.mean().item():.6f}/{lmb.max().item():.6f} "
                    f"log_lam(min/mean/max)="
                    f"{ll.min().item():.6f}/{ll.mean().item():.6f}/{ll.max().item():.6f} "
                    f"nll(min/mean/max)="
                    f"{ni.min().item():.6f}/{ni.mean().item():.6f}/{ni.max().item():.6f}"
                )

            h_seq_pre_list.append(h_pre)
            nll_list.append(nll_i)

            # Spatial-only jump update: h_post = update(h_pre, s_i)
            h_candidate = self.update(h_pre, s_curr_i)
            h = torch.where(active.unsqueeze(-1), h_candidate, h)

        h_seq_pre    = torch.stack(h_seq_pre_list, dim=1)  # (B, L, h)
        temporal_nll = torch.stack(nll_list,       dim=1)  # (B, L)

        # Kinetic energy regularization (--tpp_otreg_strength in original repo).
        # energy_list[i]: (B,) — per-sample energy for interval i (masked).
        # Normalise by total active event count to stay on the same scale as NLL.
        energy_mat   = torch.stack(energy_list, dim=1)     # (B, L)
        n_active     = energy_mat.gt(0).float().sum().clamp(min=1)
        energy_reg   = self.energy_regularization * energy_mat.sum() / n_active

        if debug_this_call:
            self._debug_neural_tpp_calls += 1
        return temporal_nll, h_seq_pre, energy_reg, h
