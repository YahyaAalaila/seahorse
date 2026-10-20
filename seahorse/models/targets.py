"""Explicit target selection for fixed-history conditional comparisons.

History encoders always see the complete input history. Only objective targets
are selected. This avoids training repeatedly on overlapping context events.
"""
from __future__ import annotations

import torch

from .abstractions import StateContext


def last_target_state(state, lengths):
    if not isinstance(state, StateContext):
        return state
    p = dict(state.payload)
    for prefix in ("diff", "smash"):
        if f"{prefix}_img" not in p:
            continue
        counts = (lengths - 1).clamp_min(0)
        selected = (counts.cumsum(0) - 1)[counts > 0]
        for suffix in ("img", "cond", "mark_targets"):
            key = f"{prefix}_{suffix}"
            if isinstance(p.get(key), torch.Tensor):
                p[key] = p[key][selected]
        p[f"{prefix}_total_events"] = p[f"{prefix}_total_events"].new_tensor(len(selected))
        p["last_target_selected"] = True
    if "glide_mask" in p:
        mask = torch.zeros_like(p["glide_mask"])
        active = lengths > 1
        mask[torch.arange(len(lengths), device=lengths.device)[active], lengths[active] - 2] = True
        p["glide_mask"] = mask
        p["glide_total_events"] = mask.sum().to(p["glide_total_events"])
        p["last_target_selected"] = True
    return StateContext(payload=p, kl_loss=state.kl_loss)


def last_target_output(output, lengths, *, target_selected=False):
    """Select exactly one scored target per nonempty conditional input window.

Reject unknown matrix layouts instead of assuming that a scalar loss describes
the last target. Regularization terms retain their original declared scaling.
    """
    if target_selected:
        return output
    matrix = output.get("nll_matrix")
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
        raise ValueError("Last-target training requires explicit eventwise NLL terms or a target-aware objective.")
    indices = output.get("eventwise_target_indices")
    original_mask = output.get("mask", torch.ones_like(matrix, dtype=torch.bool)).bool()
    if indices is None:
        width, max_length = matrix.shape[1], int(lengths.max())
        if width == max_length:
            offset = 0
        elif width == max_length - 1:
            offset = 1
        else:
            raise ValueError("Ambiguous eventwise NLL layout: eventwise_target_indices is required.")
        indices = torch.arange(width, device=matrix.device)[None].expand_as(matrix) + offset
    mask = original_mask & (indices == lengths[:, None] - 1) & (lengths[:, None] > 1)
    if not torch.equal(mask.sum(1), (lengths > 1).long()):
        raise ValueError("The model did not score the common final target of every context.")
    count = mask.sum()
    result = dict(output)
    old_nll = output["nll"]
    nll = torch.where(mask, matrix, 0.).sum() / count.clamp_min(1)
    result.update(nll=nll, loss=nll + (output.get("loss", old_nll) - old_nll),
                  total_events=count, mask=mask,
                  nll_per_event=torch.where(mask, matrix, 0.).sum(1),
                  eventwise_target_indices=indices.masked_fill(~mask, -1))
    extra = dict(output.get("extra_metrics", {}))
    # Raw/native corrections are constants for each fitted affine transform.
    if "raw_space_nll" in extra:
        correction = extra.get("raw_reporting_correction")
        if correction is None:
            correction = torch.as_tensor(extra["raw_space_nll"], device=nll.device) - old_nll.detach()
        extra["raw_space_nll"] = nll.detach() + torch.as_tensor(correction, device=nll.device)
    for component in ("temporal", "spatial"):
        component_matrix = output.get(f"{component}_nll_matrix")
        if isinstance(component_matrix, torch.Tensor) and component_matrix.shape == matrix.shape:
            value = torch.where(mask, component_matrix, 0.).sum() / count.clamp_min(1)
            raw_key = f"raw_space_{component}_nll"
            if raw_key in extra and f"{component}_nll" in output:
                correction = torch.as_tensor(extra[raw_key], device=value.device) - torch.as_tensor(output[f"{component}_nll"], device=value.device).detach()
                extra[raw_key] = value.detach() + correction
            result[f"{component}_nll"] = value
        else:
            result.pop(f"{component}_nll", None)
            extra.pop(f"raw_space_{component}_nll", None)
    result["extra_metrics"] = extra
    return result
