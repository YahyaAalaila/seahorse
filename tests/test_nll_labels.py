"""Likelihood labels must distinguish closed-form, quadrature and sample-based routes."""

from __future__ import annotations

import pytest

from seahorse.registry import build_model

CASES = {
    "auto_stpp": ("exact", "closed_form"),
    "deep_stpp": ("exact", "closed_form"),
    "rmtpp_gmm": ("exact", "closed_form"),
    "poisson_gmm": ("exact", "closed_form"),
    "nsmpp": ("quadrature", "quadrature"),
    "thp_gmm": ("quadrature", "quadrature"),
    "diffusion_stpp": ("approximate", "monte_carlo"),
    "smash": ("approximate", "surrogate"),
}


@pytest.mark.parametrize("preset", sorted(CASES))
def test_nll_label_and_path(preset):
    label, path = CASES[preset]
    caps = build_model(preset=preset, config={}, hidden_dim=8).event_model.capabilities
    assert caps.nll_label == label
    assert caps.nll_path == path
    if label == "quadrature":
        assert caps.nll_kind == "exact"
        assert "quadrature" in caps.nll_description


def test_dstpp_description_is_not_called_a_bound():
    caps = build_model(preset="diffusion_stpp", config={}, hidden_dim=8).event_model.capabilities
    assert "sample-KDE" in caps.nll_description
    assert "VB" not in caps.nll_footnote
