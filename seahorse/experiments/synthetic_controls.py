"""Deterministic synthetic controls for point-process campaign preparation.

The generators return independent observation windows; train/validation/test are
separately seeded and no corpus is materialized unless this module's CLI is run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np


SUPPORT = ((0.0, 1.0), (0.0, 1.0))


@dataclass(frozen=True)
class SyntheticRecipe:
    family: Literal["homogeneous_poisson", "moving_seasonal_poisson", "gamma_renewal"]
    horizon: float = 1.0
    rate: float = 20.0
    seasonal_amplitude: float = 0.5
    seasonal_period: float = 1.0
    hotspot_weight: float = 0.8
    hotspot_kappa: float = 8.0
    renewal_shape: float = 2.0

    def __post_init__(self):
        if self.horizon <= 0 or self.rate <= 0:
            raise ValueError("horizon and rate must be positive.")
        if not 0 <= self.seasonal_amplitude < 1:
            raise ValueError("seasonal_amplitude must lie in [0, 1).")
        if not 0 <= self.hotspot_weight <= 1 or self.renewal_shape <= 1:
            raise ValueError("hotspot_weight must be in [0,1], renewal_shape must exceed one.")


def _record(times: np.ndarray, locations: np.ndarray, recipe: SyntheticRecipe) -> dict:
    return {"start": 0.0, "end": recipe.horizon, "times": times.tolist(), "locations": locations.tolist(),
            "support": [[0., 1.], [0., 1.]], "family": recipe.family}


def _homogeneous(recipe: SyntheticRecipe, rng: np.random.Generator) -> dict:
    n = rng.poisson(recipe.rate * recipe.horizon)
    return _record(np.sort(rng.uniform(0, recipe.horizon, n)), rng.uniform(0, 1, (n, 2)), recipe)


def seasonal_rate(t: np.ndarray | float, recipe: SyntheticRecipe) -> np.ndarray:
    """Integrated spatial intensity at time t (events/time)."""
    t = np.asarray(t)
    return recipe.rate * (1 + recipe.seasonal_amplitude * np.sin(2 * np.pi * t / recipe.seasonal_period))


def moving_hotspot_center(t: np.ndarray | float, recipe: SyntheticRecipe) -> np.ndarray:
    t = np.asarray(t)
    phase = 2 * np.pi * t / recipe.seasonal_period
    return np.stack((.5 + .25 * np.cos(phase), .5 + .25 * np.sin(phase)), axis=-1) % 1.0


def _von_mises_pdf_unit(x: np.ndarray, center: np.ndarray, kappa: float) -> np.ndarray:
    angle = 2 * np.pi * (x - center)
    # Each dimension is von Mises on [0, 1), whose integral is one.
    return np.exp(kappa * np.cos(angle)).prod(axis=-1) / np.i0(kappa) ** 2


def moving_seasonal_intensity(t: np.ndarray | float, x: np.ndarray, recipe: SyntheticRecipe) -> np.ndarray:
    """Analytic intensity on time × [0,1]^2 for the moving-Poisson control."""
    t_arr = np.asarray(t); x_arr = np.asarray(x)
    density = (1 - recipe.hotspot_weight) + recipe.hotspot_weight * _von_mises_pdf_unit(x_arr, moving_hotspot_center(t_arr, recipe), recipe.hotspot_kappa)
    return seasonal_rate(t_arr, recipe) * density


def _moving_seasonal(recipe: SyntheticRecipe, rng: np.random.Generator) -> dict:
    bound = recipe.rate * (1 + recipe.seasonal_amplitude)
    candidate_n = rng.poisson(bound * recipe.horizon)
    candidates = rng.uniform(0, recipe.horizon, candidate_n)
    accept = rng.uniform(size=candidate_n) <= seasonal_rate(candidates, recipe) / bound
    times = np.sort(candidates[accept])
    centers = moving_hotspot_center(times, recipe)
    hot = rng.uniform(size=times.size) < recipe.hotspot_weight
    locations = rng.uniform(0, 1, (times.size, 2))
    if hot.any():
        locations[hot] = ((rng.vonmises(2 * np.pi * centers[hot], recipe.hotspot_kappa) / (2 * np.pi)) % 1.0)
    return _record(times, locations, recipe)


def _renewal(recipe: SyntheticRecipe, rng: np.random.Generator) -> dict:
    # Every observation window intentionally starts a fresh renewal process at t=0.
    # This tests inhibition without pretending the train/val/test windows continue one process.
    scale = 1.0 / (recipe.rate * recipe.renewal_shape)
    waits, t = [], 0.0
    while True:
        t += float(rng.gamma(recipe.renewal_shape, scale))
        if t > recipe.horizon:
            break
        waits.append(t)
    times = np.asarray(waits, dtype=float)
    return _record(times, rng.uniform(0, 1, (times.size, 2)), recipe)


def generate_windows(recipe: SyntheticRecipe, n_windows: int, *, seed: int) -> list[dict]:
    """Generate independent fixed-support windows, retaining empty windows."""
    if n_windows < 0:
        raise ValueError("n_windows must be non-negative.")
    rng = np.random.default_rng(seed)
    fn: Callable[[SyntheticRecipe, np.random.Generator], dict] = {
        "homogeneous_poisson": _homogeneous, "moving_seasonal_poisson": _moving_seasonal, "gamma_renewal": _renewal,
    }[recipe.family]
    return [fn(recipe, rng) for _ in range(n_windows)]


def generate_splits(recipe: SyntheticRecipe, *, n_train: int, n_val: int, n_test: int, seed: int) -> dict[str, list[dict]]:
    """Separate SeedSequence children guarantee split reproducibility and independence."""
    child = np.random.SeedSequence(seed).spawn(3)
    return {"train": generate_windows(recipe, n_train, seed=int(child[0].generate_state(1)[0])),
            "val": generate_windows(recipe, n_val, seed=int(child[1].generate_state(1)[0])),
            "test": generate_windows(recipe, n_test, seed=int(child[2].generate_state(1)[0]))}


def ground_truth(recipe: SyntheticRecipe) -> dict:
    """Declare the available process truth without calling a count law a PP likelihood."""
    if recipe.family == "homogeneous_poisson":
        return {"intensity": "lambda(t,x)=rate on [0,horizon]x[0,1]^2", "expected_count": recipe.rate * recipe.horizon}
    if recipe.family == "moving_seasonal_poisson":
        integral = recipe.rate * (recipe.horizon + recipe.seasonal_amplitude * recipe.seasonal_period / (2 * np.pi)
                                  * (1 - np.cos(2 * np.pi * recipe.horizon / recipe.seasonal_period)))
        return {"intensity": "moving_seasonal_intensity(t,x,recipe)", "conditional_location_density": "uniform/von-Mises mixture", "expected_count": float(integral)}
    return {"conditional_waiting_law": f"Gamma(shape={recipe.renewal_shape}, scale={1/(recipe.rate*recipe.renewal_shape):.8g})", "window_reset": True}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create seeded synthetic-control windows.")
    parser.add_argument("--family", choices=["homogeneous_poisson", "moving_seasonal_poisson", "gamma_renewal"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--n-train", type=int, default=100); parser.add_argument("--n-val", type=int, default=25); parser.add_argument("--n-test", type=int, default=25)
    parser.add_argument("--horizon", type=float, default=1.); parser.add_argument("--rate", type=float, default=20.)
    parser.add_argument("--renewal-shape", type=float, default=2.)
    args = parser.parse_args(argv)
    recipe = SyntheticRecipe(family=args.family, horizon=args.horizon, rate=args.rate, renewal_shape=args.renewal_shape)
    payload = {"recipe": asdict(recipe), "ground_truth": ground_truth(recipe), "seed": args.seed,
               "splits": generate_splits(recipe, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, seed=args.seed)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
