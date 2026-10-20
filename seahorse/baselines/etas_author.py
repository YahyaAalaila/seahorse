"""Strict bootstrap adapter for the pinned author ETAS implementation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .provenance import ETAS_PROVENANCE


def _author_etas():
    try:
        from etas.evaluation import ETASLikelihoodCalculation  # type: ignore
        from etas.inversion import ETASParameterCalculation  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ETAS requires the pinned MIT author package plus its geo stack: "
            "git+https://github.com/lmizrahi/etas@097f08b69a4f06f9c38d14799dedfbd4543144e3"
        ) from exc
    return ETASParameterCalculation, ETASLikelihoodCalculation


@dataclass(frozen=True)
class ETASContract:
    auxiliary_start: str
    timewindow_start: str
    timewindow_end: str
    testwindow_end: str
    mc: float | str
    delta_m: float
    shape_coords: list[list[float]]
    beta: float | str | None = None
    m_ref: float | None = None
    coppersmith_multiplier: float = 10.0
    model_id: str = "seahorse_etas"

    def validate(self) -> None:
        dates = pd.to_datetime(
            [self.auxiliary_start, self.timewindow_start, self.timewindow_end, self.testwindow_end]
        )
        if not bool(np.all(np.diff(dates.view("i8")) > 0)):
            raise ValueError("require auxiliary_start < train start < test start < test end")
        if not self.shape_coords or len(self.shape_coords) < 3:
            raise ValueError("ETAS requires a finite target-region polygon [lat, lon]")
        if self.mc == "var" and self.m_ref is None:
            raise ValueError("ETAS mc='var' requires m_ref and catalog mc_current")
        if not self.model_id or any(ch.isspace() for ch in self.model_id):
            raise ValueError("ETAS model_id must be a nonempty path-safe token")


class ETASAuthorAdapter:
    """Preserves the author's inversion/evaluation estimator and its finite domain."""

    required_columns = ("id", "latitude", "longitude", "time", "magnitude")

    def __init__(self, contract: ETASContract):
        contract.validate()
        self.contract = contract
        self.metadata: dict[str, Any] | None = None

    def catalog_metadata(self, catalog: pd.DataFrame, *, data_path: str | Path) -> dict[str, Any]:
        missing = set(self.required_columns).difference(catalog.columns)
        if missing:
            raise ValueError(f"ETAS catalog missing required columns: {sorted(missing)}")
        if (
            catalog.id.duplicated().any()
            or not pd.to_datetime(catalog.time, errors="coerce").notna().all()
        ):
            raise ValueError("ETAS requires unique IDs and parseable occurrence times")
        for col in ("latitude", "longitude", "magnitude"):
            if not np.isfinite(pd.to_numeric(catalog[col], errors="coerce")).all():
                raise ValueError(f"ETAS requires finite {col}")
        # ``store_results`` preserves ``fn_catalog`` in its result JSON.  A
        # DataFrame-only inversion silently writes the *training-filtered*
        # catalog on save, leaving no held-out events for the author's
        # evaluator.  Materialize the original full auxiliary/train/test
        # catalog once and hand the author code its normal file route.
        data_dir = Path(data_path)
        data_dir.mkdir(parents=True, exist_ok=True)
        catalog_path = data_dir / "catalog_input.csv"
        catalog.to_csv(catalog_path, index=True)
        md = {
            **{key: value for key, value in self.contract.__dict__.items() if key != "model_id"},
            "id": self.contract.model_id,
            "fn_catalog": str(catalog_path),
            "data_path": str(data_dir) + "/",
        }
        self.metadata = md
        return md

    def fit(self, catalog: pd.DataFrame, *, data_path: str | Path) -> dict[str, Any]:
        metadata = self.catalog_metadata(catalog, data_path=data_path)
        calculation, _ = _author_etas()
        fit = calculation(metadata)
        fit.prepare()
        parameters = fit.invert()
        # The pinned implementation concatenates file names rather than using
        # pathlib, so preserve the directory separator supplied in metadata.
        fit.store_results(metadata["data_path"])
        return parameters

    def score_author_test_window(
        self, fitted_metadata: dict[str, Any], *, quadrature_n: int = 1_000_000
    ) -> dict[str, float]:
        _, likelihood = _author_etas()
        calc = likelihood(fitted_metadata)
        calc.prepare(n=int(quadrature_n))
        scores = calc.evaluate()
        # Author API returns {"nll", "tll", "sll"}, despite the old example
        # scripts assigning it as though it were a tuple.
        return {key: float(scores[key]) for key in ("nll", "tll", "sll")}

    def sample_continuation(
        self,
        auxiliary_catalog: pd.DataFrame,
        *,
        parameters: dict[str, Any],
        simulation_end: str,
        beta_main: float,
        **author_options: Any,
    ) -> pd.DataFrame:
        """Delegate continuation sampling to the author's ETAS simulator.

        The caller must supply the same finite polygon, magnitude convention,
        and retained history used for inversion.  This deliberately has no
        generic ``sample_next`` alias: ETAS simulates a finite continuation.
        """
        self.catalog_metadata(auxiliary_catalog, data_path=author_options.pop("data_path", "."))
        try:
            from etas.simulation import simulate_catalog_continuation  # type: ignore
            from shapely.geometry import Polygon
        except ImportError as exc:
            raise RuntimeError(
                "ETAS simulation needs the pinned author package and Shapely"
            ) from exc
        return simulate_catalog_continuation(
            auxiliary_catalog=auxiliary_catalog,
            auxiliary_start=pd.to_datetime(self.contract.auxiliary_start),
            auxiliary_end=pd.to_datetime(self.contract.timewindow_end),
            polygon=Polygon(self.contract.shape_coords),
            simulation_end=pd.to_datetime(simulation_end),
            parameters=parameters,
            mc=self.contract.mc,
            beta_main=beta_main,
            delta_m=self.contract.delta_m,
            **author_options,
        )

    def write_provenance(self, directory: str | Path) -> Path:
        return ETAS_PROVENANCE.write(directory)

    @staticmethod
    def load_fitted_metadata(path: str | Path) -> dict[str, Any]:
        """Load the author ``parameters_*.json`` result for finite-window scoring."""
        data = json.loads(Path(path).read_text())
        if "final_parameters" not in data:
            raise ValueError("expected author ETAS result with final_parameters")
        return data
