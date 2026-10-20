"""Pinned upstream provenance for optional author baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuthorBaselineProvenance:
    name: str
    repository: str
    commit: str
    license: str
    adaptation: str
    score_contract: str

    def write(self, directory: str | Path) -> Path:
        path = Path(directory) / "baseline_provenance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return path


TRITPP_PROVENANCE = AuthorBaselineProvenance(
    name="TriTPP temporal flow + Seahorse spatial head",
    repository="https://github.com/shchur/triangular-tpp",
    commit="a384a50caf6f489335c6829707e2d7abddb44492",
    license="MIT",
    adaptation=(
        "Uses the author ttpp.models.TriTPP temporal flow on fixed [0,t_max] "
        "windows. Any spatial factor is a separately declared Seahorse head; "
        "this combined STPP was not published as TriTPP."
    ),
    score_contract="Fixed-horizon temporal log probability, including survival to t_max.",
)

ETAS_PROVENANCE = AuthorBaselineProvenance(
    name="Author ETAS",
    repository="https://github.com/lmizrahi/etas",
    commit="097f08b69a4f06f9c38d14799dedfbd4543144e3",
    license="MIT",
    adaptation=(
        "Calls the author's ETASParameterCalculation and ETASLikelihoodCalculation "
        "with an explicitly mapped earthquake catalog and polygon."
    ),
    score_contract=(
        "Author-returned event score only; not a validated common fixed-window "
        "likelihood. Campaign ETAS execution is blocked pending initial-compensator, "
        "terminal-survival and fitting-period resolution."
    ),
)

WSM_PROVENANCE = AuthorBaselineProvenance(
    name="WSM/AWSM with survival classification and same-backbone MLE",
    repository="https://github.com/KenCao2007/WSM_STPP",
    commit="e051e47ee39f8545f5d76154b7f240988546ffa0",
    license="unverified at pinned source; do not vendor author code",
    adaptation="Optional subprocess adapter to the author SMASH backbone; WSM and MLE share all architecture settings.",
    score_contract="Finite-horizon numerical likelihood with the author survival reconstruction enabled.",
)
