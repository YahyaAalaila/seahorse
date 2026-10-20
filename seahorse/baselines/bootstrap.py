"""Auditable optional-dependency bootstrap specifications.

Installation is intentionally an operator action, not an import side effect.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionalBaselineDependency:
    name: str
    repository: str
    commit: str
    install_requirement: str
    extra_requirements: tuple[str, ...]


TRITPP_DEPENDENCY = OptionalBaselineDependency(
    name="TriTPP author implementation",
    repository="https://github.com/shchur/triangular-tpp",
    commit="a384a50caf6f489335c6829707e2d7abddb44492",
    install_requirement="git+https://github.com/shchur/triangular-tpp@a384a50caf6f489335c6829707e2d7abddb44492",
    extra_requirements=(),
)

ETAS_DEPENDENCY = OptionalBaselineDependency(
    name="ETAS author implementation",
    repository="https://github.com/lmizrahi/etas",
    commit="097f08b69a4f06f9c38d14799dedfbd4543144e3",
    install_requirement="git+https://github.com/lmizrahi/etas@097f08b69a4f06f9c38d14799dedfbd4543144e3",
    extra_requirements=("geopandas", "pynverse", "Shapely", "tabulate"),
)

WSM_DEPENDENCY = OptionalBaselineDependency(
    name="WSM_STPP author implementation",
    repository="https://github.com/KenCao2007/WSM_STPP",
    commit="e051e47ee39f8545f5d76154b7f240988546ffa0",
    install_requirement="git+https://github.com/KenCao2007/WSM_STPP@e051e47ee39f8545f5d76154b7f240988546ffa0",
    extra_requirements=(),
)
