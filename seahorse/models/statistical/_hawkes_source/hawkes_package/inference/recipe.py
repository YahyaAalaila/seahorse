"""Save a fit as the recipe that produced it, rather than as the fit itself.

Two products are possible here and they are not the same. Storing the posterior
cloud is what a user wants when the data cannot be shared. Storing the
**recipe** -- configuration, seed, and a reference to the data -- is tiny, exact
on the temporal path, and needs no pickling. This module is the second, and the
reason it is the one that shipped first is structural rather than a preference.

**`ProcessModel` cannot be pickled**, and that is not an oversight to work
around. It holds three closures -- the builder, the branching callable and the
support -- which is why it carries a hand-written ``__repr__``: they print as
addresses. A model named by string can be rebuilt by calling the factory again;
a model handed in as an object cannot be reconstructed from anything this module
could write down, and :func:`to_recipe` says so rather than writing a file that
looks complete.

**The version stamp is checked on load.** A stamp that is written and never
compared is worse than none, because it implies a check that is not happening.
Loading a recipe written by a *newer* release warns, since this release cannot
know what changed; loading one from an older release does not, because the
recipe format is what has to stay readable and it is versioned here.

What a recipe does **not** contain is the data. The events are the user's, and a
recipe carries a reference -- a path, a URL, a DOI -- that the reader resolves.
That keeps the artefact small enough to paste into a paper's appendix, which is
the point of it.

.. versionadded:: 1.0.0
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .estimator import HawkesEstimator
from .priors import (
    ConstrainedPrior,
    Gamma,
    IndependentPrior,
    LogNormal,
    Marginal,
    Normal,
    Prior,
    Uniform,
)

__all__ = ["from_recipe", "read_recipe", "to_recipe", "write_recipe"]

#: The recipe format's own version, which moves only when the *layout* changes.
#: Separate from the package version on purpose: a recipe written by 1.0.0 and
#: read by 0.11.0 is fine unless this number moved.
RECIPE_FORMAT = 1

#: The marginals a recipe can carry, and the fields each is rebuilt from. A
#: marginal missing from here raises rather than being approximated by a
#: neighbour -- a prior silently replaced is a different model reported under
#: the same name.
_MARGINALS: dict[str, tuple[str, ...]] = {
    "LogNormal": ("mean_log", "sd_log"),
    "Normal": ("mean", "sd"),
    "Gamma": ("shape", "rate"),
    "Uniform": ("low", "high"),
}

#: Typed as returning a `Marginal` so a decoded prior is one: the values are the
#: four concrete classes, and `dict[str, type]` would widen them to `object`.
_MARGINAL_TYPES: dict[str, Callable[..., Marginal]] = {
    "LogNormal": LogNormal,
    "Normal": Normal,
    "Gamma": Gamma,
    "Uniform": Uniform,
}


def _package_version() -> str:
    """Read the package's version without importing it at module scope.

    `hawkes_package/__init__.py` imports this subpackage, so a top-level
    `from .. import __version__` here is a cycle -- and one that only bites when
    the package is imported fresh, which is every time except the one where it
    was tested.
    """
    from .. import __version__

    return str(__version__)


def _encode_prior(prior: Prior) -> dict[str, Any]:
    """Describe `prior` in JSON, refusing one that cannot be rebuilt exactly."""
    if isinstance(prior, ConstrainedPrior):
        return {
            "kind": "constrained",
            "base": _encode_prior(prior.base),
            "max_draws": int(prior.max_draws),
            # The support is `model.support`, rebuilt from the model on load
            # rather than written down: it is a bound method of an object made
            # of closures, and writing its address would be writing nothing.
            "support": "model",
        }
    if isinstance(prior, IndependentPrior):
        marginals = []
        for marginal in prior.marginals:
            name = type(marginal).__name__
            if name not in _MARGINALS:
                raise TypeError(
                    f"a recipe cannot carry a {name} marginal: it is not one of "
                    f"{sorted(_MARGINALS)}. Adding one means adding its fields here, "
                    "so that a reload is the same prior rather than a near one."
                )
            marginals.append(
                {"kind": name, **{f: float(getattr(marginal, f)) for f in _MARGINALS[name]}}
            )
        return {"kind": "independent", "marginals": marginals}
    raise TypeError(
        f"a recipe cannot carry a {type(prior).__name__}: only IndependentPrior, "
        "optionally wrapped in ConstrainedPrior. A prior that cannot be written "
        "down exactly is one a reader would have to guess at."
    )


def _decode_prior(spec: dict[str, Any], support: Any) -> Prior:
    """Rebuild a prior from :func:`_encode_prior`'s output."""
    kind = spec["kind"]
    if kind == "constrained":
        return ConstrainedPrior(
            _decode_prior(spec["base"], support),
            support,
            max_draws=int(spec.get("max_draws", 100_000)),
        )
    if kind == "independent":
        marginals: list[Marginal] = []
        for entry in spec["marginals"]:
            name = entry["kind"]
            if name not in _MARGINAL_TYPES:
                raise ValueError(
                    f"this release does not know a {name!r} marginal; the recipe was "
                    "written by one that did"
                )
            fields = [float(entry[f]) for f in _MARGINALS[name]]
            marginals.append(_MARGINAL_TYPES[name](*fields))
        return IndependentPrior(tuple(marginals))
    raise ValueError(f"unknown prior kind {kind!r} in this recipe")


def to_recipe(estimator: HawkesEstimator, *, data: str | None = None) -> dict[str, Any]:
    """Describe `estimator` as JSON: what it is, not what it found.

    Parameters
    ----------
    estimator : HawkesEstimator
        Fitted or not -- a recipe is a configuration, and rerunning it is what
        reproduces the fit.
    data : str, optional
        Where the events are: a path, a URL, a DOI. Not the events themselves,
        which are the user's and would make the artefact too large to paste into
        an appendix.

    Returns
    -------
    dict
        JSON-serialisable.

    Raises
    ------
    TypeError
        If the estimator holds a `ProcessModel` object rather than a family
        name, or a prior this module cannot write down exactly. Both refusals
        are the point: a recipe that silently dropped either would reload into a
        different fit under the same name.

    Examples
    --------
    >>> from hawkes_package.inference import (
    ...     ConstrainedPrior,
    ...     HawkesEstimator,
    ...     IndependentPrior,
    ...     LogNormal,
    ...     exponential_model,
    ...     to_recipe,
    ... )
    >>> model = exponential_model()
    >>> prior = ConstrainedPrior(IndependentPrior((LogNormal(0.0, 1.0),) * 3), model.support)
    >>> recipe = to_recipe(HawkesEstimator("exponential", prior, rng=7))
    >>> recipe["estimator"]["rng"]
    7

    .. versionadded:: 1.0.0
    """
    params = estimator.get_params()
    model = params["model"]
    if not isinstance(model, str):
        raise TypeError(
            "a recipe can only name a model by string. This estimator holds a "
            f"{type(model).__name__}, which is three closures -- a builder, a branching "
            "callable and a support -- and closures cannot be written down. Build the "
            "estimator with a family name, or store the cloud instead of the recipe."
        )
    if params["likelihood"] is not None:
        raise TypeError(
            "a recipe cannot carry an explicit likelihood object; leave it as None "
            "and the estimator picks the one exact for the model, which is what a "
            "reader can reproduce."
        )

    seed = params["rng"]
    if not isinstance(seed, int | type(None)):
        raise TypeError(
            f"a recipe needs an integer seed or None, not a {type(seed).__name__}. A "
            "Generator carries state nobody can write down, so a recipe holding one "
            "would reproduce a different run every time it was read."
        )

    return {
        "format": RECIPE_FORMAT,
        "hawkes_package": _package_version(),
        "data": data,
        "estimator": {
            "model": model,
            "rng": seed,
            "n_particles": int(params["n_particles"]),
            "blocks": params["blocks"],
            "ess_threshold": float(params["ess_threshold"]),
            "n_move": int(params["n_move"]),
            "scale": float(params["scale"]),
            "jitter": float(params["jitter"]),
            "on_invalid": params["on_invalid"],
        },
        "prior": _encode_prior(params["prior"]),
    }


def from_recipe(recipe: dict[str, Any]) -> HawkesEstimator:
    """Rebuild the estimator a recipe describes.

    The fit is *not* rebuilt: run it, on the data the recipe points at, and the
    result is the original -- exactly on the temporal path, where seeding is
    exact, and distributionally on the spatio-temporal one, which branches on
    floating-point comparisons whose last bits move with the SciPy build.

    Warns
    -----
    UserWarning
        If the recipe was written by a newer release of the package. This
        release cannot know what changed between them, and a silently different
        answer is the failure a version stamp exists to prevent.

    Raises
    ------
    ValueError
        If the recipe's *format* version is not one this release reads.

    .. versionadded:: 1.0.0
    """
    running = _package_version()
    stamp = str(recipe.get("hawkes_package", "unknown"))
    fmt = int(recipe.get("format", 0))
    if fmt != RECIPE_FORMAT:
        raise ValueError(
            f"this recipe is in format {fmt} and this release reads format "
            f"{RECIPE_FORMAT}. It was written by hawkes_package {stamp}."
        )
    if _is_newer(stamp, running):
        warnings.warn(
            f"this recipe was written by hawkes_package {stamp} and is being read by "
            f"{running}. Rerunning it may not reproduce the original fit: this "
            "release cannot know what the newer one changed. Install "
            f"hawkes_package=={stamp} to reproduce it exactly.",
            UserWarning,
            stacklevel=2,
        )

    settings = dict(recipe["estimator"])
    model_name = settings.pop("model")
    # The support is rebuilt from the model rather than read from the file,
    # because it is a bound method of an object made of closures.
    from .estimator import _TEMPORAL_FACTORIES

    model = _TEMPORAL_FACTORIES[model_name]()
    prior = _decode_prior(recipe["prior"], model.support)
    return HawkesEstimator(model_name, prior, **settings)


def _is_newer(candidate: str, current: str) -> bool:
    """Whether `candidate` is a later release than `current`.

    Compared on the numeric release segments alone. A pre-release suffix is
    ignored rather than ordered: getting `1.0.0rc1` against `1.0.0` subtly wrong
    would produce a warning nobody could act on, and the question this answers
    is only "was this written by something I have not seen".
    """

    def parts(version: str) -> tuple[int, ...]:
        head = version.split("+")[0].split("rc")[0].split("a")[0].split("b")[0]
        out = []
        for piece in head.split("."):
            digits = "".join(ch for ch in piece if ch.isdigit())
            out.append(int(digits) if digits else 0)
        return tuple(out)

    try:
        return parts(candidate) > parts(current)
    except ValueError:  # pragma: no cover - unparseable stamps are not newer
        return False


def write_recipe(estimator: HawkesEstimator, path: str | Path, *, data: str | None = None) -> None:
    """Write :func:`to_recipe`'s output to `path` as JSON.

    .. versionadded:: 1.0.0
    """
    Path(path).write_text(
        json.dumps(to_recipe(estimator, data=data), indent=2) + "\n", encoding="utf-8"
    )


def read_recipe(path: str | Path) -> HawkesEstimator:
    """Read a recipe from `path` and rebuild its estimator.

    .. versionadded:: 1.0.0
    """
    return from_recipe(json.loads(Path(path).read_text(encoding="utf-8")))
