"""Optional, author-code baselines kept outside the UnifiedSTPP registry.

These adapters deliberately have a small protocol surface.  Campaign dispatch
must call them explicitly: they are not silently substituted for native models.
"""

from importlib import import_module

_EXPORT_MODULES = {
    "ETASAuthorAdapter": ".etas_author", "ETASContract": ".etas_author",
    "TriTPPAdapter": ".tritpp", "TriTPPContract": ".tritpp",
    "WSMAuthorAdapter": ".wsm_author", "WSMContract": ".wsm_author",
}


def __getattr__(name):
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORT_MODULES))

__all__ = ["ETASAuthorAdapter", "ETASContract", "TriTPPAdapter", "TriTPPContract", "WSMAuthorAdapter", "WSMContract"]
