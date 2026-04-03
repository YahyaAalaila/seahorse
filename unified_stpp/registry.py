"""Compatibility shim for the archived top-level model factory.

The primary live construction path is ``ConfigRegistry.build(...)`` in the
active model-config layer. This module preserves ``build_model(...)`` and
``PRESETS`` at their historical import path for tests and older utilities.
"""

from archive.registry import PRESETS, build_model

__all__ = ["build_model", "PRESETS"]
