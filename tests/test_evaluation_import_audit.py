from __future__ import annotations

import ast
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCAN_ROOTS = ("seahorse", "tests")
_BANNED_MODULES = (
    "seahorse.evaluation.evaluation_helpers",
    "seahorse.evaluation.common",
    "seahorse.evaluation.io",
    "seahorse.evaluation.predictive_compare",
    "seahorse.evaluation.predictive_sampling",
    "seahorse.evaluation.predictive_benchmark",
    "seahorse.evaluation.surface_query",
    "seahorse.evaluation.surface_metrics",
    "seahorse.evaluation.surface_compute",
    "seahorse.evaluation.surface_profiles",
)


def _module_name_for_path(path: Path) -> str:
    return ".".join(path.relative_to(_REPO_ROOT).with_suffix("").parts)


def _resolve_import(path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    package_parts = list(path.relative_to(_REPO_ROOT).with_suffix("").parts[:-1])
    keep = len(package_parts) - (node.level - 1)
    if keep < 0:
        return None
    base_parts = package_parts[:keep]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


class TestEvaluationImportAudit(unittest.TestCase):
    def test_repo_code_does_not_import_deleted_evaluation_modules(self):
        offenders: list[str] = []

        for root_name in _SCAN_ROOTS:
            for path in (_REPO_ROOT / root_name).rglob("*.py"):
                if path.name == Path(__file__).name:
                    continue
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    module_names: list[str] = []
                    if isinstance(node, ast.Import):
                        module_names.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        resolved = _resolve_import(path, node)
                        if resolved is not None:
                            module_names.append(resolved)
                    for module_name in module_names:
                        if any(
                            module_name == banned or module_name.startswith(f"{banned}.")
                            for banned in _BANNED_MODULES
                        ):
                            offenders.append(f"{path}: {_module_name_for_path(path)} imports {module_name}")

        self.assertFalse(
            offenders,
            "Found imports of deleted evaluation modules:\n" + "\n".join(sorted(offenders)),
        )


if __name__ == "__main__":
    unittest.main()
