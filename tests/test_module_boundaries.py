"""Holds the package to its import and iteration boundaries."""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "tehillim_cluster"
MODULES = sorted(SRC.glob("*.py"))
IDS = [p.name for p in MODULES]


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


@pytest.mark.parametrize("path", MODULES, ids=IDS)
def test_no_module_imports_another_modules_private_name(path: pathlib.Path) -> None:
    """A leading underscore means module-internal, so importing one couples across a boundary."""
    private = [
        f"{node.module}.{alias.name}"
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("tehillim_cluster")
        for alias in node.names
        if alias.name.startswith("_")
    ]

    assert private == []


@pytest.mark.parametrize("path", MODULES, ids=IDS)
def test_every_multi_argument_zip_is_strict(path: pathlib.Path) -> None:
    """Parallel per-psalm sequences must misalign loudly, never silently truncate."""
    loose = [
        node.lineno
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "zip"
        and len(node.args) > 1
        and not any(keyword.arg == "strict" for keyword in node.keywords)
    ]

    assert loose == []


@pytest.mark.parametrize("path", MODULES, ids=IDS)
def test_every_record_type_is_a_frozen_slotted_dataclass(path: pathlib.Path) -> None:
    """A result already written under a name must not be mutable after the fact."""
    mutable = []
    for node in _tree(path).body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields = [item for item in node.body if isinstance(item, ast.AnnAssign)]
        decorated = any(
            isinstance(d, ast.Call) and getattr(d.func, "id", "") == "dataclass"
            for d in node.decorator_list
        )
        is_protocol = any(
            isinstance(base, ast.Name) and base.id == "Protocol" for base in node.bases
        )
        if fields and not decorated and not is_protocol:
            mutable.append(node.name)

    assert mutable == []


def test_a_type_name_is_defined_once_across_the_package() -> None:
    """Two different types under one name is a collision waiting to be imported wrong."""
    defined: dict[str, list[str]] = {}
    for path in MODULES:
        for node in _tree(path).body:
            if isinstance(node, ast.ClassDef):
                defined.setdefault(node.name, []).append(path.name)

    assert {name: files for name, files in defined.items() if len(files) > 1} == {}


@pytest.mark.parametrize("path", MODULES, ids=IDS)
def test_a_feature_builder_is_named_after_the_tag_it_counts(path: pathlib.Path) -> None:
    """One naming rule, so a reader finds a profile's builder from the module name alone."""
    module = path.stem.removesuffix("_profile")
    mismatched = [
        node.name
        for node in _tree(path).body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("build_")
        and node.name.endswith("_feature_matrix")
        and node.name[len("build_") : -len("_feature_matrix")].endswith("_profile")
    ]

    assert mismatched == [], module
