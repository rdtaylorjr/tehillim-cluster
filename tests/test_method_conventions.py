"""Holds every named clustering method to one shape."""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "tehillim_cluster"


def _method_classes() -> list[tuple[str, ast.ClassDef]]:
    found = []
    for path in sorted(SRC.glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            has_compute = any(
                isinstance(item, ast.FunctionDef) and item.name == "compute" for item in node.body
            )
            is_protocol = any(
                isinstance(base, ast.Name) and base.id == "Protocol" for base in node.bases
            )
            if has_compute and not is_protocol:
                found.append((path.name, node))
    return found


METHODS = [node for _, node in _method_classes()]
IDS = [f"{module}:{node.name}" for module, node in _method_classes()]


@pytest.mark.parametrize("node", METHODS, ids=IDS)
def test_every_method_is_a_frozen_slotted_dataclass(node: ast.ClassDef) -> None:
    """Frozen, so a method's name cannot drift from the results already written under it."""
    decorators = [
        {k.arg: k.value.value for k in d.keywords}
        for d in node.decorator_list
        if isinstance(d, ast.Call) and getattr(d.func, "id", "") == "dataclass"
    ]

    assert decorators == [{"frozen": True, "slots": True}]


@pytest.mark.parametrize("node", METHODS, ids=IDS)
def test_every_method_leads_with_its_name_and_description(node: ast.ClassDef) -> None:
    """Both reach the output, so every method carries them and carries them the same way."""
    fields = [item.target.id for item in node.body if isinstance(item, ast.AnnAssign)]

    assert fields[:2] == ["name", "description"]


def test_a_frozen_method_really_refuses_to_be_renamed() -> None:
    """The written results are labelled by name, so a method must not be renamed after the fact."""
    from tehillim_cluster.clustering import SpectralClusteringMethod, data_driven_k
    from tehillim_cluster.k_selection import DEFAULT_K_VALUES

    method = SpectralClusteringMethod(
        name="a-spectral", description="", k_selector=data_driven_k(DEFAULT_K_VALUES)
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        method.name = "something_else"  # type: ignore[misc]


def test_every_method_class_was_discovered() -> None:
    #: SpectralClusteringMethod is the only concrete method class this package ships.
    assert len(METHODS) >= 1
