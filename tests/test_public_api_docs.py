"""Contract checks for the documented, typed public API."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import cellpax


def _public_callables():
    """Yield exported functions and methods defined directly on exported classes."""
    for name in cellpax.__all__:
        if name == "__version__":
            continue
        obj = getattr(cellpax, name)
        if inspect.isfunction(obj):
            yield name, obj
            continue
        if not inspect.isclass(obj):
            continue
        for member_name, descriptor in obj.__dict__.items():
            if member_name.startswith("_"):
                continue
            if isinstance(descriptor, property):
                yield f"{name}.{member_name}", descriptor.fget
            elif isinstance(descriptor, (classmethod, staticmethod)):
                yield f"{name}.{member_name}", descriptor.__func__
            elif inspect.isfunction(descriptor):
                yield f"{name}.{member_name}", descriptor


def _documented_parameters(docstring: str) -> set[str]:
    """Extract names from a NumPy-style Parameters section."""
    lines = inspect.cleandoc(docstring).splitlines()
    try:
        index = lines.index("Parameters") + 2
    except ValueError:
        return set()
    names: set[str] = set()
    while index < len(lines):
        line = lines[index]
        if index + 1 < len(lines) and line and set(lines[index + 1]) == {"-"}:
            break
        if line and not line.startswith(" "):
            heading = line.split(" : ", 1)[0]
            if " : " in line or heading.startswith("*"):
                names.update(name.strip().lstrip("*") for name in heading.split(","))
        index += 1
    return names


def test_exported_api_has_docstrings_and_annotations():
    failures: list[str] = []
    for qualified_name, obj in _public_callables():
        docstring = inspect.getdoc(obj)
        if not docstring:
            failures.append(f"{qualified_name}: missing docstring")
            continue

        signature = inspect.signature(obj)
        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.name not in {"self", "cls"}
        ]
        for parameter in parameters:
            if parameter.annotation is inspect.Parameter.empty:
                failures.append(
                    f"{qualified_name}: {parameter.name!r} has no annotation"
                )
        if signature.return_annotation is inspect.Signature.empty:
            failures.append(f"{qualified_name}: return value has no annotation")

        if parameters:
            documented = _documented_parameters(docstring)
            missing = {parameter.name for parameter in parameters} - documented
            if missing:
                failures.append(
                    f"{qualified_name}: undocumented parameters {sorted(missing)}"
                )
            if "Returns\n-------" not in docstring:
                failures.append(f"{qualified_name}: missing Returns section")

    assert not failures, "\n".join(failures)


def test_exported_classes_document_their_interface():
    failures = []
    for name in cellpax.__all__:
        obj = getattr(cellpax, name)
        if not inspect.isclass(obj):
            continue
        docstring = inspect.getdoc(obj) or ""
        if not docstring:
            failures.append(f"{name}: missing class docstring")
        elif not any(
            section in docstring
            for section in ("Parameters\n----------", "Attributes\n----------")
        ):
            failures.append(f"{name}: missing Parameters or Attributes section")
    assert not failures, "\n".join(failures)


def test_reference_page_lists_each_export_once():
    root = Path(__file__).parents[1]
    reference = (root / "docs/reference/api.md").read_text()
    directives = re.findall(r"^:::\s+([\w.]+)$", reference, flags=re.MULTILINE)
    names = [directive.rsplit(".", 1)[-1] for directive in directives]
    expected = set(cellpax.__all__) - {"__version__"}

    assert set(names) == expected
    assert len(names) == len(set(names))
    assert "::: cellpax\n" not in reference
