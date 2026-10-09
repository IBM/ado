# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import inspect

import pytest

import ado.modules.operators.randomwalk  # noqa: F401 — loads operator plugins
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import OperatorModuleConf
from ado.core.operation.context import resolve_operation_project_context
from ado.metastore.project import ProjectContext
from ado.modules.operators._general_orchestration import _operator_callable_for_harness
from ado.modules.operators.collections import characterize, explore
from ado.modules.operators.provenance import operation_provenance
from ado.utilities.location import SQLiteStoreConfiguration


class _Space(DiscoverySpace):
    """Discovery space stand-in that only carries a project context."""

    def __init__(self, project_context: ProjectContext) -> None:
        """Skip parent initialization."""
        self._project_context = project_context


def test_resolve_project_context_from_input_when_operation_info_omits_it() -> None:
    """Inputs supply project context when operation info does not set one."""
    context = ProjectContext()
    resolved = resolve_operation_project_context(
        None, {"measuredSpace": _Space(context)}
    )
    assert resolved == context


def test_resolve_project_context_rejects_disagreeing_inputs() -> None:
    """Inputs that carry different project contexts are rejected."""
    first = ProjectContext()
    second = ProjectContext(
        project="other",
        metadataStore=SQLiteStoreConfiguration(path="other.db"),
    )
    with pytest.raises(ValueError, match="disagree on project context"):
        resolve_operation_project_context(
            None,
            {"a": _Space(first), "b": _Space(second)},
        )


def test_operation_provenance_uses_package_provenance() -> None:
    """Registered operators record their package under the operator identifier."""
    metadata = explore.operators["random_walk"]
    provenance = operation_provenance(metadata.reference)

    assert set(provenance.operators) == {metadata.operatorIdentifier}
    assert provenance.operators[metadata.operatorIdentifier].distributionName == (
        "ado-core"
    )
    assert provenance.experiments == []
    assert provenance.actuators == {}


def test_operation_provenance_empty_for_module_conf() -> None:
    """Non-reference operator modules record no operator provenance."""
    provenance = operation_provenance(
        OperatorModuleConf.model_construct(moduleClass="X", moduleName="x.y")
    )
    assert provenance.operators == {}
    assert provenance.experiments == []
    assert provenance.actuators == {}


@pytest.mark.parametrize(
    "operator_name",
    ["profile"],
)
def test_operator_callable_for_harness_unwraps_decorated_operator(
    operator_name: str,
) -> None:
    """Decorated operators register a wrapper; harness must call the implementation."""
    registered = characterize.operators[operator_name].function
    assert registered is not None
    resolved = _operator_callable_for_harness(registered)
    assert resolved is inspect.unwrap(registered)
