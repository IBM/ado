# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import inspect
from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest

import ado.modules.operators.randomwalk  # noqa: F401 — loads operator plugins
from ado.core.datacontainer.resource import DataContainer, DataContainerResource
from ado.core.operation.config import (
    FunctionOperationInfo,
    GenericOperatorParameters,
    OperatorModuleConf,
)
from ado.core.operation.inputs import resolve_operation_project_context
from ado.core.operation.operation import OperationOutput
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore
from ado.modules.operators._general_orchestration import _operator_callable_for_harness
from ado.modules.operators.collections import (
    characterize,
    explore,
    summarize,
    summarize_operation,
)
from ado.modules.operators.provenance import operation_provenance
from ado.utilities.location import SQLiteStoreConfiguration


def test_resolve_project_context_from_input_when_operation_info_omits_it(
    make_stub_discovery_space: Callable,
) -> None:
    """Inputs supply project context when operation info does not set one."""
    context = ProjectContext()
    resolved = resolve_operation_project_context(
        None, {"measuredSpace": make_stub_discovery_space(context)}
    )
    assert resolved == context


def test_resolve_project_context_rejects_disagreeing_inputs(
    make_stub_discovery_space: Callable,
) -> None:
    """Inputs that carry different project contexts are rejected."""
    first = ProjectContext()
    second = ProjectContext(
        project="other",
        metadataStore=SQLiteStoreConfiguration(path="other.db"),
    )
    with pytest.raises(ValueError, match="disagree on project context"):
        resolve_operation_project_context(
            None,
            {
                "a": make_stub_discovery_space(first),
                "b": make_stub_discovery_space(second),
            },
        )


def test_resolve_project_context_requires_context_without_carrying_inputs() -> None:
    """Data-container-only calls must set projectContext explicitly."""
    with pytest.raises(ValueError, match="projectContext is required"):
        resolve_operation_project_context(None, {})


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


def test_summarize_operator_receives_list_inputs_and_relates_every_member(
    sql_store: SQLStore,
    create_resources: Callable,
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """A summarize operator run with a list input stores all list members as related."""
    received: dict[str, object] = {}

    def summarize_containers(
        containers: list[DataContainerResource],
        baseline: DataContainerResource,
        operationInfo: FunctionOperationInfo | None = None,
        *,
        parameters: GenericOperatorParameters,
    ) -> OperationOutput:
        received["containers"] = containers
        received["baseline"] = baseline
        return OperationOutput()

    listed = [
        DataContainerResource(config=DataContainer(data={"key": {"value": value}}))
        for value in ("one", "two")
    ]
    baseline = DataContainerResource(
        config=DataContainer(data={"key": {"value": "baseline"}})
    )
    create_resources([*listed, baseline])

    with temporary_operator(
        summarize.operators, summarize_operation, summarize_containers
    ) as name:
        function = summarize.operators[name].function
        assert function is not None
        function(
            listed,
            baseline,
            FunctionOperationInfo(projectContext=sql_store.project_context),
            parameters=GenericOperatorParameters(),
        )

    assert received["containers"] == listed
    assert received["baseline"] == baseline

    operation_identifiers = sql_store.getResourceIdentifiersOfKind("operation")[
        "IDENTIFIER"
    ]
    assert len(operation_identifiers) == 1
    related = sql_store.getRelatedSubjectResourceIdentifiers(operation_identifiers[0])
    assert set(related["IDENTIFIER"]) == {
        resource.identifier for resource in [*listed, baseline]
    }
