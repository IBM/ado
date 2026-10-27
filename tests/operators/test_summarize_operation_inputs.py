# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for summarize operator list inputs in YAML, templates, and CLI."""

from collections.abc import Iterator

import pytest

from ado.cli.resources.operation.create import _add_operation_input_for_reference
from ado.cli.resources.operation.template import _operation_inputs_template
from ado.core.datacontainer.resource import DataContainerResource
from ado.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
    FunctionOperationInfo,
    GenericOperatorParameters,
    OperatorReference,
)
from ado.core.operation.operation import OperationOutput
from ado.core.resources import (
    ADOResourcePropertyDescriptor,
    ADOResourceReference,
    CoreResourceKinds,
)
from ado.modules.operators.collections import summarize, summarize_operation


class _Params(GenericOperatorParameters):
    value: int = 1


_OPERATOR_NAME = "_test_summarize_containers"


def _register_summarize_dummy() -> None:
    """Register a summarize operator with a list input and a scalar baseline."""

    @summarize_operation(
        name=_OPERATOR_NAME,
        version="0.0.1",
        configuration_model=_Params,
        example_configuration=_Params(),
        description="Test summarize operator",
    )
    def summarize_containers(
        containers: list[DataContainerResource],
        baseline: DataContainerResource,
        operationInfo: FunctionOperationInfo | None = None,
        *,
        parameters: _Params,
    ) -> OperationOutput: ...


@pytest.fixture
def summarize_dummy_operator() -> Iterator[None]:
    """Register and then remove the dummy summarize operator."""
    _register_summarize_dummy()
    try:
        yield
    finally:
        summarize.operators.pop(_OPERATOR_NAME, None)


def _operation_config(
    inputs: dict,
) -> DiscoveryOperationResourceConfiguration:
    """Build an operation configuration that references the dummy operator."""
    return DiscoveryOperationResourceConfiguration(
        operation=DiscoveryOperationConfiguration(
            module=OperatorReference(
                operatorName=_OPERATOR_NAME,
                operationType=DiscoveryOperationEnum.SUMMARIZE,
            ),
            parameters=_Params(),
        ),
        inputs=inputs,
    )


def test_validate_inputs_accepts_list_and_scalar(
    summarize_dummy_operator: None,
) -> None:
    """A list input plus a scalar baseline is valid for a summarize operator."""
    configuration = _operation_config(
        {
            "containers": [
                ADOResourceReference(identifier="datacontainer-aaa"),
                ADOResourceReference(identifier="datacontainer-bbb"),
            ],
            "baseline": ADOResourceReference(identifier="datacontainer-baseline"),
        }
    )
    assert isinstance(configuration.inputs["containers"], list)
    assert len(configuration.inputs["containers"]) == 2
    assert configuration.inputs["containers"][0].kind == CoreResourceKinds.DATACONTAINER
    assert configuration.inputs["baseline"].kind == CoreResourceKinds.DATACONTAINER


def test_validate_inputs_coerces_singleton_mapping_for_list_param(
    summarize_dummy_operator: None,
) -> None:
    """A single mapping for an isList parameter becomes a one-element list."""
    configuration = _operation_config(
        {
            "containers": ADOResourceReference(identifier="datacontainer-aaa"),
            "baseline": ADOResourceReference(identifier="datacontainer-baseline"),
        }
    )
    assert isinstance(configuration.inputs["containers"], list)
    assert len(configuration.inputs["containers"]) == 1
    assert configuration.inputs["containers"][0].identifier == "datacontainer-aaa"


def test_validate_inputs_rejects_empty_list(
    summarize_dummy_operator: None,
) -> None:
    """An empty list is not a valid value for an isList parameter."""
    with pytest.raises(ValueError, match="empty"):
        _operation_config(
            {
                "containers": [],
                "baseline": ADOResourceReference(identifier="datacontainer-baseline"),
            }
        )


def test_validate_inputs_rejects_list_for_scalar_param(
    summarize_dummy_operator: None,
) -> None:
    """A YAML list is rejected for a scalar resource parameter."""
    with pytest.raises(ValueError, match="baseline"):
        _operation_config(
            {
                "containers": [
                    ADOResourceReference(identifier="datacontainer-aaa"),
                ],
                "baseline": [
                    ADOResourceReference(identifier="datacontainer-baseline"),
                ],
            }
        )


def test_operation_inputs_template_emits_list_placeholder() -> None:
    """Templates emit a one-element list for isList resource inputs."""

    class _Operator:
        required_resource_inputs = (
            ADOResourcePropertyDescriptor(
                identifier="containers",
                kind=CoreResourceKinds.DATACONTAINER,
                isList=True,
            ),
            ADOResourcePropertyDescriptor(
                identifier="baseline",
                kind=CoreResourceKinds.DATACONTAINER,
            ),
        )

    template = _operation_inputs_template(_Operator())
    assert isinstance(template["containers"], list)
    assert len(template["containers"]) == 1
    assert template["containers"][0].kind == CoreResourceKinds.DATACONTAINER
    assert template["baseline"].kind == CoreResourceKinds.DATACONTAINER
    assert not isinstance(template["baseline"], list)


def test_add_operation_input_writes_one_element_list(
    summarize_dummy_operator: None,
) -> None:
    """--with for a list parameter stores a one-element list."""
    raw = {
        "operation": {
            "module": {
                "operatorName": _OPERATOR_NAME,
                "operationType": DiscoveryOperationEnum.SUMMARIZE.value,
            }
        },
        "inputs": {
            "baseline": {
                "identifier": "datacontainer-baseline",
                "kind": "datacontainer",
            }
        },
    }
    _add_operation_input_for_reference(
        raw,
        ADOResourceReference(
            identifier="datacontainer-aaa",
            kind=CoreResourceKinds.DATACONTAINER,
        ),
    )
    assert raw["inputs"]["containers"] == [
        {"identifier": "datacontainer-aaa", "kind": "datacontainer"}
    ]
    assert raw["inputs"]["baseline"]["identifier"] == "datacontainer-baseline"


def test_add_operation_input_appends_to_existing_list(
    summarize_dummy_operator: None,
) -> None:
    """A second --with for a list parameter appends to the existing list."""
    raw = {
        "operation": {
            "module": {
                "operatorName": _OPERATOR_NAME,
                "operationType": DiscoveryOperationEnum.SUMMARIZE.value,
            }
        },
        "inputs": {
            "containers": [
                {"identifier": "datacontainer-aaa", "kind": "datacontainer"}
            ],
            "baseline": {
                "identifier": "datacontainer-baseline",
                "kind": "datacontainer",
            },
        },
    }
    _add_operation_input_for_reference(
        raw,
        ADOResourceReference(
            identifier="datacontainer-bbb",
            kind=CoreResourceKinds.DATACONTAINER,
        ),
    )
    assert raw["inputs"]["containers"] == [
        {"identifier": "datacontainer-aaa", "kind": "datacontainer"},
        {"identifier": "datacontainer-bbb", "kind": "datacontainer"},
    ]
