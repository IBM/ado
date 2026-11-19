# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for validating list inputs of summarize operations."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager

import pytest

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
from ado.core.resources import ADOResourceReference, CoreResourceKinds
from ado.modules.operators.collections import summarize, summarize_operation


def _summarize_containers(
    containers: list[DataContainerResource],
    baseline: DataContainerResource,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: GenericOperatorParameters,
) -> OperationOutput: ...


@pytest.fixture
def summarize_operator_name(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> Iterator[str]:
    """Register a summarize operator with a list input and a scalar baseline."""
    with temporary_operator(
        summarize.operators, summarize_operation, _summarize_containers
    ) as name:
        yield name


def _operation_config(
    operator_name: str,
    inputs: dict,
) -> DiscoveryOperationResourceConfiguration:
    """Build an operation configuration that references *operator_name*."""
    return DiscoveryOperationResourceConfiguration(
        operation=DiscoveryOperationConfiguration(
            module=OperatorReference(
                operatorName=operator_name,
                operationType=DiscoveryOperationEnum.SUMMARIZE,
            ),
            parameters=GenericOperatorParameters(),
        ),
        inputs=inputs,
    )


def test_validate_inputs_accepts_list_and_scalar(
    summarize_operator_name: str,
) -> None:
    """A list input plus a scalar baseline is valid for a summarize operator."""
    configuration = _operation_config(
        summarize_operator_name,
        {
            "containers": [
                ADOResourceReference(identifier="datacontainer-aaa"),
                ADOResourceReference(identifier="datacontainer-bbb"),
            ],
            "baseline": ADOResourceReference(identifier="datacontainer-baseline"),
        },
    )
    assert isinstance(configuration.inputs["containers"], list)
    assert len(configuration.inputs["containers"]) == 2
    assert configuration.inputs["containers"][0].kind == CoreResourceKinds.DATACONTAINER
    assert configuration.inputs["baseline"].kind == CoreResourceKinds.DATACONTAINER


def test_validate_inputs_coerces_singleton_mapping_for_list_param(
    summarize_operator_name: str,
) -> None:
    """A single mapping for an isList parameter becomes a one-element list."""
    configuration = _operation_config(
        summarize_operator_name,
        {
            "containers": ADOResourceReference(identifier="datacontainer-aaa"),
            "baseline": ADOResourceReference(identifier="datacontainer-baseline"),
        },
    )
    assert isinstance(configuration.inputs["containers"], list)
    assert len(configuration.inputs["containers"]) == 1
    assert configuration.inputs["containers"][0].identifier == "datacontainer-aaa"


def test_validate_inputs_rejects_empty_list(
    summarize_operator_name: str,
) -> None:
    """An empty list is not a valid value for an isList parameter."""
    with pytest.raises(ValueError, match="empty"):
        _operation_config(
            summarize_operator_name,
            {
                "containers": [],
                "baseline": ADOResourceReference(identifier="datacontainer-baseline"),
            },
        )


def test_validate_inputs_rejects_wrong_kind_in_list_member(
    summarize_operator_name: str,
) -> None:
    """Every member of a list input must have the operator input's kind."""
    with pytest.raises(ValueError, match="declares kind 'discoveryspace'"):
        _operation_config(
            summarize_operator_name,
            {
                "containers": [
                    ADOResourceReference(identifier="datacontainer-aaa"),
                    ADOResourceReference(
                        identifier="space-bbb", kind=CoreResourceKinds.DISCOVERYSPACE
                    ),
                ],
                "baseline": ADOResourceReference(identifier="datacontainer-baseline"),
            },
        )


def test_validate_inputs_rejects_list_for_scalar_param(
    summarize_operator_name: str,
) -> None:
    """A YAML list is rejected for a scalar resource parameter."""
    with pytest.raises(ValueError, match="baseline"):
        _operation_config(
            summarize_operator_name,
            {
                "containers": [
                    ADOResourceReference(identifier="datacontainer-aaa"),
                ],
                "baseline": [
                    ADOResourceReference(identifier="datacontainer-baseline"),
                ],
            },
        )


def test_with_kind_writes_one_element_list(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """An unqualified --with binds to the list input and stores a one-element list."""
    from ado.cli.core.config import AdoConfiguration
    from ado.cli.models.parameters import AdoCreateCommandParameters
    from ado.cli.models.types import AdoCreateSupportedResourceTypes
    from ado.cli.resources.operation.create import apply_resource_flags

    with temporary_operator(
        summarize.operators, summarize_operation, _summarize_containers
    ) as name:
        raw = {
            "operation": {
                "module": {
                    "operatorName": name,
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
        apply_resource_flags(
            raw,
            AdoCreateCommandParameters(
                ado_configuration=AdoConfiguration(),
                dry_run=True,
                new_sample_store=False,
                override_values=[],
                resource_configuration_file=None,
                resource_type=AdoCreateSupportedResourceTypes.OPERATION,
                use_default_sample_store=False,
                use_latest=[],
                with_resources={CoreResourceKinds.DATACONTAINER: "datacontainer-aaa"},
            ),
        )
    assert raw["inputs"]["containers"] == [
        {"identifier": "datacontainer-aaa", "kind": "datacontainer"}
    ]
    assert raw["inputs"]["baseline"]["identifier"] == "datacontainer-baseline"


def test_with_named_list_input_replaces_file_value_then_appends(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """Repeated --with for a list input replace the file's list and then append."""
    from ado.cli.core.config import AdoConfiguration
    from ado.cli.models.parameters import (
        AdoCreateCommandParameters,
        OperationInputResourceOption,
    )
    from ado.cli.models.types import AdoCreateSupportedResourceTypes
    from ado.cli.resources.operation.create import apply_resource_flags

    with temporary_operator(
        summarize.operators, summarize_operation, _summarize_containers
    ) as name:
        raw = {
            "operation": {
                "module": {
                    "operatorName": name,
                    "operationType": DiscoveryOperationEnum.SUMMARIZE.value,
                }
            },
            "inputs": {
                "containers": [
                    {
                        "identifier": "datacontainer-from-file",
                        "kind": "datacontainer",
                    }
                ],
                "baseline": {
                    "identifier": "datacontainer-baseline",
                    "kind": "datacontainer",
                },
            },
        }
        apply_resource_flags(
            raw,
            AdoCreateCommandParameters(
                ado_configuration=AdoConfiguration(),
                dry_run=True,
                new_sample_store=False,
                override_values=[],
                resource_configuration_file=None,
                resource_type=AdoCreateSupportedResourceTypes.OPERATION,
                use_default_sample_store=False,
                use_latest=[],
                with_resources={},
                with_inputs=[
                    OperationInputResourceOption(
                        input_name="containers",
                        kind=CoreResourceKinds.DATACONTAINER,
                        value=identifier,
                    )
                    for identifier in ("datacontainer-aaa", "datacontainer-bbb")
                ],
            ),
        )
    assert raw["inputs"]["containers"] == [
        {"identifier": "datacontainer-aaa", "kind": "datacontainer"},
        {"identifier": "datacontainer-bbb", "kind": "datacontainer"},
    ]
