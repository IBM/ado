# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for validating list inputs of summarize operations."""

import pytest

from ado.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
    GenericOperatorParameters,
    OperatorReference,
)
from ado.core.resources import ADOResourceReference, CoreResourceKinds


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
