# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for validate_operator_call_shape and resource_inputs_from_operator_function."""

import pytest

from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    ADOResourcePropertyDescriptor,
    FunctionOperationInfo,
    GenericOperatorParameters,
)
from ado.core.operation.inputs import resource_inputs_from_operator_function
from ado.core.operation.operation import OperationOutput
from ado.core.resources import CoreResourceKinds
from ado.modules.operators.base import validate_operator_call_shape


class _EmptyParams(GenericOperatorParameters):
    """Minimal configuration model for signature tests."""


class _CompareParams(GenericOperatorParameters):
    """Empty params for multi-input signature tests."""


def _space_input(identifier: str = "discoverySpace") -> ADOResourcePropertyDescriptor:
    return ADOResourcePropertyDescriptor(
        identifier=identifier, kind=CoreResourceKinds.DISCOVERYSPACE
    )


def _datacontainer_input(
    identifier: str = "baseline",
) -> ADOResourcePropertyDescriptor:
    return ADOResourcePropertyDescriptor(
        identifier=identifier, kind=CoreResourceKinds.DATACONTAINER
    )


def valid_op(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: _EmptyParams,
) -> OperationOutput: ...


def valid_multi_input(
    baseline: DataContainerResource,
    candidate: DataContainerResource,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: _CompareParams,
) -> OperationOutput: ...


def no_annotations(
    discoverySpace,  # noqa: ANN001
    operationInfo=None,  # noqa: ANN001
    *,
    parameters,  # noqa: ANN001
) -> OperationOutput: ...


def missing_operation_info(
    discoverySpace: DiscoverySpace,
    *,
    parameters: _EmptyParams,
) -> OperationOutput: ...


def extra_positional(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    extra: int = 0,
    *,
    parameters: _EmptyParams,
) -> OperationOutput: ...


def wrong_operation_info_type(
    discoverySpace: DiscoverySpace,
    operationInfo: int = 0,
    *,
    parameters: _EmptyParams,
) -> OperationOutput: ...


def wrong_return_type(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: _EmptyParams,
) -> int: ...


def with_kwargs(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    **kwargs: object,
) -> OperationOutput: ...


class TestValidCallShapes:
    def test_valid_single_resource(self) -> None:
        validate_operator_call_shape(
            valid_op, [_space_input()], configuration_model=_EmptyParams
        )

    def test_valid_multi_resource_inputs(self) -> None:
        validate_operator_call_shape(
            valid_multi_input,
            [_datacontainer_input("baseline"), _datacontainer_input("candidate")],
            configuration_model=_CompareParams,
        )


class TestHintIntrospectionFailure:
    def test_unresolvable_forward_reference_raises(self) -> None:
        def bad_hints(
            discoverySpace: "UnresolvableType",  # noqa: F821
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _EmptyParams,
        ) -> OperationOutput: ...

        with pytest.raises(ValueError, match="type hints are missing or unresolvable"):
            validate_operator_call_shape(
                bad_hints, [_space_input()], configuration_model=_EmptyParams
            )

    def test_uninspectable_fn_raises(self) -> None:
        import inspect
        import unittest.mock as mock

        fn = lambda: None  # noqa: E731
        original_signature = inspect.signature

        def patched_signature(obj: object, **kwargs: object) -> inspect.Signature:
            if obj is fn:
                raise TypeError("not inspectable")
            return original_signature(obj, **kwargs)  # type: ignore[arg-type]

        with (
            mock.patch("inspect.signature", side_effect=patched_signature),
            pytest.raises(ValueError, match="signature could not be inspected"),
        ):
            validate_operator_call_shape(
                fn, [_space_input()], configuration_model=_EmptyParams
            )


class TestInvalidCallShapes:
    def test_missing_param_annotations_rejected(self) -> None:
        with pytest.raises(ValueError, match="type hints are missing"):
            validate_operator_call_shape(
                no_annotations, [_space_input()], configuration_model=_EmptyParams
            )

    def test_missing_operation_info(self) -> None:
        with pytest.raises(ValueError, match="operationInfo"):
            validate_operator_call_shape(
                missing_operation_info,
                [_space_input()],
                configuration_model=_EmptyParams,
            )

    def test_extra_positional_parameter(self) -> None:
        with pytest.raises(ValueError, match="extra"):
            validate_operator_call_shape(
                extra_positional, [_space_input()], configuration_model=_EmptyParams
            )

    def test_kwargs_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\*\*kwargs"):
            validate_operator_call_shape(
                with_kwargs, [_space_input()], configuration_model=_EmptyParams
            )

    def test_wrong_resource_param_order(self) -> None:
        def wrong_order(
            candidate: DataContainerResource,
            baseline: DataContainerResource,
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _CompareParams,
        ) -> OperationOutput: ...

        with pytest.raises(ValueError, match="parameters must be the declared"):
            validate_operator_call_shape(
                wrong_order,
                [_datacontainer_input("baseline"), _datacontainer_input("candidate")],
                configuration_model=_CompareParams,
            )

    def test_wrong_operation_info_type(self) -> None:
        with pytest.raises(ValueError, match="operationInfo"):
            validate_operator_call_shape(
                wrong_operation_info_type,
                [_space_input()],
                configuration_model=_EmptyParams,
            )

    def test_wrong_return_type(self) -> None:
        with pytest.raises(ValueError, match="return"):
            validate_operator_call_shape(
                wrong_return_type, [_space_input()], configuration_model=_EmptyParams
            )

    def test_wrong_parameters_model(self) -> None:
        def wrong_params(
            discoverySpace: DiscoverySpace,
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _CompareParams,
        ) -> OperationOutput: ...

        with pytest.raises(ValueError, match="parameters"):
            validate_operator_call_shape(
                wrong_params, [_space_input()], configuration_model=_EmptyParams
            )


class TestResourceInputsFromOperatorFunction:
    def test_single_space(self) -> None:
        descriptors = resource_inputs_from_operator_function(valid_op)
        assert descriptors == [_space_input()]

    def test_multi_datacontainer(self) -> None:
        descriptors = resource_inputs_from_operator_function(valid_multi_input)
        assert descriptors == [
            _datacontainer_input("baseline"),
            _datacontainer_input("candidate"),
        ]

    def test_annotated_type_unwrapped(self) -> None:
        from typing import Annotated

        def fn(
            discoverySpace: Annotated[DiscoverySpace, "meta"],
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _EmptyParams,
        ) -> OperationOutput: ...

        assert resource_inputs_from_operator_function(fn) == [_space_input()]

    def test_unsupported_type_raises(self) -> None:
        def fn(
            x: object,
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _EmptyParams,
        ) -> OperationOutput: ...

        with pytest.raises(ValueError, match="must be annotated with one of"):
            resource_inputs_from_operator_function(fn)

    def test_no_resource_inputs_raises(self) -> None:
        def fn(
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _EmptyParams,
        ) -> OperationOutput: ...

        with pytest.raises(ValueError, match="at least one resource input"):
            resource_inputs_from_operator_function(fn)

    def test_kwargs_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\*\*kwargs"):
            resource_inputs_from_operator_function(with_kwargs)

    def test_missing_trailing_params_raises(self) -> None:
        with pytest.raises(ValueError, match="operationInfo"):
            resource_inputs_from_operator_function(missing_operation_info)
