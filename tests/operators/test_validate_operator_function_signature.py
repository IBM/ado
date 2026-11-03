# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for validate_operator_function_signature."""

import pytest

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import FunctionOperationInfo, GenericOperatorParameters
from ado.core.operation.operation import OperationOutput
from ado.modules.operators.base import validate_operator_function_signature


class _Params(GenericOperatorParameters):
    value: int = 1


def valid_op(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: _Params,
) -> OperationOutput: ...


def valid_op_with_kwargs(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    **kwargs: object,
) -> OperationOutput: ...


def no_annotations(
    discoverySpace,  # noqa: ANN001
    operationInfo=None,  # noqa: ANN001
    *,
    parameters: _Params,
) -> OperationOutput: ...


def missing_operation_info(
    discoverySpace: DiscoverySpace,
    *,
    parameters: _Params,
) -> OperationOutput: ...


def extra_positional(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    extra: int = 0,
    *,
    parameters: _Params,
) -> OperationOutput: ...


def wrong_first_param_type(
    discoverySpace: int,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: _Params,
) -> OperationOutput: ...


def wrong_second_param_type(
    discoverySpace: DiscoverySpace,
    operationInfo: int = 0,
    *,
    parameters: _Params,
) -> OperationOutput: ...


def wrong_parameters_type(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: int,
) -> OperationOutput: ...


def wrong_return_type(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: _Params,
) -> int: ...


class TestValidSignatures:
    def test_valid_parameters_model(self) -> None:
        """A discoverySpace, operationInfo, and parameters model passes."""
        validate_operator_function_signature(valid_op)

    def test_kwargs_rejected(self) -> None:
        """**kwargs is rejected in favour of a parameters model."""
        with pytest.raises(ValueError, match="kwargs"):
            validate_operator_function_signature(valid_op_with_kwargs)


class TestHintIntrospectionFailure:
    def test_unresolvable_forward_reference_raises(self) -> None:
        """An unresolvable forward reference in annotations raises ValueError.

        A function whose annotation references a name that is not in scope
        causes typing.get_type_hints to raise NameError.  The function has
        valid structure but the hints are unresolvable, so validation must
        reject it rather than silently skipping the type checks.
        """

        def bad_hints(
            discoverySpace: "UnresolvableType",  # noqa: F821
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _Params,
        ) -> OperationOutput: ...

        with pytest.raises(ValueError, match="type hints are missing or unresolvable"):
            validate_operator_function_signature(bad_hints)

    def test_uninspectable_fn_raises(self) -> None:
        """A callable whose signature cannot be inspected must raise ValueError.

        Conformance cannot be confirmed when introspection fails, so silently
        passing would allow invalid callables through.
        """
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
            validate_operator_function_signature(fn)


class TestInvalidSignatures:
    def test_missing_param_annotations_rejected(self) -> None:
        """A function with unannotated parameters is rejected."""
        with pytest.raises(ValueError, match="type hints are missing"):
            validate_operator_function_signature(no_annotations)

    def test_missing_operation_info(self) -> None:
        """Function with only one positional parameter is rejected."""
        with pytest.raises(ValueError, match="operationInfo"):
            validate_operator_function_signature(missing_operation_info)

    def test_extra_positional_parameter(self) -> None:
        """Function with more positional parameters than the protocol is rejected."""
        with pytest.raises(ValueError, match="extra"):
            validate_operator_function_signature(extra_positional)

    def test_wrong_first_param_type(self) -> None:
        """Wrong type on the first positional parameter is rejected."""
        with pytest.raises(ValueError, match="discoverySpace"):
            validate_operator_function_signature(wrong_first_param_type)

    def test_wrong_second_param_type(self) -> None:
        """Wrong type on the second positional parameter is rejected."""
        with pytest.raises(ValueError, match="operationInfo"):
            validate_operator_function_signature(wrong_second_param_type)

    def test_wrong_parameters_type(self) -> None:
        """parameters must be a GenericOperatorParameters subclass."""
        with pytest.raises(ValueError, match="parameters"):
            validate_operator_function_signature(wrong_parameters_type)

    def test_wrong_return_type(self) -> None:
        """Wrong return type is rejected."""
        with pytest.raises(ValueError, match="return"):
            validate_operator_function_signature(wrong_return_type)
