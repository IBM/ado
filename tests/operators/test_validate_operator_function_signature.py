# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for validate_operator_function_signature."""

import pytest

from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import FunctionOperationInfo
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.operators.base import validate_operator_function_signature


def valid_op(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    **kwargs: object,
) -> OperationOutput: ...


def valid_op_no_kwargs(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
) -> OperationOutput: ...


def valid_op_no_annotations(
    discoverySpace,  # noqa: ANN001
    operationInfo=None,  # noqa: ANN001
) -> OperationOutput: ...


def missing_operation_info(
    discoverySpace: DiscoverySpace,
    **kwargs: object,
) -> OperationOutput: ...


def extra_positional(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    extra: int = 0,
) -> OperationOutput: ...


def wrong_first_param_type(
    discoverySpace: int,
    operationInfo: FunctionOperationInfo | None = None,
) -> OperationOutput: ...


def wrong_second_param_type(
    discoverySpace: DiscoverySpace,
    operationInfo: int = 0,
) -> OperationOutput: ...


def wrong_return_type(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
) -> int: ...


class TestValidSignatures:
    def test_valid_with_kwargs(self) -> None:
        """Full protocol-matching signature passes."""
        validate_operator_function_signature(valid_op)

    def test_valid_without_kwargs(self) -> None:
        """Omitting **kwargs is allowed."""
        validate_operator_function_signature(valid_op_no_kwargs)

    def test_valid_no_annotations(self) -> None:
        """Missing annotations are skipped without error."""
        validate_operator_function_signature(valid_op_no_annotations)


class TestInvalidSignatures:
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

    def test_wrong_return_type(self) -> None:
        """Wrong return type is rejected."""
        with pytest.raises(ValueError, match="return"):
            validate_operator_function_signature(wrong_return_type)
