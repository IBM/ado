# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for operator signature deduction and call-shape validation."""

from typing import Annotated

import pytest

from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    DiscoveryOperationEnum,
    FunctionOperationInfo,
    GenericOperatorParameters,
)
from ado.core.operation.inputs import resource_inputs_from_operator_function
from ado.core.operation.operation import OperationOutput
from ado.core.resources import CoreResourceKinds
from ado.modules.operators.base import (
    validate_operator_call_shape,
    validate_operator_registration,
)


def _validate(fn: object) -> None:
    """Deduce resource inputs and check the call shape."""
    inputs = resource_inputs_from_operator_function(fn)  # type: ignore[arg-type]
    validate_operator_call_shape(fn, inputs, configuration_model=_Params)  # type: ignore[arg-type]


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
        _validate(valid_op)

    def test_kwargs_rejected(self) -> None:
        """**kwargs is rejected in favour of a parameters model."""
        with pytest.raises(ValueError, match="kwargs"):
            _validate(valid_op_with_kwargs)


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
            _validate(bad_hints)

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
            _validate(fn)


class TestInvalidSignatures:
    def test_missing_param_annotations_rejected(self) -> None:
        """A function with unannotated parameters is rejected."""
        with pytest.raises(ValueError, match="missing"):
            _validate(no_annotations)

    def test_missing_operation_info(self) -> None:
        """Function with only one positional parameter is rejected."""
        with pytest.raises(ValueError, match="operationInfo"):
            _validate(missing_operation_info)

    def test_extra_positional_parameter(self) -> None:
        """Function with more positional parameters than the protocol is rejected."""
        with pytest.raises(ValueError, match="extra"):
            _validate(extra_positional)

    def test_wrong_first_param_type(self) -> None:
        """Wrong type on the first positional parameter is rejected."""
        with pytest.raises(ValueError, match="discoverySpace"):
            _validate(wrong_first_param_type)

    def test_wrong_second_param_type(self) -> None:
        """Wrong type on the second positional parameter is rejected."""
        with pytest.raises(ValueError, match="operationInfo"):
            _validate(wrong_second_param_type)

    def test_wrong_parameters_type(self) -> None:
        """parameters must be a GenericOperatorParameters subclass."""
        with pytest.raises(ValueError, match="parameters"):
            _validate(wrong_parameters_type)

    def test_compare_inputs_are_deduced_from_the_signature(self) -> None:
        """Leading resource parameters become the operator's required inputs."""

        def compare_spaces(
            left: DiscoverySpace,
            right: DataContainerResource,
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _Params,
        ) -> OperationOutput: ...

        descriptors = resource_inputs_from_operator_function(compare_spaces)
        assert [descriptor.identifier for descriptor in descriptors] == [
            "left",
            "right",
        ]
        assert descriptors[1].kind == CoreResourceKinds.DATACONTAINER
        validate_operator_registration(
            compare_spaces,
            descriptors,
            DiscoveryOperationEnum.COMPARE,
            configuration_model=_Params,
        )

    def test_fuse_requires_two_discovery_spaces(self) -> None:
        """A fuse operator with one discovery space is rejected."""
        descriptors = resource_inputs_from_operator_function(valid_op)
        with pytest.raises(ValueError, match="at least 2"):
            validate_operator_registration(
                valid_op,
                descriptors,
                DiscoveryOperationEnum.FUSE,
                configuration_model=_Params,
            )

    def test_wrong_return_type(self) -> None:
        """Wrong return type is rejected."""
        with pytest.raises(ValueError, match="return"):
            _validate(wrong_return_type)


class TestListResourceInputs:
    def test_list_discovery_space_is_deduced(self) -> None:
        """A list[DiscoverySpace] parameter becomes an isList discoveryspace input."""

        def summarize_spaces(
            spaces: list[DiscoverySpace],
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _Params,
        ) -> OperationOutput: ...

        descriptors = resource_inputs_from_operator_function(summarize_spaces)
        assert len(descriptors) == 1
        assert descriptors[0].identifier == "spaces"
        assert descriptors[0].kind == CoreResourceKinds.DISCOVERYSPACE
        assert descriptors[0].isList is True

    def test_list_data_container_and_scalar_baseline_are_deduced(self) -> None:
        """List and scalar resource parameters are recorded together."""

        def summarize_containers(
            containers: list[DataContainerResource],
            baseline: DataContainerResource,
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _Params,
        ) -> OperationOutput: ...

        descriptors = resource_inputs_from_operator_function(summarize_containers)
        assert [descriptor.identifier for descriptor in descriptors] == [
            "containers",
            "baseline",
        ]
        assert descriptors[0].kind == CoreResourceKinds.DATACONTAINER
        assert descriptors[0].isList is True
        assert descriptors[1].kind == CoreResourceKinds.DATACONTAINER
        assert descriptors[1].isList is False

    def test_annotated_list_discovery_space_is_deduced(self) -> None:
        """Annotated[list[DiscoverySpace], ...] unwraps to a list input."""

        def summarize_spaces(
            spaces: Annotated[list[DiscoverySpace], "batch"],
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _Params,
        ) -> OperationOutput: ...

        descriptors = resource_inputs_from_operator_function(summarize_spaces)
        assert descriptors[0].isList is True
        assert descriptors[0].kind == CoreResourceKinds.DISCOVERYSPACE

    def test_list_of_unsupported_type_is_rejected(self) -> None:
        """list[int] is not a valid resource input annotation."""

        def bad_list(
            values: list[int],
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _Params,
        ) -> OperationOutput: ...

        with pytest.raises(ValueError, match="values"):
            resource_inputs_from_operator_function(bad_list)

    def test_mixed_union_list_is_rejected(self) -> None:
        """A list whose element type is a union of resource types is rejected."""

        def mixed_list(
            resources: list[DiscoverySpace | DataContainerResource],
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _Params,
        ) -> OperationOutput: ...

        with pytest.raises(ValueError, match="resources"):
            resource_inputs_from_operator_function(mixed_list)

    def test_nested_list_is_rejected(self) -> None:
        """Nested list resource annotations are rejected."""

        def nested_list(
            spaces: list[list[DiscoverySpace]],
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _Params,
        ) -> OperationOutput: ...

        with pytest.raises(ValueError, match="spaces"):
            resource_inputs_from_operator_function(nested_list)

    def test_characterize_rejects_list_resource_inputs(self) -> None:
        """Non-summarize operators may not declare list resource parameters."""

        def list_op(
            spaces: list[DiscoverySpace],
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _Params,
        ) -> OperationOutput: ...

        descriptors = resource_inputs_from_operator_function(list_op)
        with pytest.raises(ValueError, match="list resource"):
            validate_operator_registration(
                list_op,
                descriptors,
                DiscoveryOperationEnum.CHARACTERIZE,
                configuration_model=_Params,
            )

    def test_fuse_rejects_list_resource_inputs(self) -> None:
        """Fuse operators keep named scalar parameters, not lists."""

        def list_op(
            spaces: list[DiscoverySpace],
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _Params,
        ) -> OperationOutput: ...

        descriptors = resource_inputs_from_operator_function(list_op)
        with pytest.raises(ValueError, match="list resource"):
            validate_operator_registration(
                list_op,
                descriptors,
                DiscoveryOperationEnum.FUSE,
                configuration_model=_Params,
            )

    def test_compare_rejects_list_resource_inputs(self) -> None:
        """Compare operators keep named scalar parameters, not lists."""

        def list_op(
            left: list[DiscoverySpace],
            right: DiscoverySpace,
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _Params,
        ) -> OperationOutput: ...

        descriptors = resource_inputs_from_operator_function(list_op)
        with pytest.raises(ValueError, match="list resource"):
            validate_operator_registration(
                list_op,
                descriptors,
                DiscoveryOperationEnum.COMPARE,
                configuration_model=_Params,
            )

    def test_summarize_accepts_list_and_scalar_baseline(self) -> None:
        """A summarize operator may take a list plus a scalar baseline."""

        def summarize_containers(
            containers: list[DataContainerResource],
            baseline: DataContainerResource,
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: _Params,
        ) -> OperationOutput: ...

        descriptors = resource_inputs_from_operator_function(summarize_containers)
        validate_operator_registration(
            summarize_containers,
            descriptors,
            DiscoveryOperationEnum.SUMMARIZE,
            configuration_model=_Params,
        )

    def test_summarize_rejects_scalar_only_inputs(self) -> None:
        """A summarize operator must declare at least one list resource input."""
        descriptors = resource_inputs_from_operator_function(valid_op)
        with pytest.raises(ValueError, match="at least one list"):
            validate_operator_registration(
                valid_op,
                descriptors,
                DiscoveryOperationEnum.SUMMARIZE,
                configuration_model=_Params,
            )
