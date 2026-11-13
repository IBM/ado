# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Per-collection validation of operator resource-input declarations.

Resolution of references to rich objects lives in
``ado.core.operation.inputs``.
"""

from ado.core.operation.config import DiscoveryOperationEnum
from ado.core.resources import ADOResourcePropertyDescriptor, CoreResourceKinds

# Allowed input kinds per operation type.
# study, learn, and query are absent: those collections have no decorator yet,
# so an operator of one of those types accepts any resource kind.
_ALLOWED_INPUT_KINDS: dict[DiscoveryOperationEnum, frozenset[CoreResourceKinds]] = {
    DiscoveryOperationEnum.EXPLORE: frozenset([CoreResourceKinds.DISCOVERYSPACE]),
    DiscoveryOperationEnum.MODIFY: frozenset([CoreResourceKinds.DISCOVERYSPACE]),
    DiscoveryOperationEnum.FUSE: frozenset([CoreResourceKinds.DISCOVERYSPACE]),
    DiscoveryOperationEnum.EXPORT: frozenset([CoreResourceKinds.DISCOVERYSPACE]),
    DiscoveryOperationEnum.CHARACTERIZE: frozenset(
        [
            CoreResourceKinds.DISCOVERYSPACE,
            CoreResourceKinds.DATACONTAINER,
        ]
    ),
    DiscoveryOperationEnum.COMPARE: frozenset(
        [
            CoreResourceKinds.DISCOVERYSPACE,
            CoreResourceKinds.DATACONTAINER,
        ]
    ),
    DiscoveryOperationEnum.SUMMARIZE: frozenset(
        [
            CoreResourceKinds.DISCOVERYSPACE,
            CoreResourceKinds.DATACONTAINER,
        ]
    ),
}

# Minimum count where the type needs more than the one input already required.
_MIN_RESOURCE_INPUT_COUNT: dict[DiscoveryOperationEnum, int] = {
    DiscoveryOperationEnum.FUSE: 2,
    DiscoveryOperationEnum.COMPARE: 2,
    DiscoveryOperationEnum.SUMMARIZE: 1,
}

# Exact required count. Operation types absent from this map have no exact count.
_EXACT_RESOURCE_INPUT_COUNT: dict[DiscoveryOperationEnum, int] = {
    DiscoveryOperationEnum.EXPLORE: 1,
    DiscoveryOperationEnum.EXPORT: 1,
}


def validate_resource_inputs_for_operation_type(
    required_resource_inputs: list[ADOResourcePropertyDescriptor],
    operation_type: DiscoveryOperationEnum,
) -> None:
    """Validate resource inputs against the rules for *operation_type*.

    Args:
        required_resource_inputs: The resource inputs declared by the operator.
        operation_type: The collection the operator is being registered into.

    Raises:
        ValueError: If *required_resource_inputs* is empty or violates a rule
            for *operation_type*.
    """
    if not required_resource_inputs:
        raise ValueError(
            f"Operator of type {operation_type.value!r} must declare at least "
            "one resource input."
        )

    list_inputs = [
        descriptor.identifier
        for descriptor in required_resource_inputs
        if descriptor.isList
    ]
    is_summarize = operation_type == DiscoveryOperationEnum.SUMMARIZE
    if list_inputs and not is_summarize:
        raise ValueError(
            f"Operator of type {operation_type.value!r} does not allow "
            f"list resource inputs; found: {list_inputs}. List resource "
            "inputs are only supported for summarize operators."
        )
    if is_summarize and not list_inputs:
        raise ValueError(
            "Operator of type 'summarize' requires at least one list resource input."
        )

    allowed_kinds = _ALLOWED_INPUT_KINDS.get(operation_type)
    if allowed_kinds is not None:
        allowed_kind_names = sorted(kind.value for kind in allowed_kinds)
        for descriptor in required_resource_inputs:
            if descriptor.kind not in allowed_kinds:
                raise ValueError(
                    f"Operator of type {operation_type.value!r} does not allow "
                    f"resource input kind {descriptor.kind.value!r}.  "
                    f"Allowed kinds: {allowed_kind_names}."
                )

    min_input_count = _MIN_RESOURCE_INPUT_COUNT.get(operation_type)
    if min_input_count is not None and len(required_resource_inputs) < min_input_count:
        raise ValueError(
            f"Operator of type {operation_type.value!r} requires at least "
            f"{min_input_count} resource input(s); found {len(required_resource_inputs)}."
        )

    exact_input_count = _EXACT_RESOURCE_INPUT_COUNT.get(operation_type)
    if (
        exact_input_count is not None
        and len(required_resource_inputs) != exact_input_count
    ):
        raise ValueError(
            f"Operator of type {operation_type.value!r} requires exactly "
            f"{exact_input_count} resource input(s); found {len(required_resource_inputs)}."
        )

    identifiers = [descriptor.identifier for descriptor in required_resource_inputs]
    if len(identifiers) != len(set(identifiers)):
        duplicates = [i for i in identifiers if identifiers.count(i) > 1]
        raise ValueError(
            f"Duplicate resource input identifier(s): {list(set(duplicates))}."
        )
