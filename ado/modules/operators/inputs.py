# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Per-collection validation of operator resource-input declarations.

Resolution of references to rich objects lives in
``ado.core.operation.inputs``.
"""

from ado.core.operation.config import DiscoveryOperationEnum
from ado.core.resources import ADOResourcePropertyDescriptor, CoreResourceKinds

# Allowed input kinds per operation type.
_ALLOWED_KINDS: dict[DiscoveryOperationEnum, frozenset[CoreResourceKinds]] = {
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

# Minimum required resource-input count per operation type.
_MIN_REQUIRED: dict[DiscoveryOperationEnum, int] = {
    DiscoveryOperationEnum.EXPLORE: 1,
    DiscoveryOperationEnum.MODIFY: 1,
    DiscoveryOperationEnum.FUSE: 2,
    DiscoveryOperationEnum.EXPORT: 1,
    DiscoveryOperationEnum.CHARACTERIZE: 1,
    DiscoveryOperationEnum.COMPARE: 2,
    DiscoveryOperationEnum.SUMMARIZE: 1,
}

# Exact required count. Operation types absent from this map have no exact count.
_EXACT_COUNT: dict[DiscoveryOperationEnum, int] = {
    DiscoveryOperationEnum.EXPLORE: 1,
    DiscoveryOperationEnum.EXPORT: 1,
}

# Kinds that must appear at least once.
_REQUIRED_PRIMARY_KINDS: dict[DiscoveryOperationEnum, CoreResourceKinds] = {
    DiscoveryOperationEnum.EXPLORE: CoreResourceKinds.DISCOVERYSPACE,
    DiscoveryOperationEnum.EXPORT: CoreResourceKinds.DISCOVERYSPACE,
    DiscoveryOperationEnum.MODIFY: CoreResourceKinds.DISCOVERYSPACE,
    DiscoveryOperationEnum.FUSE: CoreResourceKinds.DISCOVERYSPACE,
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
    if operation_type != DiscoveryOperationEnum.SUMMARIZE:
        if list_inputs:
            raise ValueError(
                f"Operator of type {operation_type.value!r} does not allow "
                f"list resource inputs; found: {list_inputs}. List resource "
                "inputs are only supported for summarize operators."
            )
    elif not list_inputs:
        raise ValueError(
            "Operator of type 'summarize' requires at least one list resource input."
        )

    allowed = _ALLOWED_KINDS.get(operation_type)
    if allowed is not None:
        for descriptor in required_resource_inputs:
            if descriptor.kind not in allowed:
                raise ValueError(
                    f"Operator of type {operation_type.value!r} does not allow "
                    f"resource input kind {descriptor.kind.value!r}.  "
                    f"Allowed kinds: {[k.value for k in sorted(allowed, key=lambda k: k.value)]}."
                )

    min_req = _MIN_REQUIRED.get(operation_type, 1)
    if len(required_resource_inputs) < min_req:
        raise ValueError(
            f"Operator of type {operation_type.value!r} requires at least "
            f"{min_req} resource input(s); found {len(required_resource_inputs)}."
        )

    exact = _EXACT_COUNT.get(operation_type)
    if exact is not None and len(required_resource_inputs) != exact:
        raise ValueError(
            f"Operator of type {operation_type.value!r} requires exactly "
            f"{exact} resource input(s); found {len(required_resource_inputs)}."
        )

    primary_kind = _REQUIRED_PRIMARY_KINDS.get(operation_type)
    if primary_kind is not None and not any(
        descriptor.kind == primary_kind for descriptor in required_resource_inputs
    ):
        raise ValueError(
            f"Operator of type {operation_type.value!r} requires at least one "
            f"{primary_kind.value!r} resource input."
        )

    if operation_type == DiscoveryOperationEnum.FUSE:
        non_space = [
            descriptor
            for descriptor in required_resource_inputs
            if descriptor.kind != CoreResourceKinds.DISCOVERYSPACE
        ]
        if non_space:
            raise ValueError(
                "Fuse operators require all resource inputs to be of kind "
                f"'discoveryspace'; found: {[d.identifier for d in non_space]}."
            )

    identifiers = [descriptor.identifier for descriptor in required_resource_inputs]
    if len(identifiers) != len(set(identifiers)):
        duplicates = [i for i in identifiers if identifiers.count(i) > 1]
        raise ValueError(
            f"Duplicate resource input identifier(s): {list(set(duplicates))}."
        )
