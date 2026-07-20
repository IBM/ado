# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Per-collection validation of operator resource-input declarations.

Enforces which resource kinds and how many inputs each
:class:`~ado.core.operation.config.DiscoveryOperationEnum` collection allows
in ``OperatorMetadata.requiredResourceInputs``.

Resolution of references to rich objects lives in
:mod:`ado.core.operation.inputs`.
"""

from ado.core.operation.config import (
    ADOResourcePropertyDescriptor,
    DiscoveryOperationEnum,
)
from ado.core.resources import CoreResourceKinds

#: Allowed input kinds per operation type.
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
}

#: Minimum required resource-input count per operation type. All declared
#: resource inputs are required (optional resource inputs are not supported).
_MIN_REQUIRED: dict[DiscoveryOperationEnum, int] = {
    DiscoveryOperationEnum.EXPLORE: 1,
    DiscoveryOperationEnum.MODIFY: 1,
    DiscoveryOperationEnum.FUSE: 2,  # must merge ≥ 2 spaces
    DiscoveryOperationEnum.EXPORT: 1,
    DiscoveryOperationEnum.CHARACTERIZE: 1,
    DiscoveryOperationEnum.COMPARE: 2,  # must compare ≥ 2 things
}

#: Exact required count (None = no exact constraint).
_EXACT_COUNT: dict[DiscoveryOperationEnum, int | None] = {
    DiscoveryOperationEnum.EXPLORE: 1,
    DiscoveryOperationEnum.EXPORT: 1,
}

#: Kinds that are required for the primary input.
_REQUIRED_PRIMARY_KINDS: dict[DiscoveryOperationEnum, CoreResourceKinds | None] = {
    DiscoveryOperationEnum.EXPLORE: CoreResourceKinds.DISCOVERYSPACE,
    DiscoveryOperationEnum.EXPORT: CoreResourceKinds.DISCOVERYSPACE,
    DiscoveryOperationEnum.MODIFY: CoreResourceKinds.DISCOVERYSPACE,
    DiscoveryOperationEnum.FUSE: CoreResourceKinds.DISCOVERYSPACE,
}


def validate_resource_inputs_for_operation_type(
    required_resource_inputs: list[ADOResourcePropertyDescriptor],
    operation_type: DiscoveryOperationEnum,
) -> None:
    """Validate *required_resource_inputs* against the per-operator type (collection) rules.

    Args:
        required_resource_inputs: The resource inputs declared by the operator.
        operation_type: The :class:`~ado.core.operation.config.DiscoveryOperationEnum`
            value for the collection the operator is being registered into.

    Raises:
        ValueError: If *required_resource_inputs* violate any rule for *operation_type*.
    """
    if not required_resource_inputs:
        return  # empty → uses legacy default; validated by decorators individually

    allowed = _ALLOWED_KINDS.get(operation_type)
    if allowed is not None:
        for d in required_resource_inputs:
            if d.kind not in allowed:
                raise ValueError(
                    f"Operator of type {operation_type.value!r} does not allow "
                    f"resource input kind {d.kind.value!r}.  "
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
        d.kind == primary_kind for d in required_resource_inputs
    ):
        raise ValueError(
            f"Operator of type {operation_type.value!r} requires at least one "
            f"{primary_kind.value!r} resource input."
        )

    # Fuse: all resource inputs must be discoveryspace
    if operation_type == DiscoveryOperationEnum.FUSE:
        non_space = [
            d
            for d in required_resource_inputs
            if d.kind != CoreResourceKinds.DISCOVERYSPACE
        ]
        if non_space:
            raise ValueError(
                "Fuse operators require all resource inputs to be of kind "
                f"'discoveryspace'; found: {[d.identifier for d in non_space]}."
            )

    # Identifiers must be unique
    identifiers = [d.identifier for d in required_resource_inputs]
    if len(identifiers) != len(set(identifiers)):
        duplicates = [i for i in identifiers if identifiers.count(i) > 1]
        raise ValueError(
            f"Duplicate resource input identifier(s): {list(set(duplicates))}."
        )
