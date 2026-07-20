# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Module covering conversion of resource references, to resource instances,
to the rich python classes that are passed to operators

    resource reference -> ado resource instance -> rich class instance
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.resources import ADOResourceReference, CoreResourceKinds

if TYPE_CHECKING:
    from collections.abc import Callable

    from ado.metastore.sqlstore import SQLStore

#: The Python types an operator function may receive for a resolved resource input.
OperatorInputType = DiscoverySpace | DataContainerResource

#: Maps a declared resource kind to a rich Python type an operator receives for it
OPERATOR_INPUT_TYPE_FOR_KIND: dict[CoreResourceKinds, type] = {
    CoreResourceKinds.DISCOVERYSPACE: DiscoverySpace,
    CoreResourceKinds.DATACONTAINER: DataContainerResource,
}


def _resolve_discoveryspace(
    reference: ADOResourceReference, metastore: SQLStore
) -> DiscoverySpace:
    return DiscoverySpace.from_stored_configuration(
        project_context=metastore.project_context,
        space_identifier=reference.identifier,
        metadata_store=metastore,
    )


def _resolve_datacontainer(
    reference: ADOResourceReference, metastore: SQLStore
) -> DataContainerResource:
    return metastore.getResource(  # type: ignore[return-value]
        identifier=reference.identifier,
        kind=CoreResourceKinds.DATACONTAINER,
        raise_error_if_no_resource=True,
    )


#: Kind -> resolver registry. Keys mirror :data:`OPERATOR_INPUT_TYPE_FOR_KIND`.
_RESOLVERS: dict[
    CoreResourceKinds,
    Callable[[ADOResourceReference, SQLStore], OperatorInputType],
] = {
    CoreResourceKinds.DISCOVERYSPACE: _resolve_discoveryspace,
    CoreResourceKinds.DATACONTAINER: _resolve_datacontainer,
}


def resource_references_to_rich_types(
    resource_references: dict[str, ADOResourceReference],
    metastore: SQLStore,
) -> dict[str, OperatorInputType]:
    """Resolve named resource references to rich operator-input objects.

    The resource for each reference is fetched from *metastore* and then
    converted into the appropriate rich type.

    Args:
        resource_references: A named set of :class:`~ado.core.resources.ADOResourceReference`
            objects.
            Each reference must have ``kind`` set (populated by the
            ``validate_inputs`` model validator).
        metastore: Metastore used to load all referenced resources.

    Returns:
        Mapping of name → rich input object.

    Raises:
        ValueError: If a reference has no ``kind`` set, or if an
            unsupported resource kind is requested.
        ResourceDoesNotExistError: If an identifier does not exist in the
            project's metastore.
    """
    resolved: dict[str, OperatorInputType] = {}

    for name, reference in resource_references.items():
        if reference.kind is None:
            raise ValueError(f"Input '{name}' has no kind set")

        resolver = _RESOLVERS.get(reference.kind)
        if resolver is None:
            raise ValueError(
                f"Unsupported input kind {reference.kind.value!r} for "
                f"input '{name}'.  Supported kinds: "
                f"{[k.value for k in _RESOLVERS]}."
            )

        resolved[name] = resolver(reference, metastore)

    return resolved
