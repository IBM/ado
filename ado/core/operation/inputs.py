# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Convert resource references into the objects passed to operators.

The conversion is resource reference, then stored resource, then the rich
Python object an operator function receives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.resources import ADOResourceReference, CoreResourceKinds

if TYPE_CHECKING:
    from collections.abc import Callable

    from ado.metastore.project import ProjectContext
    from ado.metastore.sqlstore import SQLStore

# Python types an operator function may receive for a resolved resource input.
OperatorInputType = DiscoverySpace | DataContainerResource

# Maps a declared resource kind to the rich type an operator receives for it.
OPERATOR_INPUT_TYPE_FOR_KIND: dict[CoreResourceKinds, type] = {
    CoreResourceKinds.DISCOVERYSPACE: DiscoverySpace,
    CoreResourceKinds.DATACONTAINER: DataContainerResource,
}


def _discovery_space_project_context(resource: OperatorInputType) -> ProjectContext:
    """Return the project context carried by a discovery space."""
    if not isinstance(resource, DiscoverySpace):
        raise TypeError(
            "Project context can only be read from a DiscoverySpace, "
            f"got {type(resource)!r}."
        )
    return resource.project_context


# Rich input types that embed a project context. Data containers do not.
OPERATOR_INPUT_PROJECT_CONTEXT_GETTERS: dict[
    type,
    Callable[[OperatorInputType], ProjectContext],
] = {
    DiscoverySpace: _discovery_space_project_context,
}


def project_contexts_from_inputs(
    inputs: dict[str, OperatorInputType],
) -> list[tuple[str, ProjectContext]]:
    """Return ``(input_name, project_context)`` for context-carrying inputs.

    Args:
        inputs: Mapping of parameter name to rich operator input.

    Returns:
        ``(name, ProjectContext)`` pairs in input iteration order. Empty when
        no input carries a project context.
    """
    result: list[tuple[str, ProjectContext]] = []
    for name, resource in inputs.items():
        for input_type, getter in OPERATOR_INPUT_PROJECT_CONTEXT_GETTERS.items():
            if isinstance(resource, input_type):
                result.append((name, getter(resource)))
                break
    return result


def _resolve_discoveryspace(
    reference: ADOResourceReference, metastore: SQLStore
) -> DiscoverySpace:
    """Load a discovery space referenced by *reference*."""
    return DiscoverySpace.from_stored_configuration(
        project_context=metastore.project_context,
        space_identifier=reference.identifier,
        metadata_store=metastore,
    )


def _resolve_datacontainer(
    reference: ADOResourceReference, metastore: SQLStore
) -> DataContainerResource:
    """Load a data container referenced by *reference*."""
    return metastore.getResource(  # type: ignore[return-value]
        identifier=reference.identifier,
        kind=CoreResourceKinds.DATACONTAINER,
        raise_error_if_no_resource=True,
    )


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

    Args:
        resource_references: Named references. Each reference must have
            ``kind`` set.
        metastore: Metastore used to load the referenced resources.

    Returns:
        Mapping of name to rich input object.

    Raises:
        ValueError: If a reference has no ``kind`` set, or if an unsupported
            resource kind is requested.
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
