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

    from ado.core.operation.config import FunctionOperationInfo
    from ado.metastore.project import ProjectContext
    from ado.metastore.sqlstore import SQLStore

# ---------------------------------------------------------------------------
# Project-context helpers
# ---------------------------------------------------------------------------

# Python types an operator function may receive for a resolved resource input.
OperatorInputType = DiscoverySpace | DataContainerResource

# Maps a declared resource kind to the rich type an operator receives for it.
OPERATOR_INPUT_TYPE_FOR_KIND: dict[CoreResourceKinds, type] = {
    CoreResourceKinds.DISCOVERYSPACE: DiscoverySpace,
    CoreResourceKinds.DATACONTAINER: DataContainerResource,
}


def project_contexts_from_inputs(
    inputs: dict[str, OperatorInputType],
) -> list[tuple[str, ProjectContext]]:
    """Return ``(input_name, project_context)`` for context-carrying inputs.

    Only ``DiscoverySpace`` inputs carry a project context; ``DataContainerResource``
    inputs do not.

    Args:
        inputs: Mapping of parameter name to rich operator input.

    Returns:
        ``(name, ProjectContext)`` pairs in input iteration order. Empty when
        no input carries a project context.
    """
    result: list[tuple[str, ProjectContext]] = []
    for name, resource in inputs.items():
        if isinstance(resource, DiscoverySpace):
            result.append((name, resource.project_context))
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


def resolve_operation_project_context(
    operation_info: FunctionOperationInfo | None,
    inputs: dict[str, OperatorInputType],
) -> ProjectContext:
    """Resolve the project context for an operation invocation.

    Prefers ``FunctionOperationInfo.projectContext`` when set. Otherwise
    extracts project context from operator inputs that carry one. When no
    input carries context, callers must set ``projectContext`` explicitly.

    Args:
        operation_info: Optional operation info that may carry a project context.
        inputs: Mapping of parameter name to rich operator input.

    Returns:
        The resolved project context.

    Raises:
        ValueError: If ``projectContext`` is unset and no input carries a
            project context, or if context-carrying inputs disagree on the
            project.
    """
    if operation_info is not None and operation_info.projectContext is not None:
        return operation_info.projectContext

    carried = project_contexts_from_inputs(inputs)
    if not carried:
        raise ValueError(
            "FunctionOperationInfo.projectContext is required when no operator "
            "input carries a project context (e.g. DataContainer-only inputs)."
        )

    reference = carried[0][1]
    if any(ctx != reference for _, ctx in carried):
        details = ", ".join(
            f"{name!r}→project={ctx.project!r}" for name, ctx in carried
        )
        raise ValueError(f"Operator inputs disagree on project context: {details}.")

    return reference


def verify_inputs_exist_in_metastore(
    inputs: dict[str, OperatorInputType],
    metastore: SQLStore,
) -> None:
    """Assert that every operator input resource belongs to *metastore*.

    Args:
        inputs: Mapping of parameter name to rich operator input.
        metastore: The store that must contain all inputs.

    Raises:
        ValueError: If an input does not belong to *metastore*, or has an
            unsupported type.
    """
    for name, resource in inputs.items():
        if isinstance(resource, DiscoverySpace):
            kind = CoreResourceKinds.DISCOVERYSPACE
            identifier = resource.uri
        elif isinstance(resource, DataContainerResource):
            kind = CoreResourceKinds.DATACONTAINER
            identifier = resource.identifier
        else:
            raise ValueError(
                f"Input {name!r} has unsupported type {type(resource)!r}; "
                "expected DiscoverySpace or DataContainerResource."
            )

        if not metastore.containsResourceWithIdentifier(identifier, kind=kind):
            raise ValueError(
                f"Input {name!r} ({kind.value} {identifier!r}) is not in the "
                f"active metastore (project {metastore.project_context.project!r})."
            )


def metastore_for_operation_inputs(
    operation_info: FunctionOperationInfo | None,
    inputs: dict[str, OperatorInputType],
) -> SQLStore:
    """Open the metastore for *inputs* and check that each input is stored there.

    Args:
        operation_info: Optional operation info that may carry a project context.
        inputs: Mapping of parameter name to rich operator input.

    Returns:
        A metastore for the resolved project context.

    Raises:
        ValueError: If the project context cannot be resolved or an input is
            missing from the metastore.
    """
    from ado.metastore.sqlstore import SQLStore

    project_context = resolve_operation_project_context(operation_info, inputs)
    metastore = SQLStore(project_context=project_context)
    verify_inputs_exist_in_metastore(inputs, metastore)
    return metastore


# ---------------------------------------------------------------------------
# Resource-reference resolution
# ---------------------------------------------------------------------------


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
