# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Resolve the project context and metastore for an operation invocation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.inputs import project_contexts_from_inputs
from ado.core.resources import CoreResourceKinds
from ado.metastore.sqlstore import SQLStore

if TYPE_CHECKING:
    from ado.core.operation.config import FunctionOperationInfo
    from ado.core.operation.inputs import OperatorInputType
    from ado.metastore.project import ProjectContext


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
    project_context = resolve_operation_project_context(operation_info, inputs)
    metastore = SQLStore(project_context=project_context)
    verify_inputs_exist_in_metastore(inputs, metastore)
    return metastore
