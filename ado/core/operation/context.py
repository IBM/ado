# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Resolve the project context / metastore for an operation invocation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.resources import CoreResourceKinds
from ado.metastore.project import get_active_project_context

if TYPE_CHECKING:
    from ado.core.operation.config import FunctionOperationInfo
    from ado.core.operation.inputs import OperatorInputType
    from ado.metastore.project import ProjectContext
    from ado.metastore.sqlstore import SQLStore


def resolve_operation_project_context(
    operation_info: FunctionOperationInfo | None,
) -> ProjectContext:
    """Resolve the project context for an operation invocation.

    Prefers :attr:`~ado.core.operation.config.FunctionOperationInfo.projectContext`
    when set; otherwise uses the process active project context. Missing both is
    a programming error.

    Args:
        operation_info: Optional operation info that may carry a project context.

    Returns:
        The resolved :class:`~ado.metastore.project.ProjectContext`.

    Raises:
        RuntimeError: If neither ``operation_info.projectContext`` nor an active
            project context is available.
    """
    if operation_info is not None and operation_info.projectContext is not None:
        return operation_info.projectContext
    active = get_active_project_context()
    if active is None:
        raise RuntimeError(
            "No projectContext on FunctionOperationInfo and no active "
            "project context — programming error."
        )
    return active


def assert_inputs_in_metastore(
    inputs: dict[str, OperatorInputType],
    metastore: SQLStore,
) -> None:
    """Assert that every operator input resource belongs to *metastore*.

    Membership is checked via
    :meth:`~ado.metastore.base.ResourceStore.containsResourceWithIdentifier`
    — the resource id must exist in *metastore* for the expected kind.

    Args:
        inputs: Mapping of parameter name → rich operator input.
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
