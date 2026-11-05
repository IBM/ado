# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Resolve the project context for an operation invocation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ado.core.operation.config import FunctionOperationInfo
    from ado.core.operation.inputs import ResolvedOperatorInput
    from ado.metastore.project import ProjectContext


def resolve_operation_project_context(
    operation_info: FunctionOperationInfo | None,
    inputs: dict[str, ResolvedOperatorInput],
) -> ProjectContext:
    """Resolve the project context for an operation invocation.

    Prefers ``FunctionOperationInfo.projectContext`` when set. Otherwise,
    extracts project context from operator inputs that carry one. When no
    input carries context, callers must set ``projectContext`` explicitly.
    List inputs are flattened, so every discovery space in a list contributes
    a project context.

    Args:
        operation_info: Optional operation info that may carry a project context.
        inputs: Mapping of parameter name to a rich operator input or a list
            of rich operator inputs.

    Returns:
        The resolved project context.

    Raises:
        ValueError: If ``projectContext`` is unset and no input carries a
            project context, or if context-carrying inputs disagree on the
            project.
    """
    # Imported lazily: inputs.py re-exports this function at module import time.
    from ado.core.operation.inputs import project_contexts_from_inputs

    if operation_info is not None and operation_info.projectContext is not None:
        return operation_info.projectContext

    carried = project_contexts_from_inputs(inputs)
    if not carried:
        raise ValueError(
            "FunctionOperationInfo.projectContext is required when no operator "
            "input carries a project context (e.g. DataContainer-only inputs)."
        )

    # ProjectContext is an unfrozen pydantic model, so it cannot be placed in a set.
    reference = carried[0][1]
    if any(ctx != reference for _, ctx in carried):
        details = ", ".join(
            f"{name!r}→project={ctx.project!r}" for name, ctx in carried
        )
        raise ValueError(f"Operator inputs disagree on project context: {details}.")

    return reference
