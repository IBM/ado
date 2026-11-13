# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Convert resource references into the objects passed to operators.

The conversion is resource reference, then stored resource, then the rich
Python object an operator function receives.
"""

from __future__ import annotations

import inspect
import typing
from typing import TYPE_CHECKING

from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.resources import (
    ADOResourcePropertyDescriptor,
    ADOResourceReference,
    CoreResourceKinds,
)
from ado.utilities.annotations import unwrap_annotated
from ado.utilities.lists import as_list

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

# A stored operation input: one resource reference, or a list of them.
OperationResourceInput = ADOResourceReference | list[ADOResourceReference]

# The rich object, or list of rich objects, passed to an operator parameter.
ResolvedOperatorInput = OperatorInputType | list[OperatorInputType]

# Maps parameter type annotations to CoreResourceKinds.
KIND_FOR_OPERATOR_INPUT_TYPE: dict[type, CoreResourceKinds] = {
    DiscoverySpace: CoreResourceKinds.DISCOVERYSPACE,
    DataContainerResource: CoreResourceKinds.DATACONTAINER,
}


def inspect_operator_function(
    fn: Callable[..., object],
) -> tuple[list[inspect.Parameter], dict[str, typing.Any]]:
    """Inspect an operator function's named parameters and type hints.

    Rejects ``*args``, ``**kwargs``, and positional-only parameters.
    ``parameters`` must be the only keyword-only parameter.

    Args:
        fn: Operator callable to inspect.

    Returns:
        The named parameters, in declaration order, and the function's type
        hints including ``Annotated`` metadata.

    Raises:
        ValueError: If the signature cannot be inspected, uses ``*args``,
            ``**kwargs``, or positional-only parameters, does not declare
            ``parameters`` as its only keyword-only parameter, or has
            unresolvable type hints.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot inspect operator function {fn!r}: signature could not "
            f"be inspected ({exc})."
        ) from exc

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        raise ValueError(
            "Operator function must not declare *args; resource inputs must be "
            "explicit named parameters."
        )
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        raise ValueError(
            "Operator function must not declare **kwargs; operation parameters "
            "must be a single 'parameters' argument typed as the configuration model."
        )
    if any(p.kind == inspect.Parameter.POSITIONAL_ONLY for p in params):
        raise ValueError(
            "Operator function must not declare positional-only parameters; "
            "orchestration passes resource inputs, operationInfo, and parameters "
            "as keyword arguments."
        )

    named = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    keyword_only_names = [
        p.name for p in named if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    if keyword_only_names != ["parameters"]:
        raise ValueError(
            "Operator function parameter 'parameters' must be keyword-only, "
            "and it must be the only keyword-only parameter."
        )

    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception as exc:
        raise ValueError(
            f"Operator function type hints are unresolvable ({exc})."
        ) from exc

    return named, hints


def discovery_spaces_from_inputs(
    inputs: dict[str, ResolvedOperatorInput],
) -> list[DiscoverySpace]:
    """Collect discovery spaces from resolved operator inputs.

    List parameters are flattened. Other resource types are skipped.

    Args:
        inputs: Mapping of parameter name to a rich operator input or a list
            of rich operator inputs.

    Returns:
        Discovery spaces in input iteration order.
    """
    return [
        resource
        for value in inputs.values()
        for resource in as_list(value)
        if isinstance(resource, DiscoverySpace)
    ]


def _reference_for_operator_input(resource: OperatorInputType) -> ADOResourceReference:
    """Return the resource reference for a resolved operator input.

    Args:
        resource: A discovery space or data container.

    Returns:
        An ADOResourceReference holding the kind and identifier of *resource*.

    Raises:
        ValueError: If *resource* is not a supported operator input type.
    """
    if isinstance(resource, DiscoverySpace | DataContainerResource):
        return resource.reference
    raise ValueError(
        f"Resolved input has unsupported type {type(resource)!r}; "
        "expected DiscoverySpace or DataContainerResource."
    )


def resource_references_from_resolved_inputs(
    inputs: dict[str, ResolvedOperatorInput],
) -> dict[str, OperationResourceInput]:
    """Dump resolved operator inputs back to resource references.

    List parameters stay lists. Scalar parameters stay scalars.

    Args:
        inputs: Mapping of parameter name to a rich operator input or a list
            of rich operator inputs.

    Returns:
        Mapping of parameter name to the corresponding resource reference or
        list of references.

    Raises:
        ValueError: If an input has an unsupported type.
    """
    dumped: dict[str, OperationResourceInput] = {}
    for name, value in inputs.items():
        if isinstance(value, list):
            dumped[name] = [_reference_for_operator_input(item) for item in value]
        else:
            dumped[name] = _reference_for_operator_input(value)
    return dumped


def _resource_input_from_annotation(
    hint: object, param_name: str
) -> tuple[CoreResourceKinds, bool]:
    """Return the resource kind of a parameter annotation and whether it is a list.

    The annotation must be a type registered in
    ``KIND_FOR_OPERATOR_INPUT_TYPE``, or a list of exactly one such type.
    ``Annotated`` wrappers on the annotation and on the list item type are
    unwrapped.

    Args:
        hint: Resolved type annotation, possibly wrapped in ``Annotated``.
        param_name: Parameter name, used in error messages.

    Returns:
        The resource kind and whether the annotation is a list of that kind.

    Raises:
        ValueError: If the annotation is neither a registered type nor a list
            of exactly one registered type.
    """
    annotation = unwrap_annotated(hint)
    list_type_arguments = typing.get_args(annotation)
    is_list = typing.get_origin(annotation) is list and len(list_type_arguments) == 1
    resource_type = unwrap_annotated(list_type_arguments[0]) if is_list else annotation

    resource_kind = KIND_FOR_OPERATOR_INPUT_TYPE.get(resource_type)  # type: ignore[arg-type]
    if resource_kind is None:
        supported_type_names = sorted(
            input_type.__name__ for input_type in KIND_FOR_OPERATOR_INPUT_TYPE
        )
        raise ValueError(
            f"Operator function parameter {param_name!r} must be annotated "
            f"with one of {supported_type_names}, or a list of one of those "
            f"types, got {annotation!r}."
        )
    return resource_kind, is_list


def resource_inputs_from_operator_function(
    fn: Callable[..., object],
) -> list[ADOResourcePropertyDescriptor]:
    """Deduce resource inputs from an operator function signature.

    The signature is interpreted as follows:

    - Leading parameters (everything before ``operationInfo`` and
      ``parameters``) are resource inputs.
    - Each leading parameter must be annotated with a type registered in
      ``KIND_FOR_OPERATOR_INPUT_TYPE``, or with a list of one such type.
      ``Annotated`` wrappers are unwrapped and the underlying type must match
      a registered type exactly.
    - Parameter names become input identifiers and annotations become
      resource kinds. A list annotation sets ``isList`` on the descriptor.
    - ``parameters`` must be keyword-only.

    Args:
        fn: Operator callable to inspect.

    Returns:
        Ordered resource-input descriptors.

    Raises:
        ValueError: If any of the following hold:

            - the signature cannot be inspected
            - the type hints cannot be resolved
            - the function uses ``*args``, ``**kwargs``, or positional-only
              parameters
            - ``parameters`` is not the only keyword-only parameter
            - the last two parameters are not ``operationInfo`` and
              ``parameters``
            - there are no resource inputs
            - a leading parameter is unannotated
            - a leading parameter is annotated with a type not in
              ``KIND_FOR_OPERATOR_INPUT_TYPE``, or with a list of anything
              other than exactly one such type
    """
    named, hints = inspect_operator_function(fn)
    if len(named) < 2 or (
        named[-2].name != "operationInfo" or named[-1].name != "parameters"
    ):
        raise ValueError(
            "Operator function parameters must end with 'operationInfo' and "
            f"'parameters'; got {[p.name for p in named]!r}."
        )

    resource_params = named[:-2]
    if not resource_params:
        raise ValueError(
            "Operator function must declare at least one resource input "
            "parameter before 'operationInfo' and 'parameters'."
        )

    descriptors: list[ADOResourcePropertyDescriptor] = []
    for param in resource_params:
        if param.name not in hints:
            raise ValueError(
                f"Operator function parameter {param.name!r} is missing a "
                "resolvable type annotation."
            )
        kind, is_list = _resource_input_from_annotation(hints[param.name], param.name)
        descriptors.append(
            ADOResourcePropertyDescriptor(
                identifier=param.name, kind=kind, isList=is_list
            )
        )

    return descriptors


def project_contexts_from_inputs(
    inputs: dict[str, ResolvedOperatorInput],
) -> list[tuple[str, ProjectContext]]:
    """Return ``(input_name, project_context)`` for context-carrying inputs.

    Args:
        inputs: Mapping of parameter name to resolved operator input.

    Returns:
        ``(name, ProjectContext)`` pairs in input iteration order. Empty when
        no input carries a project context.
    """
    # At the moment only DiscoverySpace carries context.
    # This may change with addition of new operator input types
    return [
        (name, resource.project_context)
        for name, value in inputs.items()
        for resource in as_list(value)
        if isinstance(resource, DiscoverySpace)
    ]


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


def _resolve_one_reference(
    name: str,
    reference: ADOResourceReference,
    metastore: SQLStore,
) -> OperatorInputType:
    """Resolve a single named resource reference to a rich operator-input object.

    Args:
        name: Input name, used in error messages.
        reference: The reference to resolve. Its ``kind`` must be set.
        metastore: Metastore used to load the referenced resource.

    Returns:
        The rich operator-input object for *reference*.

    Raises:
        ValueError: If ``reference.kind`` is not set or has no resolver.
        ResourceDoesNotExistError: If the identifier does not exist in the
            project's metastore.
    """
    if reference.kind is None:
        raise ValueError(f"Input '{name}' has no kind set")

    resolver = _RESOLVERS.get(reference.kind)
    if resolver is None:
        raise ValueError(
            f"Unsupported input kind {reference.kind.value!r} for "
            f"input '{name}'.  Supported kinds: "
            f"{[k.value for k in _RESOLVERS]}."
        )

    return resolver(reference, metastore)


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
    inputs: dict[str, ResolvedOperatorInput],
    metastore: SQLStore,
) -> None:
    """Assert that every operator input resource belongs to *metastore*.

    List inputs are flattened and all identifiers are checked in a single
    query.

    Args:
        inputs: Mapping of parameter name to a rich operator input or a list
            of rich operator inputs.
        metastore: The store that must contain all inputs.

    Raises:
        ResourcesDoNotExistError: If an input does not belong to *metastore*.
        ValueError: If an input has an unsupported type.
    """
    identifiers = {
        _reference_for_operator_input(resource).identifier
        for value in inputs.values()
        for resource in as_list(value)
    }
    metastore.has_resources_with_identifiers(identifiers, raise_if_missing=True)


def metastore_for_operation_inputs(
    operation_info: FunctionOperationInfo | None,
    inputs: dict[str, ResolvedOperatorInput],
) -> tuple[SQLStore, FunctionOperationInfo]:
    """Connect to the project's metastore and verify *inputs* exist in it.

    The project context comes from ``operation_info`` if set, otherwise from
    *inputs*, and is set on the returned operation info.

    Args:
        operation_info: Optional operation info that may carry a project context.
        inputs: Mapping of parameter name to a rich operator input or a list
            of rich operator inputs.

    Returns:
        The metastore, and operation info with ``projectContext`` set.

    Raises:
        ValueError: If the project context cannot be resolved, or an input has
            an unsupported type or is not in the metastore.
    """
    from ado.core.operation.config import FunctionOperationInfo
    from ado.metastore.sqlstore import SQLStore

    project_context = resolve_operation_project_context(operation_info, inputs)
    resolved_info = operation_info or FunctionOperationInfo()
    if resolved_info.projectContext is None:
        resolved_info = resolved_info.model_copy(
            update={"projectContext": project_context}
        )
    metastore = SQLStore(project_context=project_context)
    verify_inputs_exist_in_metastore(inputs, metastore)
    return metastore, resolved_info


# ---------------------------------------------------------------------------
# Resource-reference resolution
# ---------------------------------------------------------------------------


def resource_references_to_rich_types(
    resource_references: dict[str, OperationResourceInput],
    metastore: SQLStore,
) -> dict[str, ResolvedOperatorInput]:
    """Resolve named resource references to rich operator-input objects.

    Args:
        resource_references: Named references. Each reference must have
            ``kind`` set. A value may be a single reference or a list of
            references.
        metastore: Metastore used to load the referenced resources.

    Returns:
        Mapping of name to a rich input object or a list of rich input objects.

    Raises:
        ValueError: If a reference has no ``kind`` set, or if an unsupported
            resource kind is requested.
        ResourceDoesNotExistError: If an identifier does not exist in the
            project's metastore.
    """
    return {
        name: (
            [_resolve_one_reference(name, reference, metastore) for reference in value]
            if isinstance(value, list)
            else _resolve_one_reference(name, value, metastore)
        )
        for name, value in resource_references.items()
    }
