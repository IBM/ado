# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Module covering conversion of resource references, to resource instances,
to the rich python classes that are passed to operators

    resource reference -> ado resource instance -> rich class instance
"""

from __future__ import annotations

import inspect
import typing
from typing import TYPE_CHECKING, Annotated, get_args, get_origin

from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.resources import (
    ADOResourcePropertyDescriptor,
    ADOResourceReference,
    CoreResourceKinds,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ado.metastore.project import ProjectContext
    from ado.metastore.sqlstore import SQLStore

#: The Python types an operator function may receive for a resolved resource input.
OperatorInputType = DiscoverySpace | DataContainerResource

#: Maps a declared resource kind to a rich Python type an operator receives for it
OPERATOR_INPUT_TYPE_FOR_KIND: dict[CoreResourceKinds, type] = {
    CoreResourceKinds.DISCOVERYSPACE: DiscoverySpace,
    CoreResourceKinds.DATACONTAINER: DataContainerResource,
}

#: Inverse of :data:`OPERATOR_INPUT_TYPE_FOR_KIND` for signature deduction.
KIND_FOR_OPERATOR_INPUT_TYPE: dict[type, CoreResourceKinds] = {
    python_type: kind for kind, python_type in OPERATOR_INPUT_TYPE_FOR_KIND.items()
}

#: Rich input types that embed a :class:`~ado.metastore.project.ProjectContext`.
#:
#: Register a getter here when adding a new :data:`OperatorInputType` that
#: carries project context. Types absent from this map (e.g.
#: :class:`~ado.core.datacontainer.resource.DataContainerResource`) do not
#: provide a fallback context for operator wrappers.
OPERATOR_INPUT_PROJECT_CONTEXT_GETTERS: dict[
    type,
    Callable[[OperatorInputType], ProjectContext],
] = {
    DiscoverySpace: lambda resource: resource.project_context,  # type: ignore[attr-defined,return-value]
}


def _unwrap_annotated(hint: object) -> object:
    """Return the underlying type if *hint* is ``Annotated[T, ...]``, else *hint*."""
    if get_origin(hint) is Annotated:
        args = get_args(hint)
        return args[0] if args else hint
    return hint


def resource_inputs_from_operator_function(
    fn: typing.Callable,
) -> list[ADOResourcePropertyDescriptor]:
    """Deduce resource inputs from an operator function signature.

    Leading parameters (everything before ``operationInfo`` / ``parameters``)
    must be annotated with a type registered in
    :data:`KIND_FOR_OPERATOR_INPUT_TYPE`. Parameter names become input
    identifiers; annotations become resource kinds.

    Args:
        fn: Operator callable to inspect.

    Returns:
        Ordered list of :class:`~ado.core.resources.ADOResourcePropertyDescriptor`.

    Raises:
        ValueError: If the signature cannot be inspected, lacks the trailing
            ``operationInfo`` / ``parameters`` parameters, has no resource
            inputs, uses ``*args``/``**kwargs``, or annotates a leading
            parameter with an unsupported type.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot deduce resource inputs for operator function {fn!r}: "
            f"signature could not be inspected ({exc})."
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

    positional = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if len(positional) < 2:
        raise ValueError(
            "Operator function must end with parameters named 'operationInfo' "
            "and 'parameters'."
        )
    if positional[-2].name != "operationInfo" or positional[-1].name != "parameters":
        raise ValueError(
            "Operator function parameters must end with 'operationInfo' and "
            f"'parameters'; got {[p.name for p in positional]!r}."
        )

    resource_params = positional[:-2]
    if not resource_params:
        raise ValueError(
            "Operator function must declare at least one resource input "
            "parameter before 'operationInfo' and 'parameters'."
        )

    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Cannot deduce resource inputs for operator function {fn!r}: "
            f"type hints are missing or unresolvable ({exc})."
        ) from exc

    descriptors: list[ADOResourcePropertyDescriptor] = []
    supported = sorted(
        {t.__name__ for t in KIND_FOR_OPERATOR_INPUT_TYPE},
    )
    for param in resource_params:
        if param.name not in hints:
            raise ValueError(
                f"Operator function parameter {param.name!r} is missing a "
                "resolvable type annotation."
            )
        hint = _unwrap_annotated(hints[param.name])
        kind = KIND_FOR_OPERATOR_INPUT_TYPE.get(hint)  # type: ignore[arg-type]
        if kind is None:
            raise ValueError(
                f"Operator function parameter {param.name!r} must be annotated "
                f"with one of {supported}, got {hint!r}."
            )
        descriptors.append(
            ADOResourcePropertyDescriptor(identifier=param.name, kind=kind)
        )

    return descriptors


def project_contexts_from_inputs(
    inputs: dict[str, OperatorInputType],
) -> list[tuple[str, ProjectContext]]:
    """Return ``(input_name, project_context)`` for context-carrying inputs.

    Walks *inputs* and, for each value whose type is registered in
    :data:`OPERATOR_INPUT_PROJECT_CONTEXT_GETTERS`, extracts the embedded
    project context.

    Args:
        inputs: Mapping of parameter name → rich operator input.

    Returns:
        A list of ``(name, ProjectContext)`` pairs in input iteration order.
        Empty when no input carries a project context.
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
