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
from ado.utilities.annotations import _unwrap_annotated

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

# Inverse of OPERATOR_INPUT_TYPE_FOR_KIND, used when reading a function signature.
KIND_FOR_OPERATOR_INPUT_TYPE: dict[type, CoreResourceKinds] = {
    python_type: kind for kind, python_type in OPERATOR_INPUT_TYPE_FOR_KIND.items()
}


def resource_inputs_from_operator_function(
    fn: typing.Callable,
) -> list[ADOResourcePropertyDescriptor]:
    """Deduce resource inputs from an operator function signature.

    Leading parameters, everything before ``operationInfo`` and ``parameters``,
    must be annotated with a type registered in
    ``KIND_FOR_OPERATOR_INPUT_TYPE``. Parameter names become input identifiers
    and annotations become resource kinds.

    Args:
        fn: Operator callable to inspect.

    Returns:
        Ordered resource-input descriptors.

    Raises:
        ValueError: If the signature cannot be inspected, lacks the trailing
            ``operationInfo`` and ``parameters`` parameters, has no resource
            inputs, uses ``*args`` or ``**kwargs``, or annotates a leading
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

    named = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
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

    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Cannot deduce resource inputs for operator function {fn!r}: "
            f"type hints are missing or unresolvable ({exc})."
        ) from exc

    descriptors: list[ADOResourcePropertyDescriptor] = []
    supported = sorted({t.__name__ for t in KIND_FOR_OPERATOR_INPUT_TYPE})
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
