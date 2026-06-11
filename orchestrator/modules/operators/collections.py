# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import functools
import logging
import typing
from typing import Annotated

import pydantic

import orchestrator.core.operation.config
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.metadata import PackageProvenance
from orchestrator.core.operation.config import (
    DiscoveryOperationEnum,
    FunctionOperationInfo,
    OperatorMetadata,
)
from orchestrator.modules.operators.base import (
    DiscoveryOperationBase,
    DiscoverySpaceSubscribingDiscoveryOperation,
    OperationOutput,
    OperatorFunction,
    validate_operator_function_signature,
)
from orchestrator.modules.operators.orchestrate import (
    orchestrate_explore_operation,
    orchestrate_general_operation,
)

moduleLog = logging.getLogger("operation_collections")


def _warn_if_operator_name_reused(
    collection_label: str, name: str, operators: dict[str, OperatorMetadata]
) -> None:
    """Log a warning when registering under a name that is already in use."""
    if name in operators:
        moduleLog.warning(
            "Operator %r is already registered in %s; replacing the existing entry",
            name,
            collection_label,
        )


class OperatorCollection(pydantic.BaseModel):
    """A registry of operators of a single discovery operation type.

    Operators are added via the decorator functions (e.g. ``characterize_operation``,
    ``explore_operation``).  Each registered name maps to an
    :class:`~orchestrator.core.operation.config.OperatorMetadata` instance that
    carries the function, version, description, configuration model,
    example configuration, and optional actor class.

    Attributes:
        type: The discovery operation type all operators in this collection belong to.
        operators: Mapping of operator name to :class:`~orchestrator.core.operation.config.OperatorMetadata` instance.
    """

    type: DiscoveryOperationEnum
    operators: Annotated[
        dict[str, OperatorMetadata], pydantic.Field(default_factory=dict)
    ]

    def list_operators(self) -> list[str]:
        """Returns all registered operator names."""
        return list(self.operators.keys())

    def __getattr__(self, item: str) -> OperatorFunction | None:
        """Returns the operator function for the given registered name.

        Args:
            item: Registered operator name.

        Returns:
            The callable registered under that name
            Or None if no callable registered.

        Raises:
            AttributeError: If no operator is registered with that name.
        """
        if item in self.operators:
            return self.operators[item].function
        raise AttributeError(f"Unknown attribute {item}")


characterize = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE
)
explore = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH
)
modify = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.MODIFY
)
export = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.EXPORT
)
compare = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.COMPARE
)
fuse = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.FUSE
)
study = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.STUDY
)
learn = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.LEARN
)
operationCollectionMap = {
    orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE: characterize,
    orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH: explore,
    orchestrator.core.operation.config.DiscoveryOperationEnum.MODIFY: modify,
    orchestrator.core.operation.config.DiscoveryOperationEnum.EXPORT: export,
    orchestrator.core.operation.config.DiscoveryOperationEnum.COMPARE: compare,
    orchestrator.core.operation.config.DiscoveryOperationEnum.STUDY: study,
    orchestrator.core.operation.config.DiscoveryOperationEnum.LEARN: learn,
    orchestrator.core.operation.config.DiscoveryOperationEnum.FUSE: fuse,
}

#
# Decorators for registering operator functions
#


def characterize_operation(
    name: str,
    version: str,
    configuration_model: type[pydantic.BaseModel],
    example_configuration: pydantic.BaseModel,
    description: str | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a characterize operation.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        @functools.wraps(func)
        def wrapper(
            discoverySpace: DiscoverySpace,
            operationInfo: FunctionOperationInfo | None = None,
            **kwargs: object,
        ) -> OperationOutput:
            return orchestrate_general_operation(
                operator_metadata=characterize.operators[name],
                operation_parameters=kwargs,
                discovery_space=discoverySpace,
                operation_info=operationInfo or FunctionOperationInfo(),
            )

        validate_operator_function_signature(wrapper)
        wrapper = typing.cast("OperatorFunction", wrapper)
        _warn_if_operator_name_reused("characterize", name, characterize.operators)
        characterize.operators[name] = OperatorMetadata(
            name=name,
            function=wrapper,
            version=version,
            description=description,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            type=DiscoveryOperationEnum.CHARACTERIZE,
            provenance=PackageProvenance.from_module_name(func.__module__),
        )
        return wrapper

    return _register


def _validate_explore_cls(t: type, metadata: OperatorMetadata) -> None:
    """Validate a class-decorated explore operator and its metadata.

    Args:
        t: The decorated class.
        metadata: The :class:`~orchestrator.core.operation.config.OperatorMetadata`
            returned by ``t.operator_metadata()``.

    Raises:
        TypeError: If ``t`` is not a
            :class:`~orchestrator.modules.operators.base.DiscoverySpaceSubscribingDiscoveryOperation`
            subclass, or if ``metadata.cls`` is set to a class other than ``t``.
    """
    if not issubclass(t, DiscoverySpaceSubscribingDiscoveryOperation):
        raise TypeError(
            f"@explore_operation: {t.__name__} must be a subclass of "
            "DiscoverySpaceSubscribingDiscoveryOperation (e.g. subclass "
            "`Explore` or another discovery operation that subscribes to the space)."
        )
    if metadata.cls is not None and metadata.cls is not t:
        raise TypeError(
            f"@explore_operation on {t.__name__}: operator_metadata().cls is "
            f"{metadata.cls!r} but the decorated class is {t!r}. "
            "Leave cls as None in operator_metadata() — the decorator sets it."
        )


def explore_operation(
    cls: "type[DiscoveryOperationBase]",
) -> "type[DiscoveryOperationBase]":
    """Decorator that registers an explore (search) operator class.

    All metadata is sourced from the class's ``operator_metadata()``
    classmethod.  The decorator generates an :class:`OperatorFunction`,
    validates its signature, registers it in the explore collection, and
    returns the **original class unchanged**::

        @explore_operation
        class MyOp(Explore):
            @classmethod
            def operator_metadata(cls) -> OperatorMetadata:
                return OperatorMetadata(
                    name="my_op",
                    version="0.1.0",
                    configuration_model=MyOpParameters,
                    example_configuration=MyOpParameters(),
                    type=DiscoveryOperationEnum.SEARCH,
                )

            async def run(self) -> OperationOutput | None: ...

    The generated operator function is accessible via
    ``explore.operators[name].function``; the class name continues to refer
    to the class itself.

    Args:
        cls: The operator class to register.  Must be a subclass of
            :class:`~orchestrator.modules.operators.base.DiscoverySpaceSubscribingDiscoveryOperation`
            and must implement ``operator_metadata()``.

    Returns:
        *cls* unchanged.

    Raises:
        NotImplementedError: If ``cls.operator_metadata()`` is not implemented.
        TypeError: If ``cls`` fails :func:`_validate_explore_cls`.
    """
    metadata = cls.operator_metadata()
    _validate_explore_cls(cls, metadata)
    op_name = metadata.name

    def _generated(
        discoverySpace: DiscoverySpace,
        operationInfo: FunctionOperationInfo | None = None,
        **kwargs: object,
    ) -> OperationOutput:
        return orchestrate_explore_operation(
            operator_metadata=explore.operators[op_name],
            discovery_space=discoverySpace,
            parameters=kwargs,
            operation_info=operationInfo or FunctionOperationInfo(),
        )

    _generated.__name__ = op_name
    _generated.__qualname__ = op_name
    validate_operator_function_signature(_generated)
    _generated = typing.cast("OperatorFunction", _generated)
    _warn_if_operator_name_reused("explore", op_name, explore.operators)
    explore.operators[op_name] = metadata.model_copy(
        update={
            "function": _generated,
            "cls": cls,
            "provenance": PackageProvenance.from_module_name(cls.__module__),
        }
    )
    return cls


def modify_operation(
    name: str,
    version: str,
    configuration_model: type[pydantic.BaseModel],
    example_configuration: pydantic.BaseModel,
    description: str | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a modify operation.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        @functools.wraps(func)
        def wrapper(
            discoverySpace: DiscoverySpace,
            operationInfo: FunctionOperationInfo | None = None,
            **kwargs: object,
        ) -> OperationOutput:
            return orchestrate_general_operation(
                operator_metadata=modify.operators[name],
                operation_parameters=kwargs,
                discovery_space=discoverySpace,
                operation_info=operationInfo or FunctionOperationInfo(),
            )

        validate_operator_function_signature(wrapper)
        wrapper = typing.cast("OperatorFunction", wrapper)
        _warn_if_operator_name_reused("modify", name, modify.operators)
        modify.operators[name] = OperatorMetadata(
            name=name,
            function=wrapper,
            version=version,
            description=description,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            type=DiscoveryOperationEnum.MODIFY,
            provenance=PackageProvenance.from_module_name(func.__module__),
        )
        return wrapper

    return _register


def export_operation(
    name: str,
    version: str,
    configuration_model: type[pydantic.BaseModel],
    example_configuration: pydantic.BaseModel,
    description: str | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as an export operation.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        @functools.wraps(func)
        def wrapper(
            discoverySpace: DiscoverySpace,
            operationInfo: FunctionOperationInfo | None = None,
            **kwargs: object,
        ) -> OperationOutput:
            return orchestrate_general_operation(
                operator_metadata=export.operators[name],
                operation_parameters=kwargs,
                discovery_space=discoverySpace,
                operation_info=operationInfo or FunctionOperationInfo(),
            )

        validate_operator_function_signature(wrapper)
        wrapper = typing.cast("OperatorFunction", wrapper)
        _warn_if_operator_name_reused("export", name, export.operators)
        export.operators[name] = OperatorMetadata(
            name=name,
            function=wrapper,
            version=version,
            description=description,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            type=DiscoveryOperationEnum.EXPORT,
            provenance=PackageProvenance.from_module_name(func.__module__),
        )
        return wrapper

    return _register


def provenance_for_operator(
    name: str, op_type: DiscoveryOperationEnum
) -> PackageProvenance | None:
    """Return the package provenance for a registered operator.

    Looks up the operator in the collection for ``op_type`` and returns the
    :class:`~orchestrator.core.metadata.PackageProvenance` recorded on its
    registry metadata at registration time.

    Args:
        name: Canonical operator name.
        op_type: The discovery operation type the operator belongs to.

    Returns:
        A :class:`~orchestrator.core.metadata.PackageProvenance` instance,
        or ``None`` if provenance is unavailable.
    """
    collection = operationCollectionMap.get(op_type)
    if collection is None:
        return None
    metadata = collection.operators.get(name)
    if metadata is None:
        return None
    return metadata.provenance


def load_operators() -> None:
    """Load all operator plugins via ``ado.operators`` entry points."""
    from importlib.metadata import entry_points

    for operator_plugin in entry_points(group="ado.operators"):
        try:
            operator_plugin.load()
            moduleLog.debug(
                f"Loaded plugin: {operator_plugin.name} from {operator_plugin.value}"
            )
        except Exception as e:  # noqa: PERF203
            moduleLog.error(f"Failed to load plugin {operator_plugin.name}: {e}")


# Load the operator plugins
load_operators()
