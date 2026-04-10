# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import functools
import logging
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

import orchestrator.core.operation.config
from orchestrator.core.discoveryspace.space import DiscoverySpace
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    configuration_model_default: pydantic.BaseModel,
    description: str | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a characterize operation.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        configuration_model_default: Default parameter model instance.
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
        characterize.operators[name] = OperatorMetadata(
            name=name,
            function=wrapper,
            version=version,
            description=description,
            configuration_model=configuration_model,
            example_configuration=configuration_model_default,
            type=DiscoveryOperationEnum.CHARACTERIZE,
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
            "DiscoverySpaceSubscribingDiscoveryOperation (i.e. inherit from "
            "Search or Characterize)."
        )
    if metadata.cls is not None and metadata.cls is not t:
        raise TypeError(
            f"@explore_operation on {t.__name__}: operator_metadata().cls is "
            f"{metadata.cls!r} but the decorated class is {t!r}. "
            "Leave cls as None in operator_metadata() — the decorator sets it."
        )


def explore_operation(
    cls: "type[DiscoveryOperationBase]",
) -> OperatorFunction:
    """Decorator that registers an explore (search) operator class.

    All metadata is sourced from the class's ``operator_metadata()``
    classmethod.  The decorator generates an :class:`OperatorFunction`, validates
    its signature, and registers it in the explore collection::

        @explore_operation
        class MyOp(Search):
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

    Args:
        cls: The operator class to register.  Must be a subclass of
            :class:`~orchestrator.modules.operators.base.DiscoverySpaceSubscribingDiscoveryOperation`
            and must implement ``operator_metadata()``.

    Returns:
        The generated :data:`~orchestrator.modules.operators.base.OperatorFunction`
        for this operator.

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
    explore.operators[op_name] = metadata.model_copy(
        update={
            "function": _generated,
            "cls": cls,
        }
    )
    return _generated


def modify_operation(
    name: str,
    version: str,
    configuration_model: type[pydantic.BaseModel],
    configuration_model_default: pydantic.BaseModel,
    description: str | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a modify operation.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        configuration_model_default: Default parameter model instance.
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
        modify.operators[name] = OperatorMetadata(
            name=name,
            function=wrapper,
            version=version,
            description=description,
            configuration_model=configuration_model,
            example_configuration=configuration_model_default,
            type=DiscoveryOperationEnum.MODIFY,
        )
        return wrapper

    return _register


def export_operation(
    name: str,
    version: str,
    configuration_model: type[pydantic.BaseModel],
    configuration_model_default: pydantic.BaseModel,
    description: str | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as an export operation.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        configuration_model_default: Default parameter model instance.
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
        export.operators[name] = OperatorMetadata(
            name=name,
            function=wrapper,
            version=version,
            description=description,
            configuration_model=configuration_model,
            example_configuration=configuration_model_default,
            type=DiscoveryOperationEnum.EXPORT,
        )
        return wrapper

    return _register


def load_operators() -> None:
    from importlib.metadata import entry_points

    import orchestrator.modules.operators.randomwalk  # noqa: F401

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
