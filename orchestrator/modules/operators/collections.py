# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import functools
import logging
import typing
import warnings
from typing import Annotated

import pydantic
from pydantic import ConfigDict

import orchestrator.core.metadata
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    DiscoveryOperationEnum,
    FunctionOperationInfo,
)
from orchestrator.modules.operators.base import (
    DiscoveryOperationBase,
    OperationOutput,
    OperatorFunction,
)
from orchestrator.modules.operators.orchestrate import orchestrate_general_operation

moduleLog = logging.getLogger("operation_collections")


class Operator(pydantic.BaseModel):
    """Metadata and implementation for a registered operator.

    Attributes:
        name: Canonical name the operator is registered under.
        function: The callable implementing the operator.
        version: Version string for the operator (e.g. "v0.1").
        description: Human-readable description of the operator.
        configuration_model: Pydantic model class used to validate parameters.
        example_configuration: Default instance of the configuration model.
        cls: Ray actor class backing the operator (explore operators only).
        type: The discovery operation type this operator belongs to.
    """

    name: str
    function: OperatorFunction
    version: Annotated[str, pydantic.Field(default="v0.1")]
    description: Annotated[str | None, pydantic.Field(default=None)]
    configuration_model: Annotated[
        type[pydantic.BaseModel] | None, pydantic.Field(default=None)
    ]
    example_configuration: Annotated[
        pydantic.BaseModel | None, pydantic.Field(default=None)
    ]
    cls: Annotated[typing.Any, pydantic.Field(default=None)]
    type: DiscoveryOperationEnum

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OperatorCollection(pydantic.BaseModel):
    """A registry of operators of a single discovery operation type.

    Operators are added via the decorator functions (e.g. ``characterize_operation``,
    ``explore_operation``).  Each registered name maps to an :class:`Operator`
    instance that carries the function, version, description, configuration model,
    example configuration, and optional actor class.

    Attributes:
        type: The discovery operation type all operators in this collection belong to.
        operators: Mapping of operator name to :class:`Operator` instance.
    """

    type: DiscoveryOperationEnum
    operators: Annotated[dict[str, Operator], pydantic.Field(default_factory=dict)]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def list_operators(self) -> list[str]:
        """Returns all registered operator names."""
        return list(self.operators.keys())

    def list_operations(self) -> list[str]:
        """Returns all registered operator names.

        Deprecated:
            Use :meth:`list_operators` instead.
        """
        warnings.warn(
            "list_operations() is deprecated; use list_operators() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_operators()

    def configuration_model_for_operation(
        self, name: str
    ) -> type[pydantic.BaseModel] | None:
        """Returns the configuration model for the named operator.

        Deprecated:
            Access ``collection.operators[name].configuration_model`` directly.

        Args:
            name: The registered operator name.

        Returns:
            The pydantic model class used to validate parameters, or None.

        Raises:
            ValueError: If the operator name is not registered.
        """
        warnings.warn(
            "configuration_model_for_operation() is deprecated; "
            "access collection.operators[name].configuration_model directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name not in self.operators:
            raise ValueError(f"Unknown operator {name}")
        return self.operators[name].configuration_model

    def default_configuration_model_for_operation(
        self, name: str
    ) -> pydantic.BaseModel | None:
        """Returns the example configuration instance for the named operator.

        Deprecated:
            Access ``collection.operators[name].example_configuration`` directly.

        Args:
            name: The registered operator name.

        Returns:
            The default pydantic model instance, or None.

        Raises:
            ValueError: If the operator name is not registered.
        """
        warnings.warn(
            "default_configuration_model_for_operation() is deprecated; "
            "access collection.operators[name].example_configuration directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name not in self.operators:
            raise ValueError(f"Unknown operator {name}")
        return self.operators[name].example_configuration

    def description_for_operation(self, name: str) -> str | None:
        """Returns the description for the named operator.

        Deprecated:
            Access ``collection.operators[name].description`` directly.

        Args:
            name: The registered operator name.

        Returns:
            The description string, or None.

        Raises:
            ValueError: If the operator name is not registered.
        """
        warnings.warn(
            "description_for_operation() is deprecated; "
            "access collection.operators[name].description directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name not in self.operators:
            raise ValueError(f"Unknown operator {name}")
        return self.operators[name].description

    def operator_class_for_operation(self, name: str) -> type[DiscoveryOperationBase]:
        """Returns the actor class registered for the named explore operator.

        Deprecated:
            Access ``collection.operators[name].cls`` directly.

        Args:
            name: The registered operator name.

        Returns:
            The DiscoveryOperationBase subclass for this operator.

        Raises:
            ValueError: If no class has been registered for the given name.
        """
        warnings.warn(
            "operator_class_for_operation() is deprecated; "
            "access collection.operators[name].cls directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name not in self.operators or self.operators[name].cls is None:
            raise ValueError(f"No operator class registered for {name}")
        return self.operators[name].cls

    def __getattr__(self, item: str) -> OperatorFunction:
        """Returns the operator function for the given registered name.

        Args:
            item: Registered operator name.

        Returns:
            The callable registered under that name.

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
    description: str | None = None,
    version: str | None = "v0.1",
    configuration_model: type[pydantic.BaseModel] | None = None,
    configuration_model_default: pydantic.BaseModel | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a characterize operation.

    Args:
        name: Canonical operator name used in the registry.
        description: Human-readable description shown in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        configuration_model_default: Default parameter model instance.

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
                operator_function=func,
                operation_parameters=kwargs,
                parameters_model=characterize.operators[name].configuration_model,
                discovery_space=discoverySpace,
                operation_info=operationInfo or FunctionOperationInfo(),
                operation_type=orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE,
            )

        characterize.operators[name] = Operator(
            name=name,
            function=typing.cast("OperatorFunction", wrapper),
            version=version or "v0.1",
            description=description,
            configuration_model=configuration_model,
            example_configuration=configuration_model_default,
            type=DiscoveryOperationEnum.CHARACTERIZE,
        )
        return typing.cast("OperatorFunction", wrapper)

    return _register


def explore_operation(
    name: str,
    description: str | None = None,
    configuration_model: type[pydantic.BaseModel] | None = None,
    version: str | None = "v0.1",
    configuration_model_default: pydantic.BaseModel | None = None,
    operator_class: type[DiscoveryOperationBase] | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as an explore (search) operation.

    The decorator is the single source of truth for the operator name and
    version. When ``operator_class`` is supplied the actor class is stored
    in the collection keyed by ``name``, so the class is an implementation
    detail and does not need to appear in the operation YAML.

    Args:
        name: Canonical operator name used in the registry, ``ado get operators``,
            and the stored ``operatorIdentifier``.
        description: Human-readable description shown in the registry.
        configuration_model: Pydantic model used to validate operation parameters.
        version: Version string included in the ``operatorIdentifier``.
        configuration_model_default: Default parameter model instance.
        operator_class: The Ray-actor class that implements this operation.
            Must be provided for explore operators that use the function-conf
            path (``OperatorFunctionConf``).

    Returns:
        A decorator that registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        """Registers func under the outer decorator's ``name``."""
        explore.operators[name] = Operator(
            name=name,
            function=func,
            version=version or "v0.1",
            description=description,
            configuration_model=configuration_model,
            example_configuration=configuration_model_default,
            cls=operator_class,
            type=DiscoveryOperationEnum.SEARCH,
        )
        return func

    return _register


def modify_operation(
    name: str,
    description: str | None = None,
    version: str | None = "v0.1",
    configuration_model: type[pydantic.BaseModel] | None = None,
    configuration_model_default: pydantic.BaseModel | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a modify operation.

    Args:
        name: Canonical operator name used in the registry.
        description: Human-readable description shown in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        configuration_model_default: Default parameter model instance.

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
                operator_function=func,
                operation_parameters=kwargs,
                parameters_model=modify.operators[name].configuration_model,
                discovery_space=discoverySpace,
                operation_info=operationInfo or FunctionOperationInfo(),
                operation_type=orchestrator.core.operation.config.DiscoveryOperationEnum.MODIFY,
            )

        modify.operators[name] = Operator(
            name=name,
            function=typing.cast("OperatorFunction", wrapper),
            version=version or "v0.1",
            description=description,
            configuration_model=configuration_model,
            example_configuration=configuration_model_default,
            type=DiscoveryOperationEnum.MODIFY,
        )
        return typing.cast("OperatorFunction", wrapper)

    return _register


def export_operation(
    name: str,
    description: str | None = None,
    configuration_model: type[pydantic.BaseModel] | None = None,
    version: str | None = "v0.1",
    configuration_model_default: pydantic.BaseModel | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as an export operation.

    Args:
        name: Canonical operator name used in the registry.
        description: Human-readable description shown in the registry.
        configuration_model: Pydantic model used to validate operation parameters.
        version: Version string included in the operator identifier.
        configuration_model_default: Default parameter model instance.

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
                operator_function=func,
                operation_parameters=kwargs,
                parameters_model=export.operators[name].configuration_model,
                discovery_space=discoverySpace,
                operation_info=operationInfo or FunctionOperationInfo(),
                operation_type=orchestrator.core.operation.config.DiscoveryOperationEnum.EXPORT,
            )

        export.operators[name] = Operator(
            name=name,
            function=typing.cast("OperatorFunction", wrapper),
            version=version or "v0.1",
            description=description,
            configuration_model=configuration_model,
            example_configuration=configuration_model_default,
            type=DiscoveryOperationEnum.EXPORT,
        )
        return typing.cast("OperatorFunction", wrapper)

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
