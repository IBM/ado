# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import functools
import logging
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

import orchestrator.core.metadata
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    DiscoveryOperationEnum,
    FunctionOperationInfo,
)
from orchestrator.modules.operators.base import DiscoveryOperationBase, OperationOutput
from orchestrator.modules.operators.orchestrate import orchestrate_general_operation

moduleLog = logging.getLogger("operation_collections")


class OperatorCollection(pydantic.BaseModel):
    type: DiscoveryOperationEnum
    function_operations: Annotated[
        dict[typing.AnyStr, typing.Callable], pydantic.Field(default_factory=dict)
    ]
    object_operations: Annotated[
        dict[typing.AnyStr, DiscoveryOperationBase],
        pydantic.Field(default_factory=dict),
    ]
    function_operation_models: Annotated[
        dict[typing.AnyStr, type[pydantic.BaseModel]],
        pydantic.Field(default_factory=dict),
    ]
    function_operation_model_defaults: Annotated[
        dict[typing.AnyStr, pydantic.BaseModel], pydantic.Field(default_factory=dict)
    ]
    function_operation_versions: Annotated[
        dict[typing.AnyStr, str], pydantic.Field(default_factory=dict)
    ]
    function_operation_descriptions: Annotated[
        dict[typing.AnyStr, str], pydantic.Field(default_factory=dict)
    ]
    operator_classes: Annotated[
        dict[typing.AnyStr, type],
        pydantic.Field(default_factory=dict),
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_operation_function(self, name: str, fn: typing.Callable) -> None:
        """Registers a callable under the given name."""
        self.function_operations[name] = fn

    def add_operation_version(self, name: str, version: str) -> None:
        """Stores the version string for the named operation."""
        self.function_operation_versions[name] = version

    def add_operation_description(self, name: str, version: str) -> None:
        """Stores the description string for the named operation."""
        self.function_operation_descriptions[name] = version

    def add_operation_configuration_model(
        self, name: str, model: type[pydantic.BaseModel]
    ) -> None:
        """Associates a configuration model with the named operation."""
        self.function_operation_models[name] = model

    def add_operation_configuration_model_default(
        self, name: str, default: pydantic.BaseModel
    ) -> None:
        """Stores the default configuration model instance for the named operation."""
        self.function_operation_model_defaults[name] = default

    def add_operation_object(self, name: str, object: DiscoveryOperationBase) -> None:
        """Registers an object-based operation under the given name."""
        self.object_operations[name] = object

    def add_operator_class(self, name: str, cls: type[DiscoveryOperationBase]) -> None:
        """Associates a Ray actor class with the named explore operation.

        Used by explore operators so that the registered name is the single
        source of truth and the class is an implementation detail looked up
        at runtime rather than stored in the operation YAML.

        Args:
            name: The registered operator name (matches the decorator name=).
            cls: The DiscoveryOperationBase subclass backing this operation.
        """
        self.operator_classes[name] = cls

    def operator_class_for_operation(self, name: str) -> type[DiscoveryOperationBase]:
        """Returns the actor class registered for the named explore operation.

        Args:
            name: The registered operator name.

        Returns:
            The DiscoveryOperationBase subclass for this operation.

        Raises:
            ValueError: If no class has been registered for the given name.
        """
        if name not in self.operator_classes:
            raise ValueError(f"No operator class registered for {name}")
        return self.operator_classes[name]

    def list_operations(self) -> list:
        """Returns all registered operation names (function and object based)."""
        return list(self.function_operations.keys()) + list(
            self.object_operations.keys()
        )

    def configuration_model_for_operation(self, name: str) -> type[pydantic.BaseModel]:
        """Returns the configuration model for the named operation.

        Args:
            name: The registered operator name.

        Returns:
            The pydantic model class used to validate parameters, or None if
            no model was registered.

        Raises:
            ValueError: If the operator name is not registered.
        """
        if name not in self.function_operation_models:
            raise ValueError(f"Unknown operator {name}")

        return self.function_operation_models.get(name)

    def default_configuration_model_for_operation(
        self, name: str
    ) -> pydantic.BaseModel:
        """Returns the default configuration model instance for the named operation.

        Args:
            name: The registered operator name.

        Returns:
            The default pydantic model instance, or None if none was registered.

        Raises:
            ValueError: If the operator name is not registered.
        """
        if name not in self.function_operation_models:
            raise ValueError(f"Unknown operator {name}")

        return self.function_operation_model_defaults.get(name)

    def description_for_operation(self, name: str) -> str:
        """Returns the description for the named operation.

        Args:
            name: The registered operator name.

        Returns:
            The description string, or None if none was registered.

        Raises:
            ValueError: If the operator name is not registered.
        """
        if name not in self.function_operation_models:
            raise ValueError(f"Unknown operator {name}")

        return self.function_operation_descriptions.get(name)

    def __getattr__(
        self, item: str
    ) -> typing.Callable[..., object] | DiscoveryOperationBase:
        if item in self.function_operations:
            retval = self.function_operations[item]
        elif item in self.object_operations:
            retval = self.object_operations[item]
        else:
            raise AttributeError(f"Unknown attribute {item}")

        return retval


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
# Decorators for registering operation functions
#


def register_characterize_operation(
    func: typing.Callable[..., object],
) -> typing.Callable[
    [DiscoverySpace, FunctionOperationInfo | None, dict[str, dict]], OperationOutput
]:
    @functools.wraps(func)
    def characterize_operation_wrapper(
        discoverySpace: DiscoverySpace,
        operationInfo: FunctionOperationInfo | None = None,
        **kwargs: dict,
    ) -> OperationOutput:

        return orchestrate_general_operation(
            operator_function=func,
            operation_parameters=kwargs,
            parameters_model=operationCollectionMap[
                DiscoveryOperationEnum.CHARACTERIZE
            ].configuration_model_for_operation(func.__name__),
            discovery_space=discoverySpace,
            operation_info=operationInfo or FunctionOperationInfo(),
            operation_type=orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE,
        )

    characterize.add_operation_function(func.__name__, characterize_operation_wrapper)

    return characterize_operation_wrapper


def characterize_operation(
    name: str,
    description: str | None = None,
    version: str | None = "v0.1",
    configuration_model: type[pydantic.BaseModel] | None = None,
    configuration_model_default: pydantic.BaseModel | None = None,
) -> typing.Callable[
    [typing.Callable[..., object]],
    typing.Callable[
        [DiscoverySpace, FunctionOperationInfo | None, dict[str, dict]], OperationOutput
    ],
]:
    characterize.add_operation_configuration_model(name, configuration_model)
    characterize.add_operation_configuration_model_default(
        name, configuration_model_default
    )
    characterize.add_operation_version(name, version)
    characterize.add_operation_description(name, description)

    return register_characterize_operation


def explore_operation(
    name: str,
    description: str | None = None,
    configuration_model: type[pydantic.BaseModel] | None = None,
    version: str | None = "v0.1",
    configuration_model_default: pydantic.BaseModel | None = None,
    operator_class: type[DiscoveryOperationBase] | None = None,
) -> typing.Callable[[typing.Callable[..., object]], typing.Callable[..., object]]:
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
    explore.add_operation_configuration_model(name, configuration_model)
    explore.add_operation_configuration_model_default(name, configuration_model_default)
    explore.add_operation_version(name, version)
    explore.add_operation_description(name, description)
    if operator_class is not None:
        explore.add_operator_class(name, operator_class)

    def _register(func: typing.Callable[..., object]) -> typing.Callable[..., object]:
        """Registers func under the outer decorator's ``name``."""
        explore.add_operation_function(name, func)
        return func

    return _register


def register_modify_operation(
    func: typing.Callable[..., object],
) -> typing.Callable[[typing.Callable[..., object]], OperatorCollection]:
    """Registers a function that modifies a discovery space to return a new discovery space"""

    @functools.wraps(func)
    def modify_operation_wrapper(
        discoverySpace: DiscoverySpace,
        operationInfo: FunctionOperationInfo | None = None,
        **kwargs: dict,
    ) -> OperationOutput:

        return orchestrate_general_operation(
            operator_function=func,
            operation_parameters=kwargs,
            parameters_model=operationCollectionMap[
                DiscoveryOperationEnum.MODIFY
            ].configuration_model_for_operation(func.__name__),
            discovery_space=discoverySpace,
            operation_info=operationInfo or FunctionOperationInfo(),
            operation_type=orchestrator.core.operation.config.DiscoveryOperationEnum.MODIFY,
        )

    modify.add_operation_function(func.__name__, modify_operation_wrapper)

    return modify


def modify_operation(
    name: str,
    description: str | None = None,
    version: str | None = "v0.1",
    configuration_model: type[pydantic.BaseModel] | None = None,
    configuration_model_default: pydantic.BaseModel | None = None,
) -> typing.Callable[[typing.Callable[..., object]], OperatorCollection]:
    modify.add_operation_configuration_model(name, configuration_model)
    modify.add_operation_configuration_model_default(name, configuration_model_default)
    modify.add_operation_version(name, version)
    modify.add_operation_description(name, description)

    return register_modify_operation


def register_export_operation(
    func: typing.Callable[..., object],
) -> typing.Callable[
    [DiscoverySpace, FunctionOperationInfo | None, dict[str, dict]], OperationOutput
]:
    """Registers a function that performs a lakehouse operation on a DiscoverySpace"""

    @functools.wraps(func)
    def export_operation_wrapper(
        discoverySpace: DiscoverySpace,
        operationInfo: FunctionOperationInfo | None = None,
        **kwargs: dict,
    ) -> OperationOutput:
        return orchestrate_general_operation(
            operator_function=func,
            operation_parameters=kwargs,
            parameters_model=operationCollectionMap[
                DiscoveryOperationEnum.EXPORT
            ].configuration_model_for_operation(func.__name__),
            discovery_space=discoverySpace,
            operation_info=operationInfo or FunctionOperationInfo(),
            operation_type=orchestrator.core.operation.config.DiscoveryOperationEnum.EXPORT,
        )

    export.add_operation_function(func.__name__, export_operation_wrapper)

    return export_operation_wrapper


def export_operation(
    name: str,
    description: str | None = None,
    configuration_model: type[pydantic.BaseModel] | None = None,
    version: str | None = "v0.1",
    configuration_model_default: pydantic.BaseModel | None = None,
) -> typing.Callable[
    [typing.Callable[..., object]],
    typing.Callable[
        [DiscoverySpace, FunctionOperationInfo | None, dict[str, dict]], OperationOutput
    ],
]:
    export.add_operation_configuration_model(name, configuration_model)
    export.add_operation_configuration_model_default(name, configuration_model_default)
    export.add_operation_version(name, version)
    export.add_operation_description(name, description)

    return register_export_operation


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
