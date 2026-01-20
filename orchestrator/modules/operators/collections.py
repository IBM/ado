# Copyright (c) IBM Corporation
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


class OperationCollections(pydantic.BaseModel):
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_operation_function(self, name: str, fn: typing.Callable) -> None:
        self.function_operations[name] = fn

    def add_operation_version(self, name: str, version: str) -> None:
        self.function_operation_versions[name] = version

    def add_operation_description(self, name: str, version: str) -> None:
        self.function_operation_descriptions[name] = version

    def add_operation_configuration_model(
        self, name: str, model: type[pydantic.BaseModel]
    ) -> None:
        self.function_operation_models[name] = model

    def add_operation_configuration_model_default(
        self, name: str, default: pydantic.BaseModel
    ) -> None:
        self.function_operation_model_defaults[name] = default

    def add_operation_object(self, name: str, object: DiscoveryOperationBase) -> None:
        self.object_operations[name] = object

    def list_operations(self) -> list:
        return list(self.function_operations.keys()) + list(
            self.object_operations.keys()
        )

    def configuration_model_for_operation(self, name: str) -> type[pydantic.BaseModel]:
        if name not in self.function_operation_models:
            raise ValueError(f"Unknown operator {name}")

        return self.function_operation_models.get(name)

    def default_configuration_model_for_operation(
        self, name: str
    ) -> pydantic.BaseModel:
        if name not in self.function_operation_models:
            raise ValueError(f"Unknown operator {name}")

        return self.function_operation_model_defaults.get(name)

    def description_for_operation(self, name: str) -> str:
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


characterize = OperationCollections(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE
)
explore = OperationCollections(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH
)
modify = OperationCollections(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.MODIFY
)
export = OperationCollections(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.EXPORT
)
compare = OperationCollections(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.COMPARE
)
fuse = OperationCollections(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.FUSE
)
study = OperationCollections(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.STUDY
)
learn = OperationCollections(
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


def register_explore_operation(
    func: typing.Callable[..., object],
) -> typing.Callable[..., object]:
    """Registers a function that performs an explore operation on a DiscoverySpace"""

    # All explore operation function must call explore_operation_function_wrapper
    # This function will validate params, create OperationResource,
    # set up the necessary ray actors, run the operation etc.
    explore.add_operation_function(func.__name__, func)

    return func


def explore_operation(
    name: str,
    description: str | None = None,
    configuration_model: type[pydantic.BaseModel] | None = None,
    version: str | None = "v0.1",
    configuration_model_default: pydantic.BaseModel | None = None,
) -> typing.Callable[[typing.Callable[..., object]], typing.Callable[..., object]]:
    explore.add_operation_configuration_model(name, configuration_model)
    explore.add_operation_configuration_model_default(name, configuration_model_default)
    explore.add_operation_version(name, version)
    explore.add_operation_description(name, description)

    return register_explore_operation


def register_modify_operation(
    func: typing.Callable[..., object],
) -> typing.Callable[[typing.Callable[..., object]], OperationCollections]:
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
) -> typing.Callable[[typing.Callable[..., object]], OperationCollections]:
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
