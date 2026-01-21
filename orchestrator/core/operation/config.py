# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
import importlib.metadata
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

from orchestrator.core.actuatorconfiguration.config import ActuatorConfiguration
from orchestrator.core.discoveryspace.config import (
    DiscoverySpaceConfiguration,
)
from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.project import ProjectContext
from orchestrator.modules.module import (
    ModuleConf,
    ModuleTypeEnum,
    load_module_class_or_function,
)
from orchestrator.schema.measurementspace import MeasurementSpaceConfiguration

if typing.TYPE_CHECKING:
    import orchestrator.modules.operators.base


class DiscoveryOperationEnum(enum.Enum):
    CHARACTERIZE = "characterize"
    SEARCH = "search"
    COMPARE = "compare"
    MODIFY = "modify"
    STUDY = "study"
    FUSE = "fuse"
    LEARN = "learn"
    QUERY = "query"
    EXPORT = "export"


def get_actuator_configurations(
    project_context: ProjectContext, actuator_configuration_identifiers: list[str]
) -> list[ActuatorConfiguration]:
    """Retrieves actuator configurations from the metastore

    Fetches ActuatorConfiguration resources from the metastore using the provided
    identifiers and validates that each actuator has at most one configuration.

    Params:
        project_context: Project context for connecting to the metastore
        actuator_configuration_identifiers: List of identifiers for actuator
            configuration resources to retrieve

    Returns:
        List of ActuatorConfiguration instances retrieved from the metastore

    Raises:
        ValueError: If more than one ActuatorConfiguration references the same actuator
        ResourceDoesNotExistError: If any of the identifiers is not found in the project.
    """
    import orchestrator.metastore.sqlstore

    sql = orchestrator.metastore.sqlstore.SQLStore(project_context=project_context)

    actuator_configurations = [
        sql.getResource(
            identifier=identifier,
            kind=CoreResourceKinds.ACTUATORCONFIGURATION,
            raise_error_if_no_resource=True,
        ).config
        for identifier in actuator_configuration_identifiers
    ]

    actuator_identifiers = {conf.actuatorIdentifier for conf in actuator_configurations}
    if len(actuator_identifiers) != len(actuator_configuration_identifiers):
        raise ValueError("Only one ActuatorConfiguration is permitted per Actuator")

    return actuator_configurations


def validate_actuator_configurations_against_space_configuration(
    actuator_configurations: list[ActuatorConfiguration],
    discovery_space_configuration: DiscoverySpaceConfiguration,
) -> None:
    """Validates that actuator configurations are compatible with a discovery space

    Checks that all actuators referenced in the actuator configurations are used
    in the experiments defined in the discovery space configuration.

    Params:
        actuator_configurations: List of actuator configurations to validate
        discovery_space_configuration: The discovery space configuration to validate against


    Raises:
        ValueError: If any actuator identifier in actuator_configurations does not
            appear in the experiments of the discovery space
    """
    actuator_identifiers = {conf.actuatorIdentifier for conf in actuator_configurations}

    # Check the actuators configurations refer to actuators used in the MeasurementSpace
    # The experiment identifiers are in two different locations
    if isinstance(
        discovery_space_configuration.experiments, MeasurementSpaceConfiguration
    ):
        experiment_actuator_identifiers = {
            experiment.actuatorIdentifier
            for experiment in discovery_space_configuration.experiments.experiments
        }
    else:
        experiment_actuator_identifiers = {
            experiment.actuatorIdentifier
            for experiment in discovery_space_configuration.experiments
        }

    if not experiment_actuator_identifiers.issuperset(actuator_identifiers):
        raise ValueError(
            f"Actuator Identifiers {actuator_identifiers} must appear in the experiments of its space"
        )


def validate_actuator_configuration_ids_against_space_ids(
    actuator_configuration_identifiers: list[str],
    space_identifiers: list[str],
    project_context: ProjectContext,
) -> list[ActuatorConfiguration]:
    """Validates actuator configuration identifiers against space identifiers

    Retrieves actuator configurations and space configurations from the metastore,
    then validates that all actuator configurations are compatible with all specified
    discovery spaces.

    Params:
        actuator_configuration_identifiers: List of actuator configuration resource
            identifiers to validate
        space_identifiers: List of discovery space resource identifiers to validate against
        project_context: Project context for connecting to the metastore

    Returns:
        List of ActuatorConfiguration instances that were validated

    Raises:
        ValueError: If any actuator configuration is not compatible with any of the
            discovery spaces, or if more than one ActuatorConfiguration references
            the same actuator
        ResourceDoesNotExistError: If any of the identifiers is not found in the project.

    """
    import orchestrator.metastore.sqlstore

    sql = orchestrator.metastore.sqlstore.SQLStore(project_context=project_context)
    space_configurations: list[DiscoverySpaceConfiguration] = [
        sql.getResource(
            identifier=identifier,
            kind=CoreResourceKinds.DISCOVERYSPACE,
            raise_error_if_no_resource=True,
        ).config
        for identifier in space_identifiers
    ]

    actuator_configurations = get_actuator_configurations(
        project_context=project_context,
        actuator_configuration_identifiers=actuator_configuration_identifiers,
    )

    for config in space_configurations:
        validate_actuator_configurations_against_space_configuration(
            actuator_configurations=actuator_configurations,
            discovery_space_configuration=config,
        )

    return actuator_configurations


class OperatorModuleConf(ModuleConf):
    moduleType: Annotated[ModuleTypeEnum, pydantic.Field()] = ModuleTypeEnum.OPERATION

    @property
    def operationType(self) -> DiscoveryOperationEnum:
        c: type[orchestrator.modules.operators.base.DiscoveryOperationBase] = (
            load_module_class_or_function(self)
        )
        return c.operationType()

    @property
    def operatorIdentifier(self) -> str:
        c: type[orchestrator.modules.operators.base.DiscoveryOperationBase] = (
            load_module_class_or_function(self)
        )

        return c.operatorIdentifier()


class OperatorFunctionConf(pydantic.BaseModel):
    """Describes an operator vended as a function"""

    model_config = ConfigDict(extra="forbid")
    operationType: Annotated[
        DiscoveryOperationEnum, pydantic.Field(description="The type of the operation")
    ]
    operatorName: Annotated[str, pydantic.Field(description="The name of the operator")]

    def validateOperatorExists(self) -> bool:

        # Note: this is not implemented as a pydantic validator to avoid a
        # recursive import of agents.operations
        # This happens if an operator registers  a default operation configuration which instantiates this class
        # because the registrations happen on import of each operator

        from orchestrator.modules.operators.collections import operationCollectionMap

        if self.operationType not in operationCollectionMap:
            raise ValueError(f"Unknown operation type {self.operationType}")

        if (
            self.operatorName
            not in operationCollectionMap[self.operationType].function_operations
        ):
            raise ValueError(
                f"Operator {self.operatorName} had no functions of type {self.operationType}"
            )

        return True

    def operationFunction(
        self,
    ) -> "typing.Callable[..., orchestrator.modules.operators.base.OperationOutput]":

        import orchestrator.modules.operators.collections

        collection = orchestrator.modules.operators.collections.operationCollectionMap[
            self.operationType
        ]

        return collection.function_operations.get(self.operatorName)

    @property
    def operatorIdentifier(self) -> str:

        import orchestrator.modules.operators.collections

        collection = orchestrator.modules.operators.collections.operationCollectionMap[
            self.operationType
        ]

        return f"{self.operatorName}-{collection.function_operation_versions.get(self.operatorName)}"


class DiscoveryOperationConfiguration(pydantic.BaseModel):
    """Configuration for an operation agent"""

    model_config = ConfigDict(extra="forbid")

    module: Annotated[
        OperatorModuleConf | OperatorFunctionConf,
        pydantic.Field(
            description="The module or function providing the discovery operation"
        ),
    ] = OperatorModuleConf()
    parameters: Annotated[
        typing.Any,
        pydantic.Field(
            default_factory=dict,
            description="The parameters for the operation. Operation provider dependent",
        ),
    ]


class DiscoveryOperationResourceConfiguration(pydantic.BaseModel):
    """Pydantic model used to define an operation"""

    operation: DiscoveryOperationConfiguration
    metadata: Annotated[
        ConfigurationMetadata,
        pydantic.Field(
            description="Metadata about the configuration including optional name, description, "
            "labels for filtering, and any additional custom fields"
        ),
    ] = ConfigurationMetadata()
    actuatorConfigurationIdentifiers: Annotated[
        list[str], pydantic.Field(default_factory=list)
    ]
    spaces: Annotated[
        list[str],
        pydantic.Field(
            description="List of ids of the spaces the operation will be applied to. "
            "Currently, only one identifier is supported.",
            min_length=1,
            max_length=1,
        ),
    ]
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "version": importlib.metadata.version(distribution_name="ado-core")
        },
    )

    def get_actuatorconfigurations(
        self, project_context: ProjectContext
    ) -> list[ActuatorConfiguration]:
        """Gets the actuator configuration resources referenced by actuatorConfigurationIdentifiers from the metastore if any

        Params:
            project_context: Information for connection to the metastore

        Returns:
            A list of ActuatorConfigurationResource instance. The list will be empty if
            there are no actuatorConfigurationIdentifiers.


        Raises:
            ValueError if there is more than one ActuatorConfigurationResource references the same actuator
            ResourceDoesNotExistError if any actuator configuration identifier cannot be found in the project
        """

        if not self.actuatorConfigurationIdentifiers:
            return []

        return get_actuator_configurations(
            project_context=project_context,
            actuator_configuration_identifiers=self.actuatorConfigurationIdentifiers,
        )

    def validate_actuatorconfigurations(
        self, project_context: ProjectContext
    ) -> list[ActuatorConfiguration]:
        """Gets and valdidates the actuator configuration resources referenced by actuatorConfigurationIdentifiers from the metastore if any

        This also requires getting the configuration of the discovery space

        Params:
            project_context: Information for connection to the metastore

        Returns:
            A list of ActuatorConfigurationResource instance. The list will be empty if
            there are no actuatorConfigurationIdentifiers.


        Raises: ValueError if more than one ActuatorConfigurationResource references the same actuator
        """

        return validate_actuator_configuration_ids_against_space_ids(
            actuator_configuration_identifiers=self.actuatorConfigurationIdentifiers,
            space_identifiers=self.spaces,
            project_context=project_context,
        )


class FunctionOperationInfo(pydantic.BaseModel):
    """Class for providing information to operator functions"""

    metadata: Annotated[
        ConfigurationMetadata,
        pydantic.Field(
            description="Metadata about the configuration including optional name, description, "
            "labels for filtering, and any additional custom fields"
        ),
    ] = ConfigurationMetadata()
    actuatorConfigurationIdentifiers: Annotated[
        list[str], pydantic.Field(default_factory=list)
    ]
    ray_namespace: Annotated[
        str | None,
        pydantic.Field(
            description="The namespace the operation should create ray workers/actors in"
        ),
    ] = None
