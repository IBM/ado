# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
import importlib.metadata
import typing

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


class OperatorModuleConf(ModuleConf):
    moduleType: ModuleTypeEnum = pydantic.Field(default=ModuleTypeEnum.OPERATION)

    @property
    def operationType(self):
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
    operationType: DiscoveryOperationEnum = pydantic.Field(
        description="The type of the operation"
    )
    operatorName: str = pydantic.Field(description="The name of the operator")

    def validateOperatorExists(self):

        # Note: this is not implemented as a pydantic validator to avoid a
        # recursive import of agents.operations
        # This happens if an operator registers  a default operation configuration which instantiates this class
        # because the registrations happen on import of each operator

        import orchestrator.modules.operators.collections

        try:
            collection = (
                orchestrator.modules.operators.collections.operationCollectionMap[
                    self.operationType
                ]
            )
        except KeyError as e:
            raise ValueError(f"Unknown operation type {self.operationType}") from e

        function = collection.function_operations.get(self.operatorName)
        assert function is not None

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

    module: OperatorModuleConf | OperatorFunctionConf = pydantic.Field(
        default=OperatorModuleConf(),
        description="The module or function providing the discovery operation",
    )
    parameters: typing.Any = pydantic.Field(
        default={},
        description="The parameters for the operation. Operation provider dependent",
    )


class BaseOperationRunConfiguration(pydantic.BaseModel):
    """Field shared by OrchestratorRunConfiguration and OperationResourceConfiguration

    both are models used to run an operation"""

    operation: DiscoveryOperationConfiguration
    metadata: ConfigurationMetadata = pydantic.Field(
        default=ConfigurationMetadata(),
        description="User defined metadata about the configuration. A set of keys and values. "
        "Two optional keys that are used by convention are name and description",
    )
    actuatorConfigurationIdentifiers: list[str] = pydantic.Field(default=[])
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


        Raises: ValueError if there is more than one ActuatorConfigurationResource references the same actuator
        """

        import orchestrator.metastore.sqlstore

        if not self.actuatorConfigurationIdentifiers:
            return []

        sql = orchestrator.metastore.sqlstore.SQLStore(project_context=project_context)

        actuator_configurations = [
            sql.getResource(
                identifier=identifier,
                kind=CoreResourceKinds.ACTUATORCONFIGURATION,
                raise_error_if_no_resource=True,
            ).config
            for identifier in self.actuatorConfigurationIdentifiers
        ]

        actuator_identifiers = {
            conf.actuatorIdentifier for conf in actuator_configurations
        }
        if len(actuator_identifiers) != len(self.actuatorConfigurationIdentifiers):
            raise ValueError("Only one ActuatorConfiguration is permitted per Actuator")

        return actuator_configurations

    def validate_actuatorconfigurations_against_space(
        self,
        project_context: ProjectContext,
        discoverySpaceConfiguration: DiscoverySpaceConfiguration,
    ) -> list[ActuatorConfiguration]:

        actuator_configurations = self.get_actuatorconfigurations(
            project_context=project_context
        )
        actuator_identifiers = {
            conf.actuatorIdentifier for conf in actuator_configurations
        }

        # Check the actuators configurations refer to actuators used in the MeasurementSpace
        # The experiment identifiers are in two different locations
        if isinstance(
            discoverySpaceConfiguration.experiments, MeasurementSpaceConfiguration
        ):
            experiment_actuator_identifiers = {
                experiment.actuatorIdentifier
                for experiment in discoverySpaceConfiguration.experiments.experiments
            }
        else:
            experiment_actuator_identifiers = {
                experiment.actuatorIdentifier
                for experiment in discoverySpaceConfiguration.experiments
            }

        if not experiment_actuator_identifiers.issuperset(actuator_identifiers):
            raise ValueError(
                f"Actuator Identifiers {actuator_identifiers} must appear in the experiments of its space"
            )

        return actuator_configurations


class DiscoveryOperationResourceConfiguration(BaseOperationRunConfiguration):

    spaces: list[str] = pydantic.Field(
        description="The spaces the operation will be applied to"
    )

    @pydantic.field_validator("spaces")
    def check_space_set(cls, value):
        """Checks at least one space identifier has been given"""

        if len(value) == 0:
            raise ValueError(
                "You must provide at least one space identifier to an operation"
            )

        return value

    def validate_actuatorconfigurations(
        self, project_context: ProjectContext
    ) -> list[ActuatorConfiguration]:

        from orchestrator.core.discoveryspace.space import DiscoverySpace

        actuator_configurations: list[ActuatorConfiguration] = []
        for space in self.spaces:
            discovery_space = DiscoverySpace.from_stored_configuration(
                project_context=project_context,
                space_identifier=space,
            )

            actuator_configurations.extend(
                super().validate_actuatorconfigurations_against_space(
                    project_context=project_context,
                    discoverySpaceConfiguration=discovery_space.config,
                )
            )

        return actuator_configurations


class FunctionOperationInfo(pydantic.BaseModel):
    """Class for holding information for operations executed via operator functions

    Operators implemented as functions may need additional information.
    Rather that have these as multiple params we gather them in this model"""

    metadata: ConfigurationMetadata = pydantic.Field(
        default=ConfigurationMetadata(),
        description="User defined metadata about the configuration. A set of keys and values. "
        "Two optional keys that are used by convention are name and description",
    )
    actuatorConfigurationIdentifiers: list[str] = pydantic.Field(default=[])
