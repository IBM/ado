# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import enum
import importlib.metadata
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict
from typing_extensions import Self

from orchestrator.core.actuatorconfiguration.config import ActuatorConfiguration
from orchestrator.core.discoveryspace.config import (
    DiscoverySpaceConfiguration,
)
from orchestrator.core.metadata import ConfigurationMetadata, PackageProvenance
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.project import ProjectContext
from orchestrator.modules.module import (
    ModuleConf,
    ModuleTypeEnum,
    load_module_class_or_function,
)
from orchestrator.schema.measurementspace import MeasurementSpaceConfiguration
from orchestrator.utilities.pydantic import Pep440VersionStr, ignore_plugin_validation

if typing.TYPE_CHECKING:
    import orchestrator.modules.operators.base


class DiscoveryOperationEnum(enum.Enum):
    CHARACTERIZE = "characterize"
    EXPLORE = "explore"
    COMPARE = "compare"
    MODIFY = "modify"
    STUDY = "study"
    FUSE = "fuse"
    LEARN = "learn"
    QUERY = "query"
    EXPORT = "export"
    SCRIPT = "script"

    @classmethod
    def _missing_(cls, value: object) -> "DiscoveryOperationEnum | None":
        """Accept the legacy 'search' value and redirect it to EXPLORE."""
        if value == "search":
            return cls.EXPLORE
        return None


def get_actuator_configurations(
    project_context: ProjectContext, actuator_configuration_identifiers: list[str]
) -> list[ActuatorConfiguration]:
    """Retrieves and validates actuator configurations from the metastore for use.

    Fetches ActuatorConfiguration resources from the metastore using the provided
    identifiers, validates parameters against actuator plugins, and checks that
    each actuator has at most one configuration. This is the use-path fetch;
    ``ado get`` and ``getResource`` do not call this function.

    Params:
        project_context: Project context for connecting to the metastore
        actuator_configuration_identifiers: List of identifiers for actuator
            configuration resources to retrieve

    Returns:
        List of ActuatorConfiguration instances validated for use

    Raises:
        ValueError: If more than one ActuatorConfiguration references the same actuator,
            or if actuator plugin validation fails
        ResourceDoesNotExistError: If any of the identifiers is not found in the project.
    """
    import orchestrator.metastore.sqlstore

    sql = orchestrator.metastore.sqlstore.SQLStore(project_context=project_context)

    actuator_configurations = [
        sql.getResource(
            identifier=identifier,
            kind=CoreResourceKinds.ACTUATORCONFIGURATION,
            raise_error_if_no_resource=True,
            ignore_plugin_validation=False,
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
        return c.operator_metadata().type

    @property
    def operatorIdentifier(self) -> str:
        c: type[orchestrator.modules.operators.base.DiscoveryOperationBase] = (
            load_module_class_or_function(self)
        )
        return c.operator_metadata().operatorIdentifier


class OperatorMetadata(pydantic.BaseModel):
    """Registry metadata for a registered operator."""

    name: Annotated[
        str,
        pydantic.Field(description="Canonical name the operator is registered under."),
    ]
    function: Annotated[
        typing.Callable | None,
        pydantic.Field(
            description=(
                "The callable implementing the operator. None when returned by "
                "operator_metadata() before the decorator injects it."
            ),
        ),
    ] = None
    version: Annotated[
        Pep440VersionStr,
        pydantic.Field(
            description=(
                "PEP 440 version string for the operator (e.g. '0.1.0', "
                "'1.2.3.dev4+abc.dirty').  Validated on construction."
            ),
        ),
    ] = "0.1.0"

    description: Annotated[
        str | None,
        pydantic.Field(
            description="Human-readable description of the operator.",
        ),
    ] = None
    configuration_model: Annotated[
        type[pydantic.BaseModel],
        pydantic.Field(
            description="Pydantic model class used to validate operation parameters.",
        ),
    ]
    example_configuration: Annotated[
        pydantic.BaseModel,
        pydantic.Field(
            description="Default instance of the configuration model.",
        ),
    ]
    cls: Annotated[
        type | None,
        pydantic.Field(
            description=(
                "For explore operators: the unwrapped Python class implementing the "
                "operator. None for function-only operators. The concrete "
                "modules/operators layer enforces that this is a DiscoveryOperationBase "
                "subclass; config.py treats it as an opaque type to stay decoupled."
            ),
        ),
    ] = None
    type: Annotated[
        DiscoveryOperationEnum,
        pydantic.Field(
            description="The discovery operation type this operator belongs to."
        ),
    ]
    provenance: Annotated[
        PackageProvenance | None,
        pydantic.Field(
            default=None,
            description=(
                "Python distribution that provides this operator, resolved from the "
                "installed environment at registration time. None when the operator "
                "module is not installed as a distribution package."
            ),
        ),
    ]

    @property
    def operatorIdentifier(self) -> str:
        """Canonical identifier for this operator: ``{name}-{version}``."""
        return f"{self.name}-{self.version}"


class OperatorReference(pydantic.BaseModel):
    """Identifies a registered operator by name and operation type.

    A lightweight reference used to look up an operator from the registry and
    dispatch to its callable.  Paired with :class:`OperatorMetadata`, which
    holds the full operator metadata stored in the registry.
    """

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
            not in operationCollectionMap[self.operationType].operators
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

        operator = collection.operators.get(self.operatorName)
        return operator.function if operator else None

    @property
    def operatorIdentifier(self) -> str:
        """Canonical identifier delegated to ``OperatorMetadata.operatorIdentifier``.

        Returns:
            ``"{operatorName}-{version}"`` as stored in the operator registry,
            or ``"{operatorName}-None"`` if the operator is not yet registered.
        """
        import orchestrator.modules.operators.collections

        collection = orchestrator.modules.operators.collections.operationCollectionMap[
            self.operationType
        ]

        operator = collection.operators.get(self.operatorName)
        return operator.operatorIdentifier if operator else f"{self.operatorName}-None"


class ScriptOperatorConf(pydantic.BaseModel):
    """Identifies an inline script or custom operator not registered in any collection."""

    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, pydantic.Field(description="Human-readable script name")]
    version: Annotated[str, pydantic.Field()] = "0.1.0"
    operationType: Annotated[
        DiscoveryOperationEnum,
        pydantic.Field(
            description=(
                "Semantic operation type (e.g. search, characterize). "
                "Script provenance is recorded separately via operation metadata labels."
            ),
        ),
    ] = DiscoveryOperationEnum.EXPLORE

    @property
    def operatorIdentifier(self) -> str:
        """Return the canonical script operator identifier."""
        return f"script-{self.name}-{self.version}"


# ---------------------------------------------------------------------------
# Backwards-compatibility alias — use OperatorReference in new code
# ---------------------------------------------------------------------------


class OperatorFunctionConf(OperatorReference):
    """Deprecated alias for :class:`OperatorReference`.

    .. deprecated::
        ``OperatorFunctionConf`` has been renamed to :class:`OperatorReference`.
        Update imports and instantiation sites to use ``OperatorReference``
        directly.
    """

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _warn_deprecated(
        cls, value: object, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> "OperatorFunctionConf":
        """Emit a deprecation warning whenever OperatorFunctionConf is instantiated.

        Args:
            value: The raw input value passed to the model.
            handler: The pydantic validation handler.

        Returns:
            The validated model instance.
        """
        import warnings

        warnings.warn(
            "OperatorFunctionConf has been renamed to OperatorReference. "
            "Update your import to use OperatorReference instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return handler(value)


class DiscoveryOperationConfiguration(pydantic.BaseModel):
    """Configuration for an operation agent"""

    model_config = ConfigDict(extra="forbid")

    module: Annotated[
        OperatorModuleConf | OperatorReference | ScriptOperatorConf,
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

    @pydantic.field_validator("module", mode="after")
    @classmethod
    def ensure_module_is_installed(
        cls,
        module: OperatorModuleConf | OperatorReference | ScriptOperatorConf,
        info: pydantic.ValidationInfo,
    ) -> OperatorModuleConf | OperatorReference | ScriptOperatorConf:
        """Validates that the operator module is installed and accessible.

        Args:
            module: The operator module or function configuration to validate.
            info: Pydantic validation info for the current validation step.

        Returns:
            The validated module configuration.

        Raises:
            ValueError: If the operator module is not installed or cannot be imported.
        """
        if ignore_plugin_validation(info):
            return module

        if isinstance(module, OperatorReference | ScriptOperatorConf):
            return module

        import importlib

        try:
            getattr(importlib.import_module(module.moduleName), module.moduleClass)
        except ModuleNotFoundError as e:
            raise ValueError(
                f"Operator {module.moduleName}.{module.moduleClass} is not installed"
            ) from e

        return module

    @pydantic.model_validator(mode="after")
    def validate_and_downcast_parameters(self, info: pydantic.ValidationInfo) -> Self:
        """Validates and downcasts operation parameters.

        For OperatorModuleConf modules, validates parameters using the operation's
        validateOperationParameters method. For OperatorReference modules,
        validates parameters against the configuration model if available.

        Args:
            info: Pydantic validation info for the current validation step.

        Returns:
            Self: The validated instance with downcast parameters.

        Raises:
            ValidationError: If parameter validation fails.
        """
        if ignore_plugin_validation(info):
            return self

        if isinstance(self.module, OperatorModuleConf):
            # This is guaranteed to not raise an error thanks to ensure_module_is_installed
            operator_class = getattr(
                importlib.import_module(self.module.moduleName), self.module.moduleClass
            )
            operator_metadata = operator_class.operator_metadata()
            self.parameters = operator_metadata.configuration_model.model_validate(
                self.parameters
            )
        elif isinstance(self.module, ScriptOperatorConf):
            self.parameters = {}
        else:
            from orchestrator.modules.operators.collections import (
                operationCollectionMap,
            )

            operation_type = self.module.operationType
            operator_name = self.module.operatorName
            operator_metadata = operationCollectionMap[operation_type].operators[
                operator_name
            ]
            self.parameters = operator_metadata.configuration_model.model_validate(
                self.parameters
            )

        return self


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
