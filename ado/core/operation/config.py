# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import enum
import importlib.metadata
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict
from typing_extensions import Self

from ado.core.actuatorconfiguration.config import ActuatorConfiguration
from ado.core.discoveryspace.config import (
    DiscoverySpaceConfiguration,
)
from ado.core.metadata import ConfigurationMetadata, PackageProvenance
from ado.core.resources import (
    ADOResourcePropertyDescriptor,
    ADOResourceReference,
    CoreResourceKinds,
)
from ado.metastore.project import ProjectContext
from ado.modules.module import (
    ModuleConf,
    ModuleTypeEnum,
    load_module_class_or_function,
)
from ado.schema.measurementspace import MeasurementSpaceConfiguration
from ado.utilities.pydantic import StrictSemVerStr, ignore_plugin_validation

if typing.TYPE_CHECKING:
    import ado.modules.operators.base
    from ado.metastore.sqlstore import SQLStore

#: Represents the default typed input parameter used for operators that work on a single discovery space.
# i.e. a parameter `discoverySpace: DiscoverySpace`
_DEFAULT_DISCOVERY_SPACE_INPUT_PROPERTY = ADOResourcePropertyDescriptor(
    identifier="discoverySpace",
    kind=CoreResourceKinds.DISCOVERYSPACE,
)


class GenericOperatorParameters(pydantic.BaseModel):
    """Base class for operator parameter (configuration) models.

    Operator-specific parameter classes should subclass this. Schemas are
    closed: unknown fields are rejected (``extra="forbid"``).
    """

    model_config = ConfigDict(extra="forbid")


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
    actuator_configuration_identifiers: list[str],
    metastore: "SQLStore",
) -> list[ActuatorConfiguration]:
    """Retrieves and validates actuator configurations from the metastore for use.

    Fetches ActuatorConfiguration resources from the metastore using the provided
    identifiers, validates parameters against actuator plugins, and checks that
    each actuator has at most one configuration. This is the use-path fetch;
    ``ado get`` and ``getResource`` do not call this function.

    Args:
        actuator_configuration_identifiers: List of identifiers for actuator
            configuration resources to retrieve.
        metastore: Metastore to read from.

    Returns:
        List of ActuatorConfiguration instances validated for use.

    Raises:
        ValueError: If more than one ActuatorConfiguration references the same
            actuator, or if actuator plugin validation fails.
        ResourceDoesNotExistError: If any of the identifiers is not found in the
            project.
    """

    actuator_configurations = [
        metastore.getResource(
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
    """Validates that actuator configurations are compatible with a discovery space.

    Checks that all actuators referenced in the actuator configurations are used
    in the experiments defined in the discovery space configuration.

    Args:
        actuator_configurations: List of actuator configurations to validate.
        discovery_space_configuration: The discovery space configuration to
            validate against.

    Raises:
        ValueError: If any actuator identifier in actuator_configurations does not
            appear in the experiments of the discovery space.
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
    """Validates actuator configuration identifiers against space identifiers.

    Retrieves actuator configurations and space configurations from the metastore,
    then validates that all actuator configurations are compatible with all specified
    discovery spaces.

    Args:
        actuator_configuration_identifiers: List of actuator configuration resource
            identifiers to validate.
        space_identifiers: List of discovery space resource identifiers to validate
            against.
        project_context: Project context for connecting to the metastore.

    Returns:
        List of ActuatorConfiguration instances that were validated.

    Raises:
        ValueError: If any actuator configuration is not compatible with any of the
            discovery spaces, or if more than one ActuatorConfiguration references
            the same actuator.
        ResourceDoesNotExistError: If any of the identifiers is not found in the
            project.
    """
    import ado.metastore.sqlstore

    sql = ado.metastore.sqlstore.SQLStore(project_context=project_context)
    space_configurations: list[DiscoverySpaceConfiguration] = [
        sql.getResource(
            identifier=identifier,
            kind=CoreResourceKinds.DISCOVERYSPACE,
            raise_error_if_no_resource=True,
        ).config
        for identifier in space_identifiers
    ]

    actuator_configurations = get_actuator_configurations(
        actuator_configuration_identifiers=actuator_configuration_identifiers,
        metastore=sql,
    )

    for config in space_configurations:
        validate_actuator_configurations_against_space_configuration(
            actuator_configurations=actuator_configurations,
            discovery_space_configuration=config,
        )

    return actuator_configurations


class OperatorModuleConf(ModuleConf):
    moduleType: Annotated[ModuleTypeEnum, pydantic.Field()] = ModuleTypeEnum.OPERATION

    @classmethod
    def _warn_legacy_module_name(cls, old_name: str, new_name: str) -> None:
        from ado.core.resources import (
            CoreResourceKinds,
            warn_deprecated_resource_model_in_use,
        )

        warn_deprecated_resource_model_in_use(
            affected_resource=CoreResourceKinds.OPERATION,
            deprecated_from_ado_version="2.0.0",
            removed_from_ado_version="3.0.0",
            deprecated_fields="moduleName",
            latest_format_documentation_url=(
                "https://ibm.github.io/ado/migration/1x-to-2x/"
                "#renamed-python-import-package-orchestrator-ado"
            ),
        )

    @property
    def operationType(self) -> DiscoveryOperationEnum:
        c: type[ado.modules.operators.base.DiscoveryOperationBase] = (
            load_module_class_or_function(self)
        )
        return c.operator_metadata().type

    @property
    def operatorIdentifier(self) -> str:
        c: type[ado.modules.operators.base.DiscoveryOperationBase] = (
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
        StrictSemVerStr,
        pydantic.Field(
            description=(
                "Versioning information for this operator (strict SemVer "
                "MAJOR.MINOR.PATCH)."
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
        type[GenericOperatorParameters],
        pydantic.Field(
            description="Pydantic model class used to validate operation parameters.",
        ),
    ]
    example_configuration: Annotated[
        GenericOperatorParameters,
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
    required_resource_inputs: Annotated[
        tuple[ADOResourcePropertyDescriptor, ...],
        pydantic.Field(
            default_factory=tuple,
            description=(
                "Ordered list describing each operator parameter whose type is an "
                "ADO resource. Deduced from the operator function signature at "
                "registration; operators must have at least one resource input."
            ),
        ),
    ]
    required_properties: Annotated[
        list[str] | None,
        pydantic.Field(
            default=None,
            description=(
                "Target property identifiers this operator reads from a discovery space if any. "
            ),
        ),
    ]

    @property
    def operatorIdentifier(self) -> str:
        """Canonical identifier for this operator: ``{name}@{version}``."""
        return f"{self.name}@{self.version}"

    @property
    def reference(self) -> "OperatorReference":
        """Return an :class:`OperatorReference` for this operator."""
        return OperatorReference(
            operatorName=self.name,
            operationType=self.type,
            operatorVersion=self.version,
        )


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
    operatorVersion: Annotated[
        StrictSemVerStr | None,
        pydantic.Field(
            description=(
                "Versioning information of the referenced operator (strict SemVer "
                "MAJOR.MINOR.PATCH). When omitted at creation time, resolved from "
                "the operator registry and pinned on the stored resource."
            ),
        ),
    ] = None

    def validateOperatorExists(self) -> bool:

        # Note: this is not implemented as a pydantic validator to avoid a
        # recursive import of agents.operations
        # This happens if an operator registers  a default operation configuration which instantiates this class
        # because the registrations happen on import of each operator

        from ado.modules.operators.collections import operationCollectionMap

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
    ) -> "typing.Callable[..., ado.modules.operators.base.OperationOutput] | None":

        import ado.modules.operators.collections

        collection = ado.modules.operators.collections.operationCollectionMap[
            self.operationType
        ]

        operator = collection.operators.get(self.operatorName)
        return operator.function if operator else None

    @property
    def operatorIdentifier(self) -> str:
        """Canonical identifier for this operator reference.

        Returns:
            ``"{operatorName}@{version}"`` using the pinned ``operatorVersion``
            when set, otherwise the version from the operator registry, or
            ``"{operatorName}@None"`` if the operator is not registered.
        """
        if self.operatorVersion is not None:
            return f"{self.operatorName}@{self.operatorVersion}"

        from ado.modules.operators.collections import (
            operator_metadata_for_reference,
        )

        try:
            metadata = operator_metadata_for_reference(self)
        except ValueError:
            return f"{self.operatorName}@None"
        return metadata.operatorIdentifier


class ScriptOperatorConf(pydantic.BaseModel):
    """Identifies an inline script or custom operator not registered in any collection."""

    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, pydantic.Field(description="Human-readable script name")]
    version: Annotated[str, pydantic.Field()] = "0.1.0"
    operationType: Annotated[
        DiscoveryOperationEnum,
        pydantic.Field(
            description=(
                "Semantic operation type (e.g. explore, characterize). "
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
            from ado.modules.operators.collections import (
                operator_metadata_for_reference,
                resolve_operator_reference,
            )

            self.module = resolve_operator_reference(self.module)
            operator_metadata = operator_metadata_for_reference(self.module)
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
    inputs: Annotated[
        dict[str, ADOResourceReference],
        pydantic.Field(
            default_factory=dict,
            description=(
                "Details the resources in the metastore that will be passed as values to the given operator parameters"
            ),
        ),
    ]
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "version": importlib.metadata.version(distribution_name="ado-core")
        },
    )

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def spaces(self) -> list[str]:
        """Derived list of discoveryspace identifiers from ``inputs``.

        Returns all input identifiers whose ``kind`` is
        :attr:`~ado.core.resources.CoreResourceKinds.DISCOVERYSPACE`.  When
        ``kind`` is ``None`` (e.g. during dry-run with plugin validation
        disabled), falls back to checking the identifier prefix ``"space-"``.

        Returns:
            List of discoveryspace resource identifiers in insertion order.
        """
        return [
            e.identifier
            for e in self.inputs.values()
            if e.kind == CoreResourceKinds.DISCOVERYSPACE
            or (e.kind is None and e.identifier.startswith("space-"))
        ]

    @pydantic.model_validator(mode="before")
    @classmethod
    def upgrade_legacy_spaces(cls, data: object) -> object:
        """Upgrade legacy ``spaces:`` YAML to the new ``inputs:`` format.

        Strips the ``spaces`` key from raw input data before pydantic
        validates it (preventing ``extra="forbid"`` errors) and converts
        the first space identifier to an ``inputs`` entry using the
        conventional name ``"discoverySpace"``.

        Rules applied:

        1. ``spaces`` present, ``inputs`` absent → convert first space to
           ``inputs: {discoverySpace: {identifier: <id>}}``.
        2. ``spaces`` present, ``inputs`` also present → discard ``spaces``
           (caller is already using the new format).
        3. ``spaces`` absent → no-op.

        Args:
            data: Raw model input data.

        Returns:
            Possibly modified data dict.
        """
        if not isinstance(data, dict):
            return data
        if "spaces" not in data:
            return data
        data = dict(data)
        # We need to remove spaces as pydantic does not handle
        # the situation of computed fields with extra=forbid
        spaces = data.pop("spaces")
        if "inputs" not in data and spaces:
            data["inputs"] = {"discoverySpace": {"identifier": spaces[0]}}
        return data

    @pydantic.model_validator(mode="after")
    def validate_inputs(self, info: pydantic.ValidationInfo) -> Self:
        """Validate and populate ``kind`` on all input entries.

        For :class:`OperatorModuleConf` and :class:`OperatorReference`
        operators, loads the operator's ``required_resource_inputs`` and:

        * fills ``kind=None`` entries from the descriptor map,
        * checks every entry's (explicit or filled) kind matches the
          descriptor's declared kind, and
        * checks all required inputs are present (``inputs`` must be
          non-empty).

        For :class:`ScriptOperatorConf` operators, which are always
        created via :meth:`~ado.core.discoveryspace.space.DiscoverySpace.operation_context`:

        * exactly one input is expected,
        * ``kind`` must be explicitly set (never ``None``), and
        * ``kind`` must be :attr:`~ado.core.resources.CoreResourceKinds.DISCOVERYSPACE`.

        Skipped when plugin validation is disabled via
        :func:`~ado.utilities.pydantic.ignore_plugin_validation`.

        Args:
            info: Pydantic validation info carrying optional context.

        Returns:
            self after validation and kind population.

        Raises:
            ValueError: If ``inputs`` is empty, an input name is undeclared, a
                kind mismatches its binding, a required input is missing, or a
                script input has a missing or wrong kind.
        """
        if ignore_plugin_validation(info):
            return self

        module = self.operation.module

        if isinstance(module, ScriptOperatorConf):
            # Script operators are always created via DiscoverySpace.operation_context,
            # which provides a fully-typed ADOResourceReference (kind=DISCOVERYSPACE).
            if not self.inputs:
                raise ValueError(
                    "ScriptOperatorConf operations require exactly one input "
                    "(the discovery space); 'inputs' is empty."
                )
            if len(self.inputs) > 1:
                raise ValueError(
                    "ScriptOperatorConf operations support exactly one input "
                    f"(the discovery space); found: {list(self.inputs)}."
                )
            for name, entry in self.inputs.items():
                if entry.kind is None:
                    raise ValueError(
                        f"Input '{name}' has no kind. ScriptOperatorConf "
                        "operations must be created via "
                        "DiscoverySpace.operation_context(), which always "
                        "provides an explicit kind."
                    )
                if entry.kind != CoreResourceKinds.DISCOVERYSPACE:
                    raise ValueError(
                        f"Input '{name}' has kind {entry.kind.value!r}; "
                        "ScriptOperatorConf only supports discoveryspace inputs."
                    )
            return self

        # Load operator required_resource_inputs (mirrors validate_and_downcast_parameters).
        if isinstance(module, OperatorModuleConf):
            import importlib

            operator_class = getattr(
                importlib.import_module(module.moduleName), module.moduleClass
            )
            operator_metadata = operator_class.operator_metadata()
        else:  # OperatorReference (already resolved by validate_and_downcast_parameters)
            from ado.modules.operators.collections import (
                operator_metadata_for_reference,
            )

            operator_metadata = operator_metadata_for_reference(module)

        # Use the default single-space input when the operator declares none
        # (should not happen for correctly registered operators).
        required_resource_inputs = operator_metadata.required_resource_inputs or (
            _DEFAULT_DISCOVERY_SPACE_INPUT_PROPERTY,
        )
        input_map = {d.identifier: d.kind for d in required_resource_inputs}

        if not self.inputs:
            required_names = [d.identifier for d in required_resource_inputs]
            raise ValueError(
                "Operation 'inputs' must not be empty. Required input(s): "
                f"{required_names}."
            )

        for name, entry in self.inputs.items():
            expected_kind = input_map.get(name)
            if expected_kind is None:
                raise ValueError(
                    f"Input '{name}' is not declared in the operator's "
                    "required_resource_inputs."
                )
            if entry.kind is None:
                entry.kind = expected_kind
            elif entry.kind != expected_kind:
                raise ValueError(
                    f"Input '{name}' declares kind {entry.kind.value!r} but "
                    f"the operator input expects {expected_kind.value!r}."
                )

        for d in required_resource_inputs:
            if d.identifier not in self.inputs:
                raise ValueError(
                    f"Required input '{d.identifier}' (kind={d.kind.value}) "
                    "is missing from the operation's 'inputs' field."
                )

        return self

    def get_actuatorconfigurations(
        self, project_context: ProjectContext
    ) -> list[ActuatorConfiguration]:
        """Gets the actuator configuration resources referenced by actuatorConfigurationIdentifiers from the metastore if any.

        Args:
            project_context: Information for connection to the metastore.

        Returns:
            A list of ActuatorConfiguration instances. The list will be empty if
            there are no actuatorConfigurationIdentifiers.

        Raises:
            ValueError: If more than one ActuatorConfigurationResource references
                the same actuator.
            ResourceDoesNotExistError: If any actuator configuration identifier
                cannot be found in the project.
        """
        import ado.metastore.sqlstore

        if not self.actuatorConfigurationIdentifiers:
            return []

        return get_actuator_configurations(
            actuator_configuration_identifiers=self.actuatorConfigurationIdentifiers,
            metastore=ado.metastore.sqlstore.SQLStore(project_context=project_context),
        )

    def validate_actuatorconfigurations(
        self, project_context: ProjectContext
    ) -> list[ActuatorConfiguration]:
        """Gets and validates the actuator configuration resources referenced by actuatorConfigurationIdentifiers from the metastore if any.

        Args:
            project_context: Information for connection to the metastore.

        Returns:
            A list of ActuatorConfiguration instances. The list will be empty if
            there are no actuatorConfigurationIdentifiers.

        Raises:
            ValueError: If more than one ActuatorConfigurationResource references
                the same actuator, or configurations are incompatible with the
                operation's spaces.
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
    projectContext: Annotated[
        ProjectContext | None,
        pydantic.Field(
            description=("Project this operation runs in."),
        ),
    ] = None
