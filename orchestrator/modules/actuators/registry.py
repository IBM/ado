# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging
import typing
import uuid

import orchestrator.schema
from orchestrator.core.actuatorconfiguration.config import (
    GenericActuatorParameters,
)
from orchestrator.core.metadata import PackageProvenance
from orchestrator.modules.actuators.base import (
    ActuatorBase,
)
from orchestrator.modules.actuators.catalog import (
    ExperimentCatalog,
    ExperimentReferenceMatchMode,
)
from orchestrator.modules.actuators.errors import (
    AmbiguousExperimentIdentifierError,
    DeprecatedExperimentError,
    ExperimentVersionMismatchError,
    MissingActuatorConfigurationForCatalogError,
    UnexpectedCatalogRetrievalError,
    UnknownActuatorError,
    UnknownExperimentError,
)
from orchestrator.schema.experiment import (
    Experiment,
    ExperimentInterfaceIssue,
    ExperimentInterfaceIssueKind,
    ParameterizedExperiment,
)
from orchestrator.schema.measurementspace import MeasurementSpace
from orchestrator.schema.reference import ExperimentReference
from orchestrator.utilities.logging import configure_logging

if typing.TYPE_CHECKING:
    import pandas as pd

configure_logging()

CATALOG_EXTENSIONS_CONFIGURATION_FILE_NAME = "custom_experiments.yaml"
moduleLogger = logging.getLogger("registry")

_MEASUREMENT_SPACE_INTERFACE_ISSUE_TEMPLATES: dict[
    ExperimentInterfaceIssueKind, str
] = {
    ExperimentInterfaceIssueKind.MISSING_REQUIRED_CONSTITUTIVE_IN_PROVIDED: (
        "measurement-space experiment requires constitutive input "
        "{identifier!r} that is not declared in the actuator catalog"
    ),
    ExperimentInterfaceIssueKind.EXTRA_REQUIRED_CONSTITUTIVE_IN_PROVIDED: (
        "actuator catalog experiment requires constitutive input "
        "{identifier!r} that is not required in the measurement-space "
        "experiment"
    ),
    ExperimentInterfaceIssueKind.MISSING_REQUIRED_OBSERVED_IN_PROVIDED: (
        "measurement-space experiment requires observed input "
        "{identifier!r} that is not declared in the actuator catalog"
    ),
    ExperimentInterfaceIssueKind.EXTRA_REQUIRED_OBSERVED_IN_PROVIDED: (
        "actuator catalog experiment requires observed input "
        "{identifier!r} that is not required in the measurement-space "
        "experiment"
    ),
    ExperimentInterfaceIssueKind.PARAMETERIZED_OPTIONAL_NOT_IN_PROVIDED: (
        "measurement-space experiment parameterizes optional input "
        "{identifier!r} that is not optional in the actuator catalog"
    ),
    ExperimentInterfaceIssueKind.OPTIONAL_NOT_DECLARED_IN_PROVIDED: (
        "measurement-space experiment declares optional input "
        "{identifier!r} that is not declared in the actuator catalog"
    ),
    ExperimentInterfaceIssueKind.DOMAIN_NOT_COMPATIBLE: (
        "domain for {identifier!r} in the measurement-space experiment is "
        "not compatible with the actuator catalog"
    ),
    ExperimentInterfaceIssueKind.PARAMETERIZED_VALUE_OUT_OF_DOMAIN: (
        "parameterized value {value!r} for {identifier!r} in the "
        "measurement-space experiment is not in the actuator catalog property domain"
    ),
    ExperimentInterfaceIssueKind.OPTIONAL_DEFAULT_MISMATCH: (
        "default value for optional input {identifier!r} is "
        "{expectedDefault!r} in the measurement-space experiment but "
        "{providedDefault!r} in the actuator catalog"
    ),
    ExperimentInterfaceIssueKind.OUTPUT_NOT_IN_PROVIDED: (
        "output {identifier!r} declared in the measurement-space experiment "
        "is not produced by the actuator catalog experiment"
    ),
}


def format_measurement_space_interface_issue(
    expected_experiment: Experiment,
    issue: ExperimentInterfaceIssue,
) -> str:
    """Format a structured interface issue for measurement-space support checks.

    Args:
        expected_experiment: The measurement-space experiment being validated.
        issue: Structured interface mismatch returned by compatibility checking.

    Returns:
        A user-facing issue string for logging or error reporting.
    """
    template = _MEASUREMENT_SPACE_INTERFACE_ISSUE_TEMPLATES[issue.kind]
    detail = template.format(**issue.model_dump(exclude={"kind"}, exclude_none=True))
    return (
        f"ExperimentInterfaceMismatchError: "
        f"{expected_experiment.actuatorIdentifier}.{expected_experiment}: {detail}"
    )


class ActuatorRegistry:
    gRegistry = None

    """Provides access to actuators and the experiments they can execute"""

    @classmethod
    def globalRegistry(cls) -> "ActuatorRegistry":

        if ActuatorRegistry.gRegistry is not None:
            moduleLogger.debug("Global registry exists - using")
            return ActuatorRegistry.gRegistry

        moduleLogger.debug("No  global registry - creating one")
        ActuatorRegistry.gRegistry = ActuatorRegistry()
        moduleLogger.debug(f"Created global registry {ActuatorRegistry.gRegistry}")
        return ActuatorRegistry.gRegistry

    def __init__(
        self,
        actuator_configurations: dict[str, GenericActuatorParameters] | None = None,
    ) -> None:
        """Detects and loads Actuator plugins"""

        import importlib.metadata
        import inspect
        import pkgutil

        import orchestrator.modules.actuators as builtin_actuators
        from orchestrator.modules.actuators.base import ActuatorBase

        # Maps actuator ids to generic actuator parameter payloads from configuration.
        self.actuatorConfigurationMap: dict[str, GenericActuatorParameters] = {}
        if actuator_configurations:
            self.actuatorConfigurationMap.update(actuator_configurations)

        # Maps actuator ids to ActuatorBase instances
        self.actuatorIdentifierMap: dict[str, type[ActuatorBase]] = {}
        # Maps actuator ids to ExperimentCatalog instances
        self.catalogIdentifierMap: dict[str, ExperimentCatalog] = {}
        # Maps actuator ids to metadata (version, description, and distributionName)
        self.actuatorMetadataMap: dict[str, dict[str, str | None]] = {}
        self.log = logging.getLogger("registry")
        self.id = uuid.uuid4()

        # Get ado-core version once for all builtin actuators
        self._ado_core_version = importlib.metadata.version("ado-core")

        # We handle builtin actuators
        for module in pkgutil.iter_modules(
            builtin_actuators.__path__, f"{builtin_actuators.__name__}."
        ):
            for _name, member in inspect.getmembers(
                importlib.import_module(module.name)
            ):
                # Builtin actuators are undecorated; operators/setup.py applies
                # ray.remote when instantiating them.
                actuator_class = None
                if (
                    isinstance(member, type)
                    and member is not ActuatorBase
                    and issubclass(member, ActuatorBase)
                ):
                    actuator_class = member

                if actuator_class:
                    self.registerActuator(
                        actuator_class.identifier, actuator_class, is_builtin=True
                    )

        # Load actuator plugins via entry points
        for actuator_entry_point in importlib.metadata.entry_points(
            group="ado.actuators"
        ):
            try:
                actuator_class = actuator_entry_point.load()
                if not (
                    isinstance(actuator_class, type)
                    and issubclass(actuator_class, ActuatorBase)
                ):
                    self.log.error(
                        f"Entry point {actuator_entry_point.name} does not point to an ActuatorBase subclass"
                    )
                    continue
                self.registerActuator(
                    actuatorid=actuator_class.identifier,
                    actuatorClass=actuator_class,
                    is_builtin=False,
                )
                self.log.debug(
                    f"Loaded actuator plugin {actuator_entry_point.name} from entry point: {actuator_entry_point.value}"
                )
            except Exception as e:
                self.log.error(
                    f"Failed to load actuator entry point {actuator_entry_point.name}: {e}"
                )

    def __str__(self) -> str:

        return f"Registry id {self.id}"

    def set_actuator_configurations_for_catalogs(
        self, configurations: dict[str, GenericActuatorParameters]
    ) -> None:
        """Supply information for catalogs that require configuration

        If a configuration has already been supplied for an actuator it is not updated - you will need to create a
        new registry instance.
        """

        self.actuatorConfigurationMap.update(
            {
                k: v
                for k, v in configurations.items()
                if k not in self.actuatorConfigurationMap
            }
        )

    def _get_builtin_actuator_metadata(
        self, actuator_class: "type[ActuatorBase]"
    ) -> dict[str, str | None]:
        """Extract metadata for builtin actuators.

        Args:
            actuator_class: The actuator class

        Returns:
            Dictionary with 'version' and 'description' keys
        """
        version = self._ado_core_version

        # Get first line of docstring as description if available
        description = None
        try:
            if actuator_class.__doc__:
                description = actuator_class.__doc__.strip().split("\n")[0]
        except (AttributeError, IndexError):
            pass

        return {
            "version": version,
            "description": description,
            "distributionName": "ado-core",
        }

    def _get_plugin_actuator_metadata(
        self, actuator_class: "type[ActuatorBase]"
    ) -> dict[str, str | None]:
        """Extract metadata for plugin actuators.

        Args:
            actuator_class: The actuator class

        Returns:
            Dictionary with 'version' and 'description' keys
        """
        from orchestrator.core.metadata import PackageProvenance

        description = None
        provenance = PackageProvenance.from_module_name(actuator_class.__module__)
        if provenance is not None:
            try:
                import importlib.metadata

                dist = importlib.metadata.distribution(provenance.distributionName)
                description = dist.metadata.get("Summary", None)
            except Exception as e:
                self.log.debug(
                    f"Could not extract description for plugin actuator "
                    f"{actuator_class}: {e}"
                )
            return {
                "version": provenance.distributionVersion,
                "description": description,
                "distributionName": provenance.distributionName,
            }

        return {
            "version": None,
            "description": None,
            "distributionName": None,
        }

    def registerActuator(
        self,
        actuatorid: str,
        actuatorClass: "type[ActuatorBase]",
        is_builtin: bool = False,
    ) -> None:
        """Adds an actuator and a catalog of experiments it can execute to the registry

        Note: Currently each actuator can only have one catalog although further experiments can be added to it

        Parameters:
            actuatorid: The id of this actuator. This id is how consumers will access it
            actuatorClass: The class that implements the actuator.
            is_builtin: Whether this is a builtin actuator (from orchestrator.modules.actuators)
        """

        if self.actuatorIdentifierMap.get(actuatorid) is None:
            self.actuatorIdentifierMap[actuatorid] = actuatorClass

            # Extract and store metadata
            if is_builtin:
                metadata = self._get_builtin_actuator_metadata(actuatorClass)
            else:
                metadata = self._get_plugin_actuator_metadata(actuatorClass)

            self.actuatorMetadataMap[actuatorid] = metadata

    def provenance_for_actuator(self, identifier: str) -> PackageProvenance | None:
        """Return the package provenance for a registered actuator.

        Returns ``None`` if the actuator is not registered or its distribution
        could not be resolved (e.g. the actuator was loaded from an unpackaged
        local path).

        Args:
            identifier: The actuator identifier.

        Returns:
            A :class:`~orchestrator.core.metadata.PackageProvenance` instance,
            or ``None`` if provenance is unavailable.
        """
        metadata = self.actuatorMetadataMap.get(identifier)
        if metadata is None:
            return None
        dist_name = metadata.get("distributionName")
        version = metadata.get("version")
        if dist_name is None or version is None:
            return None
        return PackageProvenance(
            distributionName=dist_name, distributionVersion=version
        )

    def catalogForActuatorIdentifier(self, actuatorid: str) -> ExperimentCatalog:
        """Returns the catalog for a given actuator via its identifier

        If the actuator has not been registered this method raises UnknownActuatorError

        If an actuator catalog requires configuration and this has not been provided
        then this method will raise a MissingActuatorConfigurationForCatalogError

        Any other exception while retrieving the catalog will raise UnexpectedCatalogRetrievalError
        """

        from orchestrator.modules.actuators.base import (
            CatalogConfigurationRequirementEnum,
        )

        actuator = self.actuatorForIdentifier(
            actuatorid=actuatorid
        )  # type: type[ActuatorBase]

        cfg = None
        try:
            catalog = self.catalogIdentifierMap[actuatorid]
        except KeyError as error:
            # Load catalog on demand
            # Get configuration if any registered
            cfg = self.catalogIdentifierMap.get(actuatorid)
            # Check if configuration is required and then raise error if it is and there is none
            if (
                actuator.catalog_requires_actuator_configuration()
                == CatalogConfigurationRequirementEnum.REQUIRED
            ) and not cfg:
                raise MissingActuatorConfigurationForCatalogError(
                    f"Actuator {actuatorid} requires configuration information to create catalog."
                ) from error

            # If the catalog config is not required we can continue if cfg is None or a configuration instance
            if (
                actuator.catalog_requires_actuator_configuration()
                in [
                    CatalogConfigurationRequirementEnum.REQUIRED,
                    CatalogConfigurationRequirementEnum.OPTIONAL,
                ]
                and cfg
            ):
                try:
                    catalog = actuator.catalog(actuator_configuration=cfg)
                except Exception as error:
                    self.log.warning(
                        f"Unexpected exception, '{error}', retrieving catalog of actuator {actuatorid} using configuration {cfg}"
                    )
                    raise UnexpectedCatalogRetrievalError(
                        f"Unexpected exception, '{error}', retrieving catalog of actuator {actuatorid} using configuration {cfg}"
                    ) from error
                else:
                    self.catalogIdentifierMap[actuatorid] = catalog
                    self.log.debug(
                        f"Loaded catalog {catalog} for actuator with id {actuatorid} to {self} on-demand"
                    )
            else:
                try:
                    catalog = actuator.catalog()
                except Exception as error:
                    self.log.warning(
                        f"Unexpected exception retrieving catalog of actuator {actuatorid} using configuration {cfg}"
                    )
                    raise UnexpectedCatalogRetrievalError(
                        f"Unexpected exception {error} retrieving catalog of actuator {actuatorid} using configuration {cfg}"
                    ) from error
                else:
                    self.catalogIdentifierMap[actuatorid] = catalog
                    self.log.debug(
                        f"On-demand loaded catalog {catalog} for actuator with id {actuatorid} to {self}"
                    )

        return catalog

    def actuatorForIdentifier(self, actuatorid: str) -> type[ActuatorBase]:
        """Returns the actuator class corresponding to an identifier

        If the actuator has not been registered this method raises UnknownActuatorError
        """

        try:
            actuator_class = self.actuatorIdentifierMap[actuatorid]
        except KeyError as error:
            raise UnknownActuatorError(
                f"No actuator called {actuatorid} has been added to the registry"
            ) from error

        return actuator_class

    def experimentForReference(
        self,
        reference: ExperimentReference,
        additionalCatalogs: list[ExperimentCatalog] | None = None,
        *,
        match_on: ExperimentReferenceMatchMode = "major_version",
        resolve: bool = False,
    ) -> Experiment | ParameterizedExperiment:
        """Return the experiment corresponding to reference.

        Searches the actuator's catalog and any additional catalogs. When
        ``resolve=True``, applies strict version matching, rejects deprecated
        experiments, and wraps parameterization. The registry always raises on
        miss; it never returns ``None``.

        Args:
            reference: A reference to an experiment.
            additionalCatalogs: Additional catalogs to search for the experiment.
            match_on: ``"major_version"`` (default), ``"fully_qualified_version"``,
                ``"base"``, or ``"any"``. See
                :meth:`~orchestrator.modules.actuators.catalog.ExperimentCatalog.experimentForReference`.
            resolve: When ``True``, apply version checks, deprecated checks, and
                parameterization.

        Returns:
            The matching experiment or parameterized experiment.

        Raises:
            UnknownExperimentError: If the experiment cannot be found in any catalog.
            UnknownActuatorError: If the actuator cannot be found.
            AmbiguousExperimentIdentifierError: When ``match_on='base'`` or
                ``match_on='any'`` finds multiple catalog versions for the same
                base identifier.
            ExperimentVersionMismatchError: When ``resolve=True`` and
                ``match_on='fully_qualified_version'`` with a version mismatch.
            DeprecatedExperimentError: When ``resolve=True`` and the experiment
                is deprecated.
            UnexpectedCatalogRetrievalError: If the actuators catalog cannot be
            retrieved
        """

        log = logging.getLogger("registry")
        additionalCatalogs = (
            additionalCatalogs if additionalCatalogs is not None else []
        )

        catalogs_to_try: list[ExperimentCatalog] = []
        actuator_catalog: ExperimentCatalog | None = None

        try:
            log.debug(
                f"Checking registry for the catalog of actuator {reference.actuatorIdentifier}"
            )
            # Either raises or returns non None
            actuator_catalog = self.catalogForActuatorIdentifier(
                actuatorid=reference.actuatorIdentifier
            )
            catalogs_to_try.append(actuator_catalog)
        except UnknownActuatorError:
            log.warning(f"No actuator registered called {reference.actuatorIdentifier}")
            raise
        except UnexpectedCatalogRetrievalError:
            # We continue as their may be additional catalogs
            log.warning(
                f"Unable to retrieve the catalog for {reference.actuatorIdentifier}"
            )
        except MissingActuatorConfigurationForCatalogError:
            # We continue as there may be additional catalogs
            log.warning(
                f"The catalog for {reference.actuatorIdentifier} requires configuration but this has not been supplied"
            )

        catalogs_to_try.extend(additionalCatalogs)

        if not catalogs_to_try:
            raise UnexpectedCatalogRetrievalError(
                f"No catalogs available for {reference.actuatorIdentifier}"
            )

        # Now try to find the experiment
        experiment = None
        for catalog in catalogs_to_try:
            log.debug(
                f"Looking up {reference} from catalog {catalog} with match_on={match_on}, resolve={resolve}"
            )
            try:
                experiment = catalog.experimentForReference(
                    reference, match_on=match_on, resolve=resolve
                )
            except ExperimentVersionMismatchError:
                raise
            except AmbiguousExperimentIdentifierError:
                raise
            except UnknownExperimentError:
                log.debug(f"No experiment matching {reference} found in {catalog}")
                continue
            except DeprecatedExperimentError:
                raise
            else:
                if experiment is not None:
                    log.debug(f"Found {experiment}")
                    break

        if not experiment:
            if actuator_catalog is not None:
                message = (
                    f"The {reference.actuatorIdentifier} actuator was found but a match to "
                    f"{reference} was not found using mode {match_on}."
                )
                candidates = actuator_catalog.experiments_matching_identifier(reference)
                if candidates:
                    available_versions = ", ".join(
                        sorted({e.version for e in candidates if e.version})
                    )
                    message = f"{message} Available versions in catalog: {available_versions}."
            else:
                message = (
                    f"No match for {reference} was found in the available catalogs "
                    f"using mode {match_on}."
                )

            raise UnknownExperimentError(message)

        return experiment

    @property
    def catalogs(self) -> list[ExperimentCatalog]:
        """Returns an iterator over the catalogs of the registered actuators

        If a catalog requires configuration and this has not been supplied it will be skipped.
        If there UnexpectedCatalogRetrievalError this is also skipped
        """

        # Since catalogs may be loaded on demand we cannot go to "catalogIdentifierMap" directly
        catalogs = []
        for actuatorid in self.actuatorIdentifierMap:
            try:
                catalog = self.catalogForActuatorIdentifier(actuatorid=actuatorid)
            except (  # noqa: PERF203
                MissingActuatorConfigurationForCatalogError,
                UnexpectedCatalogRetrievalError,
            ):
                pass
            else:
                catalogs.append(catalog)

        return catalogs

    @property
    def experiments(self) -> "pd.DataFrame":
        """Returns a dataframe of the experiments in the receiver"""

        import pandas as pd

        data = []
        for actuatorid in self.actuatorIdentifierMap:
            try:
                catalog = self.catalogForActuatorIdentifier(actuatorid=actuatorid)
            except MissingActuatorConfigurationForCatalogError:  # noqa: PERF203
                self.log.warning(
                    f"Cannot retrieve experiments from actuator {actuatorid} as it requires configuration information for its catalog and this has not been provided"
                )
            else:
                rows = [
                    [catalog.identifier, f"{e.actuatorIdentifier}.{e.identifier}"]
                    for e in catalog.experiments
                ]
                data.extend(rows)

        return pd.DataFrame(data=data, columns=["catalog", "experiment reference"])

    def updateCatalogs(
        self,
        catalogExtension: orchestrator.modules.actuators.catalog.ActuatorCatalogExtension,
    ) -> None:
        """Updates the receivers catalogs with the experiments in catalogExtension

        Its expected that catalogExtension will only contain experiments for a single actuator, but it is not enforced

        If there is no matching actuator for an experiment(s) this method raises UnknownActuatorError
        In this case no changes will be made to any catalogs"""

        unknownActuators = []
        for experiment in catalogExtension.experiments:
            try:
                self.catalogForActuatorIdentifier(experiment.actuatorIdentifier)
            except UnknownActuatorError:  # noqa: PERF203
                unknownActuators.append(experiment.actuatorIdentifier)

        if len(unknownActuators) > 0:
            raise UnknownActuatorError(
                f"Failed to update catalogs with {catalogExtension}. Unknown actuators: {unknownActuators}"
            )
        for experiment in catalogExtension.experiments:
            catalog = self.catalogForActuatorIdentifier(experiment.actuatorIdentifier)
            catalog.addExperiment(experiment)

    def checkMeasurementSpaceSupported(
        self, measurement_space: MeasurementSpace
    ) -> list:
        """Check that all actuators and experiments in *measurement_space* are available.

        Uses :meth:`~orchestrator.modules.actuators.catalog.ExperimentCatalog.experimentForReference`
        with ``resolve=True`` so that major version mismatches are detected and reported.
        When lookup succeeds, also compares the measurement-space experiment interface
        against the registry catalog experiment (inputs, domains, optional defaults,
        and outputs). All interface mismatches for an experiment are collected.

        Returns:
            A list of issue strings. An empty list means no issues were found.
        """
        from orchestrator.modules.actuators.errors import DeprecatedExperimentError
        from orchestrator.schema.experiment import check_experiment_interface_compatible

        issues = []
        for experiment in measurement_space.experiments:
            ref = experiment.reference
            try:
                catalog = self.catalogForActuatorIdentifier(ref.actuatorIdentifier)
                provided_experiment = catalog.experimentForReference(ref, resolve=True)
                if provided_experiment is not None:
                    interface_issues = check_experiment_interface_compatible(
                        expected_experiment=experiment,
                        provided_experiment=provided_experiment,
                    )
                    issues.extend(
                        format_measurement_space_interface_issue(experiment, issue)
                        for issue in interface_issues
                    )
            except ExperimentVersionMismatchError as error:  # noqa: PERF203
                issues.append(f"ExperimentVersionMismatchError: {error!s}")
            except UnknownExperimentError as error:
                issues.append(f"UnknownExperimentError: {error!s}")
            except UnknownActuatorError as error:
                issues.append(f"UnknownActuatorError: {error!s}")
            except DeprecatedExperimentError as error:
                issues.append(f"DeprecatedExperimentError: {error!s}")
            except Exception as error:
                issues.append(str(error))

        return issues
