# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging
import typing
import uuid

import orchestrator.modules.module
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
)
from orchestrator.schema.measurementspace import MeasurementSpace
from orchestrator.schema.reference import ExperimentReference
from orchestrator.utilities.logging import configure_logging

if typing.TYPE_CHECKING:
    import pandas as pd

    from orchestrator.schema.experiment import Experiment

configure_logging()

CATALOG_EXTENSIONS_CONFIGURATION_FILE_NAME = "custom_experiments.yaml"
moduleLogger = logging.getLogger("registry")


class UnknownExperimentError(Exception):
    pass


class UnknownActuatorError(Exception):
    """The actuator was never registered to the registry"""


class MissingActuatorConfigurationForCatalogError(Exception):
    """The actuator requires configuration information for it catalog, but it hasn't been provided"""


class UnexpectedCatalogRetrievalError(Exception):
    """The actuator catalog method raised on unexpected exception"""


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

        If the actuator has not been registered this method raises ActuatorNotFoundError

        If an actuator catalog requires configuration and this has not been provided
        then this method will raise a UnconfiguredActuatorCatalogError

        Any other exception while retrieving the catalog will raise UnexpectedCatalogRetrievalError
        """

        from orchestrator.modules.actuators.base import (
            CatalogConfigurationRequirementEnum,
        )

        actuator = self.actuatorForIdentifier(
            actuatorid=actuatorid
        )  # type: ActuatorBase

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
    ) -> "Experiment":
        """
        Returns the Experiment object corresponding to reference

        By default, searches all actuator catalogs

        Params:
            reference: A reference to an experiment (ExperimentReference)
            additionalCatalogs: Additional catalogs to search for the experiment
        Returns:
            The matching experiment
        Raises:
            Raises UnknownExperimentError if the experiment cannot be found in any catalog
            Raises UnknownActuatorError if the actuator cannot be found

        """

        log = logging.getLogger("registry")
        additionalCatalogs = (
            additionalCatalogs if additionalCatalogs is not None else []
        )

        # Get Catalog for Actuator
        experiment = None
        try:
            log.debug(
                f"Checking registry for the catalog of actuator {reference.actuatorIdentifier}"
            )
            catalog = self.catalogForActuatorIdentifier(
                actuatorid=reference.actuatorIdentifier
            )
            experiment = catalog.experimentForReference(reference)
            if experiment is not None:
                log.debug(f"Found {experiment}")
            else:
                log.debug(f"No experiment matching {reference} found")
        except KeyError:
            try:
                self.actuatorForIdentifier(reference.actuatorIdentifier)
            except UnknownActuatorError:
                log.warning(
                    f"No actuator registered called {reference.actuatorIdentifier}"
                )
            else:
                log.warning(
                    f"No catalog registered for actuator {reference.actuatorIdentifier}"
                )

        if experiment is None:
            for catalog in additionalCatalogs:
                log.debug(f"Checking external catalog {catalog} for {reference}")
                log.debug(f"Known experiments {catalog.experiments}")
                experiment = catalog.experimentForReference(reference)
                if experiment is not None:
                    log.debug(f"Found {experiment}")
                    break
                log.warning(f"No experiment matching {reference} found")

        if experiment is None:
            # AP: we haven't been able to find either the actuator
            #     or the experiment. We want to raise an accurate error
            if not self.actuatorForIdentifier(reference.actuatorIdentifier):
                raise UnknownActuatorError(reference.actuatorIdentifier)
            log.error(
                f"The {reference.actuatorIdentifier}  actuator was found but it did not contain "
                f"the {reference.experimentIdentifier} experiment."
            )
            raise UnknownExperimentError(reference)

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
        """Checks that all the actuators and experiments in measurement_space are in/available via the registry

        Returns:
            A list with an entry for each experiment that is not supported. Empty list means no issues
        """

        issues = []
        for experiment in measurement_space.experiments:
            try:
                self.experimentForReference(experiment.reference)
            except UnknownExperimentError as error:  # noqa: PERF203
                issues.append(f"UnknownExperimentError: {error!s}")
            except UnknownActuatorError as error:
                issues.append(f"UnknownActuatorError: {error!s}")
            except Exception as error:
                issues.append(str(error))

        return issues
