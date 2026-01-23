# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import typing
import uuid

import yaml

import orchestrator.modules.module
import orchestrator.schema
from orchestrator.core.actuatorconfiguration.config import (
    GenericActuatorParameters,
)
from orchestrator.modules.actuators.catalog import (
    ExperimentCatalog,
)
from orchestrator.schema.measurementspace import MeasurementSpace
from orchestrator.schema.reference import ExperimentReference
from orchestrator.utilities.logging import configure_logging

if typing.TYPE_CHECKING:
    import pandas as pd

    from orchestrator.modules.actuators.base import (
        ActuatorBase,
    )
    from orchestrator.schema.experiment import Experiment

configure_logging()

ACTUATOR_CONFIGURATION_FILE_NAME = "actuator_definitions.yaml"
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

        # Mpass actuator ids to actuator configurations: G
        self.actuatorConfigurationMap = (
            {}
        )  # type: typing.Dict[typing.AnyStr, "orchestrator.model.config.GenericActuatorParameters"]
        if actuator_configurations:
            self.actuatorConfigurationMap.update(actuator_configurations)

        # Maps actuator ids to ActuatorBase instances
        self.actuatorIdentifierMap = (
            {}
        )  # type: typing.Dict[typing.AnyStr, "ActuatorBase"]
        # Maps actuator ids to ExperimentCatalog instances
        self.catalogIdentifierMap = (
            {}
        )  # type: typing.Dict[typing.AnyStr, ExperimentCatalog]
        self.log = logging.getLogger("registry")
        self.id = uuid.uuid4()

        # We handle builtin actuators
        import importlib.resources
        import inspect
        import pkgutil

        import orchestrator.modules.actuators as builtin_actuators
        from orchestrator.modules.actuators.base import ActuatorBase, ActuatorModuleConf

        for module in pkgutil.iter_modules(
            builtin_actuators.__path__, f"{builtin_actuators.__name__}."
        ):
            for _name, member in inspect.getmembers(
                importlib.import_module(module.name)
            ):
                # MJ: The Actuator classes are decorated ray.remote
                # This means the member mymodule.myactuatorclass will be an instance of ray "ActorClass(MyActuatorClass)" and not the class!
                #
                # Ray has added code so ActuatorBase.__subclasscheck__(ActorClass(MyActuatorClass))" returns True
                # i.e. it identifies that the ray "wrapped" subclass is a subclass
                #
                # This finally means isinstance(mymodule.myactuatorclass, ActuatorBase) works although unexpectedly as you might expect the first arg to be a class not an instance
                # Why? mymodule.myactorclass -> is an instance of ActorClass(MyActuatorClass) -> the class of this is  ActorClass(MyActuatorClass) -> this evaluates as subclass of ActuatorBase

                # It's slightly clearer to use issubclass, as this is what you want to know, but correct for the fact that
                # when "member" is an ActuatorBase subclass it will be decorated with a ray object, and we need to use __class__

                if issubclass(member.__class__, ActuatorBase):
                    self.registerActuator(member.identifier, member)

        try:
            import ado_actuators as plugins
        except ImportError:
            return

        from pathlib import Path

        import pydantic

        ActuatorFileModel = pydantic.RootModel[list[ActuatorModuleConf]]

        self.log.debug(f"{plugins.__path__}, {plugins.__name__}")

        # This adds the plugins to the ActuatorRegistry
        for module in pkgutil.iter_modules(plugins.__path__, f"{plugins.__name__}."):
            module_contents = {
                entry.name for entry in importlib.resources.files(module.name).iterdir()
            }
            self.log.debug(
                f"Checking if module {module.name} is an actuator plugin. Contents: {module_contents}"
            )
            if ACTUATOR_CONFIGURATION_FILE_NAME in module_contents:
                self.log.debug(f"Found {ACTUATOR_CONFIGURATION_FILE_NAME}")

                actuator_configuration_file = Path(
                    str(importlib.resources.files(module.name))
                ) / Path(ACTUATOR_CONFIGURATION_FILE_NAME)

                try:
                    actuators = ActuatorFileModel(
                        yaml.safe_load(actuator_configuration_file.read_text())
                    ).root
                except pydantic.ValidationError:
                    self.log.exception(
                        f"{module.name}'s {ACTUATOR_CONFIGURATION_FILE_NAME} raised a validation error"
                    )
                    raise

                for actuator in actuators:

                    # AP 02/09/2024
                    # While this is not strictly needed anymore, we keep it
                    # to validate that all the requirements for the actuator
                    # are met. If this wasn't the case, we would get an error
                    # when importing orchestrator.actuators or calling the
                    # load_module_class method
                    try:
                        actuator_class = (
                            orchestrator.modules.module.load_module_class_or_function(
                                actuator
                            )
                        )
                    except ModuleNotFoundError as e:
                        self.log.exception(
                            f"Skipping actuator {actuator.moduleName}: an exception was raised indicating "
                            "unmet requirements. Please ensure all the actuators' requirements are installed.\n"
                            f"Exception was:\n{e}"
                        )
                        continue
                    except ImportError as e:
                        self.log.exception(
                            f"Skipping actuator {actuator.moduleName} because of an exception while importing it.\n"
                            f"Exception was:\n{e}"
                        )
                        continue
                    except AttributeError as e:
                        self.log.exception(
                            f"Skipping actuator {actuator.moduleName} because we could not find the actuator class {actuator.moduleClass} in it.\n"
                            f"Exception was:\n{e}"
                        )
                        continue

                    # AP: we are initialising the ActuatorRegistry
                    # we do not need to check whether we have already
                    # registered the actuator
                    self.log.debug(f"Add actuator plugin {actuator}")
                    self.registerActuator(
                        actuatorid=actuator_class.identifier,
                        actuatorClass=actuator_class,
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

    def registerActuator(
        self,
        actuatorid: str,
        actuatorClass: "type[ActuatorBase]",
    ) -> None:
        """Adds an actuator and a catalog of experiments it can execute to the registry

        Note: Currently each actuator can only have one catalog although further experiments can be added to it

        Parameters:
            actuatorid: The id of this actuator. This id is how consumers will access it
            actuatorClass: The class that implements the actuator.
                Note: Since these are decorated with "ray.remote" they will actually be instances of ray.actor.ActorClass
        """

        if self.actuatorIdentifierMap.get(actuatorid) is None:
            self.actuatorIdentifierMap[actuatorid] = actuatorClass

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

    def actuatorForIdentifier(
        self, actuatorid: str
    ) -> "orchestrator.modules.actuators.base.ActuatorBase":
        """Returns the actuator class corresponding to an identifier

        If the actuator has not been registered this method raises UnknownActuatorError
        """

        try:
            acuatorClass = self.actuatorIdentifierMap[actuatorid]
        except KeyError as error:
            raise UnknownActuatorError(
                f"No actuator called {actuatorid} has been added to the registry"
            ) from error

        return acuatorClass

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
