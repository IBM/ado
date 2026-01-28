# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import abc
import logging
import typing
from typing import Annotated

import pydantic

import orchestrator.modules.actuators.catalog
from orchestrator.core.actuatorconfiguration.config import (
    GenericActuatorParameters,
)
from orchestrator.modules.actuators.catalog import CatalogConfigurationRequirementEnum
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.module import (
    ModuleConf,
    ModuleTypeEnum,
)
from orchestrator.schema.entity import (
    Entity,
)
from orchestrator.schema.measurementspace import MeasurementSpace
from orchestrator.schema.reference import ExperimentReference

moduleLog = logging.getLogger("actuatorsbase")


class MeasurementError(Exception):
    """Raised when an error occurs while an actuator is measuring properties of entities."""


class DeprecatedExperimentError(Exception):
    """Raised when an actuator is attempting to run an experiment that has been deprecated."""


class MissingConfigurationForExperimentError(Exception):
    """Raised when an actuator is attempting to run an experiment but required configuration information is not present"""


# The actuator will have to
# - Take additional input files
# - Know how to map the entity/inputs to the measurement


class ActuatorBase(abc.ABC):
    """Base class for actuators defining their interface"""

    identifier: str
    parameters_class: type[GenericActuatorParameters] = GenericActuatorParameters

    def __init__(self, queue: MeasurementQueue, params: dict | None = None) -> None:
        """
        :param queue: A StateUpdateQueue the actuator can use to put results.
        :return: An ActuatorBase subclass
        """

        import os

        LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
        logging.basicConfig(level=LOGLEVEL)
        self.log = logging.getLogger("actuator")
        self._stateUpdateQueue = queue
        self._parameters = params if params is not None else {}
        self._measurementSpace = None  # type: typing.Optional[MeasurementSpace]

    def ready(self) -> bool:
        """This method is used to determine if the Actuator died on init"""
        return True

    @abc.abstractmethod
    def submit(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
    ) -> list[str]:
        """Submits the entities for measurement by experiment via the receiver

        :param entities: A list of Entity representing the entities to be measured
        :param experimentReference: An ExperimentReference defining the experiment to run on the entities
        :param requesterid: Unique identifier of the requester of this measurement
        :param requestIndex: The index of this request i.e. this is the ith request by the requester in the current run

        Returns:

            A list with the ids of the experiments it submitted.
            NOTE: An actuator may not submit all entities in one batch.

        Raises:
            DeprecatedExperimentError: if the experimentReference points to an experiment that is deprecated.
            UnknownExperimentError: if the experimentReference points to an experiment that is deprecated.
        """

    @classmethod
    @abc.abstractmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> orchestrator.modules.actuators.catalog.ExperimentCatalog:
        """Returns the Actuators ExperimentCatalog

        Keyword Parameters:
            actuator_configuration:
                If supplied the value of this parameter should be an instance of the Actuators configuration model

                Note: Subclasses do not have to support this parameter.

                You can determine if an Actuator subclass requires this parameter using the cls.catalog_requires_actuator_configuration method

        Exceptions:
            If cls.catalog_requires_actuator_configuration() returns REQUIRED this method should raise a ValueError if the
            keyword parameter actuator_configuration is not present; or if its value is not the correct model; or if
            the model does not contain the required information
        """

    @classmethod
    def catalog_requires_actuator_configuration(
        cls,
    ) -> CatalogConfigurationRequirementEnum:
        """Specifies if the catalog class method requires actuator configuration information

        Returns:
           - "CatalogConfigurationRequirementEnum.REQUIRED" if the catalog method requires an
             actuator configuration model of the correct type i.e. will raise an exception if it's not present.
           - "CatalogConfigurationRequirementEnum.OPTIONAL" if the catalog method can be passed an
             actuator configuration model of the correct type but will function correctly without it.
           - "CatalogConfigurationRequirementEnum.NOT_REQUIRED" (the default) if the catalog method does not require an
             actuator configuration model

        """

        return CatalogConfigurationRequirementEnum.NOT_REQUIRED

    def setMeasurementSpace(self, measurementSpace: MeasurementSpace) -> None:
        """Add a measurement space to the receiver to give it access to experiments beyond its catalog.

        It is Actuator implementation specific whether it uses the MeasurementSpace or not
        """

        self._measurementSpace = measurementSpace

    def default_parameters(
        self, is_template: bool = False
    ) -> GenericActuatorParameters:
        """
        Returns a default set of parameters for the actuator.

        Returns:
            An instance of orchestrator.model.config.GenericActuatorParameters
        """
        return (
            self.parameters_class.model_construct()
            if is_template
            else self.parameters_class()
        )

    def validate_parameters(
        self, parameters: GenericActuatorParameters
    ) -> GenericActuatorParameters:
        """
        Validates parameters provided by an actuator configuration.
        """
        return self.parameters_class.model_validate(parameters, from_attributes=True)


class ActuatorModuleConf(ModuleConf):
    moduleType: Annotated[ModuleTypeEnum, pydantic.Field()] = ModuleTypeEnum.ACTUATOR


if typing.TYPE_CHECKING:
    from ray.actor import ActorHandle

    ActuatorActor = type[ActorHandle[ActuatorBase]]
