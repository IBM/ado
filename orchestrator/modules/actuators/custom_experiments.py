# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
import uuid

import pydantic
import ray

import orchestrator.modules.actuators.catalog
from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.modules.actuators.base import (
    ActuatorBase,
    ActuatorModuleConf,
    DeprecatedExperimentError,
)
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.module import load_module_class_or_function
from orchestrator.schema.entity import (
    CheckRequiredObservedPropertyValuesPresent,
    Entity,
)
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from orchestrator.schema.result import ValidMeasurementResult
from orchestrator.utilities.environment import enable_ray_actor_coverage
from orchestrator.utilities.logging import configure_logging

configure_logging()


async def custom_experiment_wrapper(
    function: typing.Callable,
    parameters: dict,
    measurementRequest: MeasurementRequest,
    targetExperiment: Experiment,
    queue: MeasurementQueue,
):
    """
    :param function: The function to call
    :param parameters: The custom parameters to the function
    :param measurementRequest: The entity and custom experiment to be measured
    :param targetExperiment: The experiment to execute.
        Required as the measurementRequest only includes an ExperimentReference
    :param queue: The queue to put the result on
    :return:
    """

    measurement_results = []
    for entity in measurementRequest.entities:
        values = function(entity, targetExperiment, parameters=parameters)

        # Record the results in the entity
        if len(values) > 0:
            measurement_result = ValidMeasurementResult(
                entityIdentifier=entity.identifier, measurements=values
            )
            measurement_results.append(measurement_result)

    if len(measurement_results) > 0:
        measurementRequest.measurements = measurement_results
        measurementRequest.status = MeasurementRequestStateEnum.SUCCESS
    else:
        measurementRequest.status = MeasurementRequestStateEnum.FAILED

    await queue.put_async(measurementRequest, block=False)


@ray.remote
class CustomExperiments(ActuatorBase):
    identifier = "custom_experiments"

    """Actuator for applying user supplied custom experiments
    """

    def __init__(self, queue, params: dict | None = None):
        """

        :param queue: The StateUpdates queue instance
        :param params: The params for the objective-function

        """

        enable_ray_actor_coverage("custom_experiments")
        super().__init__(queue=queue, params=params)

        params = params if params else {}
        self.log.debug(f"Queue is {self._stateUpdateQueue}")
        self.log.debug(f"Params are {params}")

        import orchestrator.modules.actuators.registry

        # Load custom_experiments catalog from registry
        registry = (
            orchestrator.modules.actuators.registry.ActuatorRegistry.globalRegistry()
        )
        self._catalog = registry.catalogForActuatorIdentifier(self.__class__.identifier)

        self.log.debug(f"Catalog is {self._catalog}")

        self._functionImplementations = {}
        for experiment in self._catalog.experiments:
            # We cannot do a check of the dependencies here because
            # (a) Catalog addition to registry is non-deterministic so any dependent experiment
            # may not be in the registry yet

            # Store function name in the experiments metadata
            try:
                experiment_module_conf = ActuatorModuleConf.model_validate(
                    experiment.metadata.get("module", None)
                )
            except pydantic.ValidationError:
                self.log.exception(
                    f"Experiment {experiment} did not provide a valid module configuration - skipping"
                )
            else:
                self._functionImplementations[experiment.identifier] = (
                    load_module_class_or_function(experiment_module_conf)
                )

                self.log.info(
                    f"Experiment name: {experiment.identifier}. "
                    f"Function Name: {experiment_module_conf.moduleFunction}. "
                    f"Function Implementation: {self._functionImplementations[experiment.identifier]}. "
                    f"Experiment: {experiment}"
                )

        self.log.debug("Completed init")

    def loadedExperiment(
        self,
        experimentReference: ExperimentReference,
    ):

        return (
            self._functionImplementations.get(experimentReference.experimentIdentifier)
            is not None
        )

    async def submit(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
    ):

        self.log.debug(
            f"Received a request to measure {experimentReference} on {[e.identifier for e in entities]}"
        )

        if self._catalog.experimentForReference(experimentReference) is None:
            if self._catalog.experiments:
                raise ValueError(
                    f"Requested experiments {experimentReference} is not in the CustomExperiments actuator catalog. "
                    f"Known experiments are {list(self._catalog.experimentsMap.keys())}"
                )
            raise ValueError(
                f"Requested experiments {experimentReference} is not in the CustomExperiments actuator catalog (which is empty). "
            )

        targetExperiment = self._catalog.experimentForReference(experimentReference)
        if targetExperiment.deprecated:
            raise DeprecatedExperimentError(
                f"{targetExperiment.actuatorIdentifier}.{targetExperiment.identifier} is deprecated."
            )

        # Check all required property values are present to actuate on
        for entity in entities:
            if not CheckRequiredObservedPropertyValuesPresent(
                entity, targetExperiment, exactMatch=False
            ):
                raise ValueError(
                    f"Entity {entity.identifier} does not have values for properties required "
                    f"as inputs for experiment {experimentReference.experimentIdentifier}"
                )

        # Create Measurement Request
        requestid = str(uuid.uuid4())[:6]
        request = MeasurementRequest(
            operation_id=requesterid,
            requestIndex=requestIndex,
            experimentReference=experimentReference,
            entities=entities,
            requestid=requestid,
        )

        self.log.debug(f"Create measurement request {request}")
        # TODO: Allow functions to specify if they should be remote
        experiment = self._catalog.experimentForReference(request.experimentReference)
        function = experiment.metadata.get("function", experiment.identifier)
        self.log.debug(f"Calling custom experiment {function}")

        await custom_experiment_wrapper(
            self._functionImplementations[
                request.experimentReference.experimentIdentifier
            ],
            self._catalog.experimentForReference(
                request.experimentReference
            ).metadata.get("parameters", {}),
            request,  # The request - contains experiment reference
            targetExperiment,  # Experiment to execute
            self._stateUpdateQueue,
        )

        # We only send one request
        return [requestid]

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> orchestrator.modules.actuators.catalog.ExperimentCatalog:
        return orchestrator.modules.actuators.catalog.ExperimentCatalog(
            catalogIdentifier="CustomExperiments"
        )

    def current_catalog(
        self,
    ) -> orchestrator.modules.actuators.catalog.ExperimentCatalog:
        return self._catalog
