# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import asyncio
import logging
import random
import uuid

import ray

import orchestrator.modules.actuators.catalog
import orchestrator.schema.property_value
from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.modules.actuators.base import ActuatorBase, DeprecatedExperimentError
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.observed_property import ObservedPropertyValue
from orchestrator.schema.property import AbstractPropertyDescriptor
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from orchestrator.schema.result import InvalidMeasurementResult, ValidMeasurementResult
from orchestrator.utilities.environment import enable_ray_actor_coverage
from orchestrator.utilities.logging import configure_logging

configure_logging()


async def mock_experiment_wait(
    request: MeasurementRequest, stateUpdateQueue: MeasurementQueue
) -> None:
    import functools

    import numpy as np

    await asyncio.sleep(np.random.default_rng().integers(1, 5))

    numberSuccessfulMeasurements = functools.reduce(
        lambda x, y: x + y,
        [isinstance(r, ValidMeasurementResult) for r in request.measurements],
    )

    request.status = (
        MeasurementRequestStateEnum.SUCCESS
        if numberSuccessfulMeasurements > 0
        else MeasurementRequestStateEnum.FAILED
    )

    stateUpdateQueue.put_nowait(request)


@ray.remote
class MockActuator(ActuatorBase):
    identifier = "mock"

    """A actuator class for testing

    Will make "random" measurements of any requested properties and submit them directly
    to StateUpdatesQueue"""

    def __init__(self, queue: MeasurementQueue, params: dict | None = None) -> None:

        enable_ray_actor_coverage("mock")
        super().__init__(queue=queue, params=params)
        self.log = logging.getLogger("mock-actuator")
        self.log.info(f"Queue is {self._stateUpdateQueue}")
        self._catalog = orchestrator.modules.actuators.catalog.ExperimentCatalog()
        self.running_tasks = set()

    def submit(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
    ) -> list[str]:

        self.log.info(
            f"Remote actuator submitting measurement of {[e.identifier for e in entities]} by {experimentReference}"
        )

        # Create a measurement request
        request = MeasurementRequest(
            operation_id=requesterid,
            requestIndex=requestIndex,
            experimentReference=experimentReference,
            entities=entities,
            requestid=str(uuid.uuid4())[:6],
        )

        self.log.debug(f"Created request {request}")
        if self._measurementSpace is None:
            raise AttributeError(
                "MockActuator requires a MeasurementSpace to execute experiment"
            )

        experiment = self._measurementSpace.experimentForReference(experimentReference)
        if experiment is None:
            raise ValueError(
                f"MockActuator unable to find {experimentReference} in the registered catalogs"
            )
        if experiment.deprecated:
            raise DeprecatedExperimentError(
                f"{experiment.actuatorIdentifier}.{experiment.identifier} is deprecated."
            )

        failRate = 5
        measurement_results = []
        for entity in entities:

            if random.randint(0, 100) < failRate:  # noqa: S311 - not crypto purposes
                measurement_result = InvalidMeasurementResult(
                    entityIdentifier=entity.identifier,
                    experimentReference=request.experimentReference,
                    reason="Simulated failure",
                )
            else:
                measurements = []
                for op in experiment.observedProperties:
                    self.log.debug(f"Creating mock measured value of {op} for {entity}")
                    # Create fake values for each property in the experiment
                    value = ObservedPropertyValue(
                        value=random.randint(  # noqa: S311 - not crypto purposes
                            0, 1000
                        ),
                        property=op,
                        valueType=orchestrator.schema.property_value.ValueTypeEnum.NUMERIC_VALUE_TYPE,
                    )
                    measurements.append(value)

                measurement_result = ValidMeasurementResult(
                    entityIdentifier=entity.identifier, measurements=measurements
                )

            measurement_results.append(measurement_result)

        if (
            len(
                [
                    valid_measurement
                    for valid_measurement in measurement_results
                    if isinstance(valid_measurement, ValidMeasurementResult)
                ]
            )
            > 0
        ):
            request.status = MeasurementRequestStateEnum.SUCCESS
        else:
            request.status = MeasurementRequestStateEnum.FAILED

        request.measurements = measurement_results
        task = asyncio.create_task(
            mock_experiment_wait(request, self._stateUpdateQueue)
        )
        self.running_tasks.add(task)
        task.add_done_callback(lambda t: self.running_tasks.remove(t))

        return [request.requestid]

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> orchestrator.modules.actuators.catalog.ExperimentCatalog:
        return orchestrator.modules.actuators.catalog.ExperimentCatalog(
            catalogIdentifier=cls.identifier,
            experiments={
                "test-experiment": Experiment(
                    identifier="test-experiment",
                    actuatorIdentifier="mock",
                    targetProperties=[AbstractPropertyDescriptor(identifier="score")],
                ),
                "test-experiment-two": Experiment(
                    identifier="test-experiment-two",
                    actuatorIdentifier="mock",
                    targetProperties=[AbstractPropertyDescriptor(identifier="score")],
                ),
            },
        )
