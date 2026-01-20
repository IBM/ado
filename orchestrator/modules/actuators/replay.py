# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import uuid

import ray

import orchestrator.modules.actuators.catalog
from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.modules.actuators.base import ActuatorBase
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.entity import Entity
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import (
    MeasurementRequest,
    MeasurementRequestStateEnum,
    ReplayedMeasurement,
)
from orchestrator.schema.result import InvalidMeasurementResult
from orchestrator.utilities.environment import enable_ray_actor_coverage
from orchestrator.utilities.logging import configure_logging

configure_logging()
moduleLog = logging.getLogger("replay")


def replay(
    requestIndex: int,
    requesterid: str,
    entities: list[Entity],
    experiment_reference: ExperimentReference,
    measurement_queue: MeasurementQueue,
) -> list[ReplayedMeasurement]:
    """Reuses (memoizes) pre-existing results for executing experiment_reference on entities if possible.

    Memoization involves creating a ReplayedMeasurementRequest for each existing MeasurementResult for experiment_reference on an entity.
    i.e. if entity A has 5 MeasurementResults for experiment_reference, 5 ReplayedMeasurement are created for that entity

    If memoization is not possible for an entity nothing happens.

    Params:
        requestIndex: The index of this request
        requesterid: The id of the requester
        entities: A list entities to be measured
        experiment_reference: The experiment to perform
        measurement_queue: Used for forwarding already completed measurements


    Returns: A list of the created ReplayedMeasurements
    """

    moduleLog.debug(
        f"Checking if application of {experiment_reference} to any of {[e.identifier for e in entities]} can be memoized"
    )

    requests = []
    for e in entities:
        moduleLog.debug(
            e.measurement_results_for_experiment_reference(
                experiment_reference=experiment_reference
            )
        )
        moduleLog.debug([r.experimentReference for r in e.measurement_results])
        for measurement_to_replay in e.measurement_results_for_experiment_reference(
            experiment_reference=experiment_reference
        ):
            request = ReplayedMeasurement(
                operation_id=requesterid,
                requestIndex=requestIndex,
                entities=[e],
                experimentReference=experiment_reference,
                status=MeasurementRequestStateEnum.SUCCESS,
                measurements=(measurement_to_replay,),
            )

            moduleLog.debug(
                f"Replaying measurement: id: {request.requestid}, index: {requestIndex}, expt: {experiment_reference} from requester {requesterid}"
            )

            measurement_queue.put_nowait(request)
            requests.append(request)

    return requests


@ray.remote
class Replay(ActuatorBase):
    """Special actuator for handling externally defined experiments (experiments we don't have code for)

    NOte: Using this actuator to try to run external experiments (via submit) on an Entity always results in an InvalidMeasurementResult.
    This is because these experiments cannot be executed as we don't have the code.

    The replay function above allows ado to reuse (memoize) pre-existing result of applying any experiment to an entity, including external experiments.
    This is how results from external defined experiments can be reused.

    The reason this Actuator is required is
    - (a) all experiments need to be associated with an ActuatorClass -> this is the Actuator for external defined experiments
    - (b) there are cases the submit method will be called and hence it must exist and behaves correctly

    The submit method will be called when:
    - An explore operation needs to apply an external defined experiment to an entity but no pre-existing results exist (and memoization is on)
    - An explore operation needs to apply an external defined experiment to an entity and memoization is off
    """

    identifier = "replay"

    def __init__(self, queue: MeasurementQueue, params: dict | None = None) -> None:
        enable_ray_actor_coverage("replay")
        super().__init__(queue=queue, params=params)
        self.log = logging.getLogger("replay")

        self._catalog = orchestrator.modules.actuators.catalog.ExperimentCatalog()

    def submit(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
    ) -> list[str]:

        # submit a request to the Replay actuator to run an experimentReference always results in InvalidMeasurementResults
        # The replay actuator cannot perform any of these experiments - they can only be replayed/memoized
        request = MeasurementRequest(
            operation_id=requesterid,
            requestIndex=requestIndex,
            requestid=f"{requesterid}-{str(uuid.uuid4())[:6]}",
            entities=entities,
            experimentReference=experimentReference,
            status=MeasurementRequestStateEnum.FAILED,
            measurements=tuple(
                [
                    InvalidMeasurementResult(
                        entityIdentifier=e.identifier,
                        reason=f"Externally defined experiments cannot be applied to entities: {experimentReference}. ",
                        experimentReference=experimentReference,
                    )
                    for e in entities
                ]
            ),
        )

        self._stateUpdateQueue.put_nowait(request)

        return [request.requestid]

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> orchestrator.modules.actuators.catalog.ExperimentCatalog:
        return orchestrator.modules.actuators.catalog.ExperimentCatalog()
