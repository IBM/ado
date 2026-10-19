# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Sequential measurement scheduling for predictive characterization operators."""

from __future__ import annotations

import typing

from ado.modules.operators.randomwalk import RandomWalk
from ado.schema.request import MeasurementRequestStateEnum

if typing.TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from queue import Queue

    from ado.modules.actuators.base import ActuatorBase
    from ado.modules.operators.discovery_space_manager import DiscoverySpaceManager
    from ado.schema.entity import Entity
    from ado.schema.measurementspace import MeasurementSpace
    from ado.schema.request import MeasurementRequest


class _SequentialPredictiveWalk(RandomWalk):
    """Complete an entity's measurements before requesting the next entity."""

    def __init__(
        self,
        operationActorName: str,
        namespace: str,
        discovery_space_manager: DiscoverySpaceManager,
        actuators: dict[str, ActuatorBase],
        params: dict | None = None,
    ) -> None:
        """Initialize the completed-request count and measurement scheduler."""

        self._finished_requests = 0
        super().__init__(
            operationActorName=operationActorName,
            namespace=namespace,
            discovery_space_manager=discovery_space_manager,
            actuators=actuators,
            params=params,
        )

    async def run(self) -> None:
        """Run measurements and propagate critical infrastructure failures."""

        await super().run()
        if self.criticalError:
            raise RuntimeError(
                "measurement operation stopped after a critical infrastructure error"
            )

    def _processCompletedMeasurement(
        self,
        measurementRequest: MeasurementRequest,
        measurementSpace: MeasurementSpace,
        continuousBatchingQueue: Queue,
    ) -> bool:
        """Count completed attempts and report exhausted experiment failures."""

        if measurementRequest.operation_id != self.operationIdentifier():
            return False
        completed = super()._processCompletedMeasurement(
            measurementRequest, measurementSpace, continuousBatchingQueue
        )
        self._finished_requests += 1
        if (
            completed
            and measurementRequest.status != MeasurementRequestStateEnum.SUCCESS
        ):
            identifiers = [entity.identifier for entity in measurementRequest.entities]
            raise RuntimeError(
                f"experiment {measurementRequest.experimentReference} exhausted "
                f"its retry limit ({self.params.maxRetries}) for entities {identifiers}"
            )
        return completed

    async def _sampleEntityAndAddMeasurementsToQueue(
        self,
        continuousBatchingQueue: Queue,
        iterator: AsyncGenerator[list[Entity], None],
        measurementSpace: MeasurementSpace,
    ) -> list[Entity]:
        """Sample only after all submitted measurement attempts have finished."""

        if self._finished_requests < self._experimentsRequested:
            return []
        return await super()._sampleEntityAndAddMeasurementsToQueue(
            continuousBatchingQueue, iterator, measurementSpace
        )
