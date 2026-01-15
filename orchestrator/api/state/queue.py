# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""
Queue monitor implementation for the Orchestrator API.

The :class:`~orchestrator.api.state.queue.QueueMonitorActor` is a Ray
remote actor that keeps an in-memory representation of all measurement
requests.  It listens to a shared :class:`~orchestrator.modules.actuators.measurement_queue.MeasurementQueue`
and updates a nested ``dict`` that maps
``ExperimentReference`` ➜ ``requestid`` ➜ :class:`MeasurementRequest`.

This in-memory store can be queried by the API to retrieve all requests for
an experiment or a specific request by its identifier.
"""

import asyncio
import logging

import ray
from fastapi import HTTPException, status
from ray.util import queue

from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest


@ray.remote
class QueueMonitorActor:
    """Ray remote actor that monitors the shared :class:`MeasurementQueue`.

    The actor maintains an in-memory data structure that the API can access
    to retrieve Measurement Requests. It launches a background task that
    continuously pulls from the queue until the actor is terminated.
    """

    def __init__(self) -> None:
        """Create a new :class:`QueueMonitorActor`.

        The constructor configures logging, obtains the shared queue, and
        starts the background monitoring coroutine.
        """
        self.logger = logging.getLogger("QueueMonitorActor")
        self.shared_queue = MeasurementQueue()

        # The in-memory store: experiment reference → request ID → request instance.
        self.requests_memory_storage: dict[
            ExperimentReference, dict[str, MeasurementRequest]
        ] = {}

        self.task_reference = asyncio.get_event_loop().create_task(
            self.start_monitoring_queue()
        )

    def get_queue(self) -> MeasurementQueue:
        """Return the shared :class:`MeasurementQueue`.

        Returns:
            MeasurementQueue
                The queue instance used by the actor.
        """
        return self.shared_queue

    def add_measurement_request(self, measurement_request: MeasurementRequest) -> None:
        """Persist a ``MeasurementRequest`` in the in-memory store.

        The request is keyed by its :attr:`experimentReference` and
        :attr:`requestid`.

        Args:
            measurement_request:
                The :class:`MeasurementRequest` to store.
        """
        if measurement_request.experimentReference not in self.requests_memory_storage:
            self.requests_memory_storage[measurement_request.experimentReference] = {}

        self.requests_memory_storage[measurement_request.experimentReference][
            measurement_request.requestid
        ] = measurement_request

    def get_measurement_requests(
        self,
        experiment_reference: ExperimentReference,
    ) -> list[MeasurementRequest]:
        """Return all requests belonging to the given experiment reference.

        Args:
            experiment_reference:
                The :class:`ExperimentReference` whose requests should be
                returned.

        Returns:
            list[MeasurementRequest]
                All measurement requests associated with the experiment.  If the
                experiment has no recorded requests, an empty list is returned.
        """
        if experiment_reference not in self.requests_memory_storage:
            return []

        return list(self.requests_memory_storage[experiment_reference].values())

    def get_measurement_request_by_id(
        self, experiment_reference: ExperimentReference, request_id: str
    ) -> MeasurementRequest:
        """Retrieve a specific measurement request by ID.

        Args:
            experiment_reference:
                The experiment that owns the request.
            request_id:
                The unique identifier for the request.

        Returns:
            MeasurementRequest
                The requested measurement request.

        Raises:
            fastapi.HTTPException
                If the experiment does not exist or the request ID cannot be
                found.
        """
        if (
            experiment_reference not in self.requests_memory_storage
            or request_id not in self.requests_memory_storage[experiment_reference]
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Request {request_id} not found for {experiment_reference}",
            )

        return self.requests_memory_storage[experiment_reference][request_id]

    async def start_monitoring_queue(self) -> None:
        """Continuously consume the shared MeasurementQueue and update memory.

        The coroutine waits for new :class:`MeasurementRequest` objects via
        :meth:`MeasurementQueue.get_async`. Any unexpected exception
        is logged, the coroutine sleeps for a second, and then retries.

        This loop only terminates when the actor is cancelled.
        """

        while True:
            try:
                try:
                    self.logger.debug("Waiting for new MeasurementRequests")
                    measurement_request: MeasurementRequest = (
                        await self.shared_queue.get_async(block=True, timeout=30)
                    )
                except queue.Empty:
                    self.logger.info(
                        "Did not get any new MeasurementRequests after 30 secs - will continue waiting"
                    )
                else:
                    self.logger.debug(
                        "Adding measurement request to the entries %s",
                        measurement_request,
                    )
                    self.add_measurement_request(
                        measurement_request=measurement_request
                    )

            except Exception as error:  # noqa: PERF203
                self.logger.warning(
                    f"Unexpected exception in monitor loop: {type(error)} {error}"
                )
                self.logger.warning("Assuming transient - will continue")
                await asyncio.sleep(1)
