# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""Watch the shared MeasurementQueue and update in-memory request storage.

The :func:`watch_queue` coroutine continuously pulls from the
``orchestrator.modules.actuators.measurement_queue.MeasurementQueue`` and
stores requests via :func:`set_request_in_memory_storage`. It handles
timeouts and exception cases by logging and retrying after a short delay.
"""

import asyncio
import logging

from ray.util import queue

from orchestrator.api.state.in_memory_requests_storage import (
    set_request_in_memory_storage,
)
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.request import MeasurementRequest

shared_queue = MeasurementQueue.get_measurement_queue()


async def watch_queue():
    """Continuously consume the shared MeasurementQueue and update in-memory storage.

    Blocks on :meth:`MeasurementQueue.get_async` with a 30-second timeout.  When
    a :class:`orchestrator.schema.request.MeasurementRequest` is retrieved it
    is forwarded to ``set_request_in_memory_storage``. A timeout results in an
    informational log entry; unexpected exceptions are logged and the loop
    retries after sleeping one second. The coroutine exits only when
    cancelled.
    """
    logger = logging.getLogger("Queue Monitor")

    while True:
        try:
            # Get new updates
            try:
                logger.debug("Waiting for new MeasurementRequests")
                measurement_request: MeasurementRequest = await shared_queue.get_async(
                    block=True, timeout=30
                )
            except queue.Empty:
                logger.info(
                    "Did not get any new MeasurementRequests after 30 secs - will continue waiting"
                )
            else:
                set_request_in_memory_storage(measurement_request=measurement_request)

        except Exception as error:
            logger.warning(
                f"Unexpected exception in monitor loop: {type(error)} {error}"
            )
            logger.warning("Assuming transient - will continue")
            await asyncio.sleep(1)
