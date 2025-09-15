# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import asyncio
import logging

import ray.util.queue

from orchestrator.api.state.in_memory_requests_storage import (
    set_request_in_memory_storage,
)
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.request import MeasurementRequest

shared_queue = MeasurementQueue.get_measurement_queue()


async def watch_queue():

    logger = logging.getLogger("Queue")

    while True:
        try:

            # Get new updates
            try:
                logger.debug("Awaiting update queue get")
                update: MeasurementRequest = await shared_queue.get_async(
                    block=True, timeout=30
                )
            except ray.util.queue.Empty:
                logger.info(
                    "Did not get an update after 30 secs - will continue waiting"
                )
            else:
                set_request_in_memory_storage(measurement_request=update)

        except Exception as error:
            logger.warning(
                f"Unexpected exception in monitor loop: {type(error)} {error}"
            )
            logger.warning("Assuming transient - will continue")
            await asyncio.sleep(1)
