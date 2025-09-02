# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging

import ray.util.queue


class MeasurementQueue(ray.util.queue.Queue):
    """Actuators place completed MeasurementRequests in the singleton instance of this queue for addition to the active DiscoverySpace"""

    # stateUpdateQueue = None

    @classmethod
    def get_measurement_queue(cls, maxsize=0):
        """This returns the singleton measurement queue for the current explore operation

        However, its only singleton in the process that creates it.
        i.e. if you execute locally and in a remote actor you will get different objects.
        Solution: Pass the queue to any remote objects that need it"""

        log = logging.getLogger()
        log.debug(f"Getting measurement queue via {cls}")

        return cls(maxsize=maxsize)
