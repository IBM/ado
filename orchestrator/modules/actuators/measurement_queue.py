# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import ray.util.queue


class MeasurementQueue(ray.util.queue.Queue):
    """Class used to relay measurements for an explore operation

    The DiscoverySpaceManager and all Actuators in an explore operation MUST
    use the same MeasurementQueue instance."""

    def __init__(self, maxsize: int = 0, ray_namespace: str | None = None) -> None:
        """Parameters:

        ray_namespace: The namespace of the operation this queue is for.
        Can be None in which case this indicates it is the default ray namespace for the job
        (get_runtime_context().namespace).
        """

        super().__init__(maxsize=maxsize)
        self._ray_namespace = ray_namespace

    def ray_namespace(self) -> str:
        """Returns the ray namespace the operation actors are using"""

        return self._ray_namespace
