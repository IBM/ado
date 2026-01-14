# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import typing
from collections import OrderedDict
from collections.abc import Callable

import ray
from ray.actor import ActorHandle

shutdown_signal_received = False
CLEANER_ACTOR = "resource_cleaner"

moduleLog = logging.getLogger("orchestration_cleanup")
cleanup_callback_functions: dict[str, Callable[[], None]] = OrderedDict()


def graceful_operation_shutdown_signal_handler() -> (
    typing.Callable[[int, typing.Any | None], None]
):
    """Handler which executes cleanup callbacks registered by operations on receiving a signal"""

    def handler(sig: int, frame: typing.Any | None) -> None:  # noqa: ANN401

        moduleLog.critical(f"Got signal {sig}")
        global shutdown_signal_received
        global cleanup_callback_functions

        if shutdown_signal_received:
            moduleLog.info("Graceful shutdown already completed")

        shutdown_signal_received = True
        moduleLog.info("Calling cleanup callbacks")
        for entry in cleanup_callback_functions:
            moduleLog.info(f"Cleaning {entry}")
            cleanup_callback_functions[entry]()

    return handler


@ray.remote
class ResourceCleaner:
    """
    This is a singleton allowing various custom actors to clean up before shutdown,
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        # list of handles for the actors to be cleaned
        self.to_clean = []

    def add_to_cleanup(self, handle: ActorHandle) -> None:
        """
        Add to clean up
        Can be used by any custom actor to add itself to clean up list. This class has to implement cleanup method
        :param handle: handle of the actor to be cleaned
        :return: None
        """
        self.to_clean.append(handle)

    def cleanup(self) -> None:
        """
        Clean up all required classes
        :return: None
        """
        if len(self.to_clean) > 0:
            handles = [h.cleanup.remote() for h in self.to_clean]
            done, not_done = ray.wait(
                ray_waitables=handles, num_returns=len(handles), timeout=60.0
            )
            moduleLog.info(f"cleaned {len(done)}, clean failed {len(not_done)}")


def initialize_ray_resource_cleaner(namespace: str | None = None) -> None:
    # create a cleaner actor.
    # We are creating Named detached actor (https://docs.ray.io/en/latest/ray-core/actors/named-actors.html)
    # so that we do not need to pass its handle (can get it by name) and it does not go out of scope, until
    # we explicitly kill it
    ResourceCleaner.options(
        name=CLEANER_ACTOR, get_if_exists=True, lifetime="detached", namespace=namespace
    ).remote()
