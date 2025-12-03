# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import typing
from collections import OrderedDict
from collections.abc import Callable

import ray
from ray.actor import ActorHandle

shutdown_signal_received = False
shutdown_signal_received = False
CLEANER_ACTOR = "resource_cleaner"

moduleLog = logging.getLogger("orchestration_cleanup")
signal_cleanup_callbacks: dict[str, Callable[[], None]] = OrderedDict()


def graceful_operation_shutdown_signal_handler() -> (
    typing.Callable[[int, typing.Any | None], None]
):
    """Handler which executes cleanup callbacks registered by operations on receiving a signal"""

    def handler(sig, frame):

        moduleLog.warning(f"Got signal {sig}")
        global shutdown_signal_received
        global signal_cleanup_callbacks

        if not shutdown_signal_received:
            shutdown_signal_received = True
            moduleLog.debug("Cleanup custom actors")
            try:
                cleaner_handle = ray.get_actor(name=CLEANER_ACTOR)
                ray.get(cleaner_handle.cleanup.remote())
                # deleting a cleaner actor. It is detached one, so has to be deleted explicitly
                ray.kill(cleaner_handle)
            except ValueError:
                moduleLog.debug("No ray actor")
            except Exception as e:
                moduleLog.warning(f"Failed to cleanup custom actors {e}")

            moduleLog.warning("Calling cleanup callbacks")
            for subscriber in signal_cleanup_callbacks:
                moduleLog.warning(f"Cleaning {subscriber}")
                signal_cleanup_callbacks[subscriber]()
        else:
            moduleLog.info("Graceful shutdown already completed")


def graceful_operation_shutdown_signal_handler() -> (
    typing.Callable[[int, typing.Any | None], None]
):
    """Handler which executes cleanup callbacks registered by operations on receiving a signal"""

    def handler(sig, frame):

        moduleLog.warning(f"Got signal {sig}")
        global shutdown_signal_received
        global signal_cleanup_callbacks

        if not shutdown_signal_received:
            shutdown_signal_received = True
            moduleLog.debug("Cleanup custom actors")
            try:
                cleaner_handle = ray.get_actor(name=CLEANER_ACTOR)
                ray.get(cleaner_handle.cleanup.remote())
                # deleting a cleaner actor. It is detached one, so has to be deleted explicitly
                ray.kill(cleaner_handle)
            except ValueError:
                moduleLog.debug("No ray actor")
            except Exception as e:
                moduleLog.warning(f"Failed to cleanup custom actors {e}")

            moduleLog.warning("Calling cleanup callbacks")
            for subscriber in signal_cleanup_callbacks:
                moduleLog.warning(f"Cleaning {subscriber}")
                signal_cleanup_callbacks[subscriber]()
        else:
            moduleLog.info("Graceful shutdown already completed")
            moduleLog.warning("Calling cleanup callbacks")
            for subscriber in signal_cleanup_callbacks:
                moduleLog.warning(f"Cleaning {subscriber}")
                signal_cleanup_callbacks[subscriber]()
        else:
            moduleLog.info("Graceful shutdown already completed")

    return handler


@ray.remote
class ResourceCleaner:
    """
    This is a singleton allowing various custom actors to clean up before shutdown,
    """

    def __init__(self):
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


def initialize_ray_resource_cleaner():
    # create a cleaner actor.
    # We are creating Named detached actor (https://docs.ray.io/en/latest/ray-core/actors/named-actors.html)
    # so that we do not need to pass its handle (can get it by name) and it does not go out of scope, until
    # we explicitly kill it
    ResourceCleaner.options(
        name=CLEANER_ACTOR, get_if_exists=True, lifetime="detached", namespace=namespace
    ).remote()
