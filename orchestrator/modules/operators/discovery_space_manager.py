# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import abc
import asyncio
import logging
import typing

import ray
import ray.util.queue
from ray.actor import ActorHandle

import orchestrator.core.discoveryspace.space
import orchestrator.core.operation.resource
import orchestrator.core.resources
import orchestrator.schema.entityspace
import orchestrator.schema.measurementspace
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.metastore.project import (
    ProjectContext,
)
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.property_value import PropertyValue
from orchestrator.schema.request import MeasurementRequest
from orchestrator.utilities.environment import enable_ray_actor_coverage
from orchestrator.utilities.logging import configure_logging

if typing.TYPE_CHECKING:
    from orchestrator.metastore.sqlstore import SQLStore
    from orchestrator.schema.entity import Entity
    from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
    from orchestrator.schema.observed_property import ObservedProperty
    from orchestrator.schema.property import AbstractPropertyDescriptor

configure_logging()


class DiscoverySpaceUpdateSubscriber(abc.ABC):

    @abc.abstractmethod
    def onUpdate(self, measurementRequest: MeasurementRequest) -> None:
        pass

    @abc.abstractmethod
    def onCompleted(self) -> None:
        pass

    @abc.abstractmethod
    def onError(self, error: Exception) -> None:
        pass


@ray.remote
class DiscoverySpaceManager:
    """A Ray actor wrapping a DiscoverySpace

    ray namespace scoped i.e. All ray actors accessing a DiscoverySpaceManager instance
    should be in the same ray namespace as that instance.

    - Provides remote/async access to discovery space properties
    - Handles insertion of new entities and measurements into the space coming from a MeasurementQueue.
    - Notifies subscribers of update events.
    - Notifies subscribers of shutdown
    """

    @classmethod
    def fromConfiguration(
        cls,
        queue: MeasurementQueue,
        name: str,
        definition: DiscoverySpaceConfiguration,
        project_context: ProjectContext,
        namespace: str | None = None,
    ) -> ActorHandle["DiscoverySpaceManager"]:

        configure_logging()

        log = logging.getLogger("space_manager")
        log.debug("Initialising InternalState from DiscoverySpace configuration")

        # Note: Have to trigger loading of base actuators in every remote actor
        log.debug("The definition is for creating a new DiscoverySpace")
        conf = definition
        discoverySpace = (
            orchestrator.core.discoveryspace.space.DiscoverySpace.from_configuration(
                conf=conf, project_context=project_context, identifier=None
            )
        )

        # noinspection PyArgumentList
        return DiscoverySpaceManager.options(namespace=namespace, name=name).remote(
            queue=queue, space=discoverySpace, namespace=namespace
        )

    @classmethod
    def fromStorage(
        cls,
        queue: MeasurementQueue,
        name: str,
        project_context: ProjectContext,
        space_identifier: str,
        namespace: str | None = None,
    ) -> ActorHandle["DiscoverySpaceManager"]:

        log = logging.getLogger("space_manager")
        log.debug("Initialising InternalState from DiscoverySpace configuration")

        # Note: Have to trigger loading of base actuators in every remote actor
        log.debug("The definition is for accessing an existing DiscoverySpace")
        conf = project_context
        discoverySpace = orchestrator.core.discoveryspace.space.DiscoverySpace.from_stored_configuration(
            conf, space_identifier=space_identifier
        )

        # noinspection PyArgumentList
        return DiscoverySpaceManager.options(namespace=namespace, name=name).remote(
            queue=queue, space=discoverySpace, namespace=namespace
        )

    def __init__(
        self,
        queue: MeasurementQueue,
        space: orchestrator.core.discoveryspace.space.DiscoverySpace,
        namespace: str | None = None,
    ) -> None:
        """
        :param queue: A MeasurementQueue instance for this operation
            All actuators in the same operation must use this queue
        :param space: The DiscoverySpace instance

        """

        enable_ray_actor_coverage("space_manager")
        self.log = logging.getLogger("space_manager")
        self.log.debug("Initialising DiscoverySpaceManager")

        self._namespace = namespace
        if self._namespace != queue.ray_namespace():
            raise ValueError(
                f"The provided measurement queue's ray namespace, {queue.ray_namespace()}, does "
                f"not match the namespace provided to DiscoverySpaceManager, {self._namespace} "
            )

        # This ivar will be used to mimic receiving updates on Measurements
        self._measurement_queue = queue
        self._discoverySpace = space

        self.log.debug(f"Accessing DiscoverySpace {self._discoverySpace.uri}")
        self._subscribers: dict[str, DiscoverySpaceUpdateSubscriber] = {}
        self.isalive = True
        self.iscompleted = False
        # Required to keep a strong ref to the co-routine created by create_task in startMonitoring
        self.monitoringTask = None

        self.log.debug("Completed DiscoverySpaceManager initialization")

    def measurement_queue(self) -> MeasurementQueue:

        return self._measurement_queue

    def discoverySpace(self) -> orchestrator.core.discoveryspace.space.DiscoverySpace:

        return self._discoverySpace

    def measurementSpace(self) -> orchestrator.schema.measurementspace.MeasurementSpace:

        return self._discoverySpace.measurementSpace

    def saveSpace(self) -> None:

        self._discoverySpace.saveSpace()

    def entitySpace(self) -> orchestrator.schema.entityspace.EntitySpaceRepresentation:

        return self._discoverySpace.entitySpace

    def startMonitoring(self) -> None:

        # See https://github.com/python/cpython/issues/88831 for the reason you need to keep the ref
        self.monitoringTask = asyncio.get_event_loop().create_task(
            self.monitorUpdates(debug=False)
        )

    async def matchingEntitiesInSource(
        self, selection: list[int] | None = None
    ) -> list["Entity"]:
        """Returns an ordered list of all matchingEntities or a selected subset of them

        :param: selection: A list of ints. If supplied the entities at these indexes are returned.
            Default=None. Therefore, all matchingEntities are returned

        :return: A list of Entity instances"""

        entities = self._discoverySpace.matchingEntities()
        if selection is not None:
            selected = [entities[i] for i in selection]
            entities = selected

        return entities

    async def entity(self, index: int = 0) -> "Entity":

        entities = await self.matchingEntitiesInSource()
        return entities[index]

    async def entitiesSlice(self, start: int = 0, stop: int = 1) -> list["Entity"]:

        entities = await self.matchingEntitiesInSource()
        return entities[start:stop]

    def storedEntityWithIdentifier(self, entityIdentifier: str) -> "Entity | None":

        return self._discoverySpace.sample_store.entityWithIdentifier(
            entityIdentifier=entityIdentifier
        )

    def storedEntitiesWithConstitutivePropertyValues(
        self, propVals: list[PropertyValue]
    ) -> list["Entity"]:

        return self._discoverySpace.storedEntitiesWithConstitutivePropertyValues(
            values=propVals
        )

    def entity_for_point(self, point: dict[str, typing.Any]) -> "Entity":

        return self._discoverySpace.entity_for_point(point)

    def numberOfMatchingEntitiesInSource(self) -> int:

        return len(self._discoverySpace.matchingEntities())

    async def monitorUpdates(self, debug: bool = False) -> None:

        # Unexpected exceptions from async functions in ray are silently swallowed
        # The purpose of this wrapper is to ensure any such error from the
        # core monitor_updates loop is logged
        # This will be a bug so we don't try to handle it
        try:
            await self._monitor_updates_private()
        except Exception as error:
            self.log.critical(f"Unexpected error {error}")

    async def _monitor_updates_private(self) -> None:
        promises = []
        monitoringError = False
        self.log.debug("Beginning observation of measurement queue")
        while self.isalive:
            measurement_request: MeasurementRequest | None = None
            try:
                self.log.debug("Waiting for new measurement")
                measurement_request = await self._measurement_queue.get_async(
                    block=True, timeout=5
                )  # type: MeasurementRequest
                self._discoverySpace.addMeasurement(measurement_request)
            except ray.util.queue.Empty:
                # e object to help in debugging
                self.log.info(
                    "Did not get a new measurement after 5 secs - will continue waiting"
                )
            except Exception as error:
                self.log.critical(
                    f"Unexpected exception for measurement {measurement_request} - {error}. Will shutdown"
                )
                self.isalive = False
                monitoringError = True
                self.log.info(
                    f"There are {len(self._subscribers.items())} subscribers to notifify"
                )
                subscriber_copy = self._subscribers.copy()
                for subscriberName, subscriber in subscriber_copy.items():
                    self.log.info(f"Notifying subscriber {subscriber}")
                    subscriber.onError.remote(error)
                    self.log.info("Unsubscribing subscriber due to error")
                    self.unsubscribeFromUpdates(subscriberName)
                    self.log.info("Complete")
            else:
                self.log.debug(
                    f"Received new measurement: {measurement_request} notifying subscribers"
                )
                subscriber_copy = self._subscribers.copy()
                for subscriberName, subscriber in subscriber_copy.items():
                    try:
                        promise = subscriber.onUpdate.remote(measurement_request)
                        promises.append(promise)
                    except Exception as error:  # noqa: PERF203
                        self.log.info(
                            f"Exception {error} while notifying subscriber of update {subscriber}"
                        )
                        self.log.info("Notifying subscriber")
                        subscriber.onError.remote(error)
                        self.log.info("Unsubscribing subscriber due to error")
                        self.unsubscribeFromUpdates(subscriberName)

        # Don't send iscomplete until subscribers are finished.
        # The subscribers may have outstanding updates to process and not know about it
        self.log.info(f"Awaiting {len(promises)} sent updates")
        await asyncio.gather(*promises)
        self.log.info("All updates processed")

        if not monitoringError:
            self.log.info(
                "Measurement queue observation complete - notifying subscribers"
            )
            for subscriber in self._subscribers.values():
                try:
                    self.log.info("Notifying subscriber")
                    await subscriber.onCompleted.remote()
                except Exception as error:  # noqa: PERF203
                    self.log.info(
                        f"Exception error {error} while notifying subscriber of completion {subscriber}"
                    )
        else:
            self.log.critical("Measurement queue observation exited due to error")

        self.iscompleted = True

    def subscribeToUpdates(self, subscriberName: str) -> bool:

        self.log.debug(f"Subscription request {subscriberName}")
        self._subscribers[subscriberName] = ray.get_actor(
            subscriberName, namespace=self._namespace
        )
        return True

    def unsubscribeFromUpdates(self, subscriberName: str) -> None:

        try:
            self._subscribers.pop(subscriberName)
        except KeyError:
            self.log.info(
                f"Received request to unsubscribe {subscriberName} but it appears"
                f" it was already unsubscribed - ignoring"
            )

    async def shutdown(self) -> None:

        self.isalive = False
        self.log.info(
            "Received shutdown - waiting on measurement queue observation to complete"
        )
        while not self.iscompleted:
            self.log.info("Not complete - waiting 10 secs")
            await asyncio.sleep(10)
        self._discoverySpace.sample_store.commit()
        self.log.info("Shutdown complete")

    def stateIdentifier(self) -> str:

        return self._discoverySpace.uri

    def targetProperties(self) -> list["AbstractPropertyDescriptor"]:

        return self._discoverySpace.measurementSpace.targetProperties

    def observedProperties(self) -> list["ObservedProperty"]:

        return self._discoverySpace.measurementSpace.observedProperties

    def experiments(self) -> "list[Experiment | ParameterizedExperiment]":

        return self._discoverySpace.measurementSpace.experiments

    def metadataStore(self) -> "SQLStore":

        return self._discoverySpace.metadataStore


if typing.TYPE_CHECKING:
    DiscoverySpaceManagerActor = type[ActorHandle[DiscoverySpaceManager]]
