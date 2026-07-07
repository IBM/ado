# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import time
import typing
import uuid

import ray

from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager

if typing.TYPE_CHECKING:
    from orchestrator.schema.entity import Entity


@ray.remote
class _ErrorCapturingSubscriber:
    """Ray actor that records the first error delivered via onError."""

    def __init__(self) -> None:
        """Initialise with no captured error."""
        self._error: Exception | None = None

    def onUpdate(self, measurement_request: object) -> None:
        """Ignore updates."""

    def onCompleted(self) -> None:
        """Ignore completion."""

    def onError(self, error: Exception) -> None:
        """Capture the error for later inspection."""
        self._error = error

    def get_error(self) -> Exception | None:
        """Return the captured error, or None if onError was not called yet."""
        return self._error


def test_internal_state_direct_init(
    pfas_space: DiscoverySpace,
) -> None:
    """Tests InternalState actor can be initialised with a DiscoverySpace instance"""

    queue = MeasurementQueue()
    state = DiscoverySpaceManager.remote(queue=queue, space=pfas_space)

    try:
        assert state

        targetProperties = ray.get(state.targetProperties.remote())
        observedProperties = ray.get(state.observedProperties.remote())
        numberEntities = ray.get(state.numberOfMatchingEntitiesInSource.remote())
        experiments = ray.get(state.experiments.remote())
        firstEntity = ray.get(state.entity.remote())
        lastEntity = ray.get(state.entity.remote(index=numberEntities - 1))

        assert set(targetProperties) == set(
            pfas_space.measurementSpace.targetProperties
        )
        assert set(observedProperties) == set(
            pfas_space.measurementSpace.observedProperties
        )
        assert numberEntities == pfas_space.sample_store.numberOfEntities
        assert experiments == pfas_space.measurementSpace.experiments
        assert firstEntity == pfas_space.sample_store.entities[0]
        assert lastEntity == pfas_space.sample_store.entities[-1]
    finally:
        ray.kill(state)


def test_internal_state_conf_init(
    pfas_space_configuration: DiscoverySpaceConfiguration,
    pfas_space: DiscoverySpace,
) -> None:
    """Tests InternalState actor can be initialised with a DiscoverySpaceConfiguration"""

    pfas_space_configuration.sampleStoreIdentifier = pfas_space.sample_store.identifier

    queue = MeasurementQueue()
    state = DiscoverySpaceManager.fromConfiguration(
        queue=queue,
        name="State",
        definition=pfas_space_configuration,
        project_context=pfas_space.project_context,
    )

    try:
        assert state

        targetProperties = ray.get(state.targetProperties.remote())
        observedProperties = ray.get(state.observedProperties.remote())
        numberEntities = ray.get(state.numberOfMatchingEntitiesInSource.remote())
        experiments = ray.get(state.experiments.remote())
        firstEntity: Entity = ray.get(state.entity.remote())
        lastEntity: Entity = ray.get(state.entity.remote(index=numberEntities - 1))

        assert set(targetProperties) == set(
            pfas_space.measurementSpace.targetProperties
        )
        assert set(observedProperties) == set(
            pfas_space.measurementSpace.observedProperties
        )
        assert numberEntities == pfas_space.sample_store.numberOfEntities
        assert experiments == pfas_space.measurementSpace.experiments

        for expected, actual in [
            (firstEntity, pfas_space.matchingEntities()[0]),
            (lastEntity, pfas_space.matchingEntities()[-1]),
        ]:
            assert expected.constitutiveProperties == actual.constitutiveProperties
            assert (
                expected.constitutive_property_values
                == actual.constitutive_property_values
            )
            assert expected.observedProperties == actual.observedProperties
            assert expected.observedPropertyValues == actual.observedPropertyValues

    finally:
        ray.kill(state)


def test_on_error_passes_plain_exception_to_subscriber(
    pfas_space: DiscoverySpace,
) -> None:
    """DSM must forward a plain Exception to subscribers when the queue dies.

    When the MeasurementQueue's backing Ray actor is killed, get_async raises
    an ActorUnavailableError that contains an unresolvable ObjectRef.  Passing
    that raw exception to subscriber.onError.remote() would fail silently
    because Ray cannot serialise ObjectRef-embedded exceptions across the
    remote boundary.  The fix wraps it as Exception(str(error)), which is
    always serialisable.  This test verifies:

    1. onError is called on the subscriber at all.
    2. The received argument is a plain Exception, not a Ray internal type.
    """
    subscriber_name = f"test-error-subscriber-{uuid.uuid4().hex[:8]}"

    queue = MeasurementQueue()
    state = DiscoverySpaceManager.remote(queue=queue, space=pfas_space)
    subscriber = _ErrorCapturingSubscriber.options(name=subscriber_name).remote()

    try:
        ray.get(state.subscribeToUpdates.remote(subscriber_name))

        # Kill the queue's backing actor to trigger ActorUnavailableError
        # inside _monitor_updates_private.
        ray.kill(queue.actor)

        ray.get(state.startMonitoring.remote())

        # Give the async monitor loop time to detect the dead queue and fire
        # onError on the subscriber.  Poll so the test does not take longer
        # than necessary, but cap at 15 s to avoid hanging CI.
        deadline = time.monotonic() + 15.0
        captured_error: Exception | None = None
        while time.monotonic() < deadline:
            captured_error = ray.get(subscriber.get_error.remote())
            if captured_error is not None:
                break
            time.sleep(0.25)

        assert captured_error is not None, (
            "onError was not called on the subscriber within 15 s after "
            "the queue actor was killed"
        )
        assert type(captured_error) is Exception, (
            f"Expected a plain Exception but received {type(captured_error)}"
        )
    finally:
        ray.kill(state)
        ray.kill(subscriber)
