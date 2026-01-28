# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import typing

import ray

from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager

if typing.TYPE_CHECKING:
    from orchestrator.schema.entity import Entity


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
