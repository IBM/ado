# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from orchestrator.core.discoveryspace.samplers import (
    ExplicitEntitySpaceGridSampleGenerator,
    RandomSampleSelector,
    SequentialSampleSelector,
    WalkModeEnum,
    sample_random_entity_from_space,
)
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.operators.discovery_space_manager import (
    DiscoverySpaceManager,
)
from orchestrator.schema.entity import Entity
from orchestrator.schema.entityspace import EntitySpaceRepresentation
from orchestrator.schema.measurementspace import MeasurementSpace
from orchestrator.schema.property import ConstitutiveProperty


@pytest.fixture(params=[WalkModeEnum.RANDOM, WalkModeEnum.SEQUENTIAL])
def walk_mode(request: pytest.FixtureRequest) -> WalkModeEnum:

    return request.param


@pytest.fixture
def explicit_entity_space(
    constitutive_property_configuration_general: list[ConstitutiveProperty],
) -> EntitySpaceRepresentation:

    return EntitySpaceRepresentation(constitutive_property_configuration_general)


def test_explicit_space_grid_sampler_entity_space_iterator(
    explicit_entity_space: EntitySpaceRepresentation, walk_mode: WalkModeEnum
) -> None:

    if not explicit_entity_space.isDiscreteSpace:
        assert (
            ExplicitEntitySpaceGridSampleGenerator.samplerCompatibleWithEntitySpace(
                explicit_entity_space
            )
            is False
        )
    else:
        assert (
            ExplicitEntitySpaceGridSampleGenerator.samplerCompatibleWithEntitySpace(
                explicit_entity_space
            )
            is True
        )

        names = [c.identifier for c in explicit_entity_space.constitutiveProperties]
        sampler = ExplicitEntitySpaceGridSampleGenerator(mode=walk_mode)
        count = 0
        entity = None

        iterator = sampler.entitySpaceIterator(
            entitySpace=explicit_entity_space, batchsize=5
        )

        for entityBatch in iterator:
            for entity in entityBatch:  # type: Entity
                count += 1
                cps = [c.identifier for c in entity.constitutiveProperties]
                assert len([k for k in cps if k not in names]) == 0

        if walk_mode == WalkModeEnum.SEQUENTIAL:
            # If its sequential we know the last point must be the constructed from last value in each dim
            for c in explicit_entity_space.constitutiveProperties:
                if c.propertyDomain.values is not None:
                    assert (
                        entity.valueForProperty(c).value == c.propertyDomain.values[-1]
                    )
                else:
                    arange = np.arange(
                        start=c.propertyDomain.domainRange[0],
                        stop=c.propertyDomain.domainRange[1],
                        step=c.propertyDomain.interval,
                    )
                    assert entity.valueForProperty(c).value == arange[-1]

        if walk_mode == WalkModeEnum.RANDOM:
            # If its random its improbable that the last point is the last sequential point
            sequentialLastPoint = {}
            randomLastPoint = {}
            for c in explicit_entity_space.constitutiveProperties:
                if c.propertyDomain.values is not None:
                    sequentialLastPoint[c.identifier] = c.propertyDomain.values[-1]
                    randomLastPoint[c.identifier] = entity.valueForProperty(c).value
                else:
                    arange = np.arange(
                        start=c.propertyDomain.domainRange[0],
                        stop=c.propertyDomain.domainRange[1],
                        step=c.propertyDomain.interval,
                    )
                    sequentialLastPoint[c.identifier] = arange[-1]
                    randomLastPoint[c.identifier] = entity.valueForProperty(c).value

            if randomLastPoint == sequentialLastPoint:
                pytest.xfail(
                    "The last point while using random walk was the last sequential point"
                )
            else:
                assert randomLastPoint != sequentialLastPoint

        assert count == explicit_entity_space.size


@pytest.mark.asyncio
async def test_explicit_space_grid_sampler_async_entity_iterator(
    ml_multi_cloud_space: DiscoverySpace, walk_mode: WalkModeEnum
) -> None:

    space = ml_multi_cloud_space
    if not space.entitySpace.isDiscreteSpace:
        assert (
            ExplicitEntitySpaceGridSampleGenerator.samplerCompatibleWithDiscoverySpace(
                space
            )
            is False
        )
    else:
        assert (
            ExplicitEntitySpaceGridSampleGenerator.samplerCompatibleWithDiscoverySpace(
                space
            )
            is True
        )

        sampler = ExplicitEntitySpaceGridSampleGenerator(mode=walk_mode)
        count = 0
        entity = None
        queue = MeasurementQueue()
        manager = DiscoverySpaceManager.remote(space=space, queue=queue)

        assert (
            ExplicitEntitySpaceGridSampleGenerator.samplerCompatibleWithDiscoverySpaceRemote(
                manager
            )
            is True
        )

        iterator = await sampler.remoteEntityIterator(
            remoteDiscoverySpace=manager, batchsize=5
        )

        async for entityBatch in iterator:
            for entity in entityBatch:  # type: Entity # noqa: B007
                count += 1

        if walk_mode == WalkModeEnum.SEQUENTIAL:
            # If its sequential we know the last point must be the constructed from last value in each dim
            for c in space.entitySpace.constitutiveProperties:
                if c.propertyDomain.values is not None:
                    assert (
                        entity.valueForProperty(c).value == c.propertyDomain.values[-1]
                    )
                else:
                    arange = np.arange(
                        start=c.propertyDomain.domainRange[0],
                        stop=c.propertyDomain.domainRange[1],
                        step=c.propertyDomain.interval,
                    )
                    assert entity.valueForProperty(c).value == arange[-1]

        if walk_mode == WalkModeEnum.RANDOM:
            # If its random its improbable that the last point is the last sequential point
            sequentialLastPoint = {}
            randomLastPoint = {}
            for c in space.entitySpace.constitutiveProperties:
                if c.propertyDomain.values is not None:
                    sequentialLastPoint[c.identifier] = c.propertyDomain.values[-1]
                    randomLastPoint[c.identifier] = entity.valueForProperty(c).value
                else:
                    arange = np.arange(
                        start=c.propertyDomain.domainRange[0],
                        stop=c.propertyDomain.domainRange[1],
                        step=c.propertyDomain.interval,
                    )
                    sequentialLastPoint[c.identifier] = arange[-1]
                    randomLastPoint[c.identifier] = entity.valueForProperty(c).value

            if randomLastPoint == sequentialLastPoint:
                pytest.xfail(
                    "The last point while using random walk was the last sequential point"
                )
            else:
                assert randomLastPoint != sequentialLastPoint

        assert count == space.entitySpace.size


def test_explicit_space_grid_sampler_entity_iterator(
    ml_multi_cloud_space: DiscoverySpace, walk_mode: WalkModeEnum
) -> None:

    space = ml_multi_cloud_space
    if not space.entitySpace.isDiscreteSpace:
        assert (
            ExplicitEntitySpaceGridSampleGenerator.samplerCompatibleWithDiscoverySpace(
                space
            )
            is False
        )
    else:
        assert (
            ExplicitEntitySpaceGridSampleGenerator.samplerCompatibleWithDiscoverySpace(
                space
            )
            is True
        )

        sampler = ExplicitEntitySpaceGridSampleGenerator(mode=walk_mode)
        count = 0
        entity = None

        iterator = sampler.entityIterator(discoverySpace=space, batchsize=5)

        for entityBatch in iterator:
            for entity in entityBatch:  # type: Entity # noqa: B007
                count += 1

        if walk_mode == WalkModeEnum.SEQUENTIAL:
            # If its sequential we know the last point must be the constructed from last value in each dim
            for c in space.entitySpace.constitutiveProperties:
                if c.propertyDomain.values is not None:
                    assert (
                        entity.valueForProperty(c).value == c.propertyDomain.values[-1]
                    )
                else:
                    arange = np.arange(
                        start=c.propertyDomain.domainRange[0],
                        stop=c.propertyDomain.domainRange[1],
                        step=c.propertyDomain.interval,
                    )
                    assert entity.valueForProperty(c).value == arange[-1]

        if walk_mode == WalkModeEnum.RANDOM:
            # If its random its improbable that the last point is the last sequential point
            sequentialLastPoint = {}
            randomLastPoint = {}
            for c in space.entitySpace.constitutiveProperties:
                if c.propertyDomain.values is not None:
                    sequentialLastPoint[c.identifier] = c.propertyDomain.values[-1]
                    randomLastPoint[c.identifier] = entity.valueForProperty(c).value
                else:
                    arange = np.arange(
                        start=c.propertyDomain.domainRange[0],
                        stop=c.propertyDomain.domainRange[1],
                        step=c.propertyDomain.interval,
                    )
                    sequentialLastPoint[c.identifier] = arange[-1]
                    randomLastPoint[c.identifier] = entity.valueForProperty(c).value

            if randomLastPoint == sequentialLastPoint:
                pytest.xfail(
                    "The last point while using random walk was the last sequential point"
                )
            else:
                assert randomLastPoint != sequentialLastPoint

        assert count == space.entitySpace.size


@pytest.mark.asyncio
async def test_random_sample_selector(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:

    space = ml_multi_cloud_space
    sampler = RandomSampleSelector()
    count = 0

    assert space.sample_store.numberOfEntities % 5 != 0
    entities = []
    for entities in sampler.entityIterator(space, batchsize=5):
        count += len(entities)

    assert len(entities) != 5, "Expected the last batch not to be equal to batchsize"

    assert count == len(
        space.matchingEntities()
    ), "Expected the number of entities iterated was equal to number matching entities in source"

    queue = MeasurementQueue()
    manager = DiscoverySpaceManager.remote(space=space, queue=queue)
    assert RandomSampleSelector.samplerCompatibleWithDiscoverySpaceRemote(manager)

    iterator = await sampler.remoteEntityIterator(
        remoteDiscoverySpace=manager, batchsize=5
    )

    count = 0
    async for entities in iterator:
        assert len(entities) <= 5
        count += len(entities)

    assert len(entities) != 5, "Expected the last batch not to be equal to batchsize"

    assert count == len(
        space.matchingEntities()
    ), "Expected the number of entities iterated was equal to number matching entities in source"


@pytest.mark.asyncio
async def test_sequential_sample_selector(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:

    space = ml_multi_cloud_space
    assert space.sample_store.numberOfEntities % 5 != 0

    sampler = SequentialSampleSelector()
    count = 0
    entities = None
    for entities in sampler.entityIterator(space, batchsize=5):
        count += len(entities)

    assert len(entities) != 5, "Expected the last batch not to be equal to batchsize"

    assert (
        entities[-1] == space.matchingEntities()[-1]
    ), "Expect the last entity of sequential iterator to be last matching entity returned by source"

    assert count == len(
        space.matchingEntities()
    ), "Expected the number of entities iterated was equal to number matching entities in source"

    queue = MeasurementQueue()
    manager = DiscoverySpaceManager.remote(space=space, queue=queue)
    assert SequentialSampleSelector.samplerCompatibleWithDiscoverySpaceRemote(manager)

    iterator = await sampler.remoteEntityIterator(
        remoteDiscoverySpace=manager, batchsize=5
    )

    count = 0
    async for entities in iterator:
        count += len(entities)

    assert len(entities) != 5, "Expected the last batch not to be equal to batchsize"

    assert count == len(
        space.matchingEntities()
    ), "Expected the number of entities iterated was equal to number matching entities in source"

    assert (
        entities[-1] == space.matchingEntities()[-1]
    ), "Expect the last entity of sequential iterator to be last matching entity returned by source"


def test_sample_random_entity(
    measurement_space_from_single_parameterized_experiment: MeasurementSpace,
) -> None:

    exp = measurement_space_from_single_parameterized_experiment.experiments[0]
    if not exp.requiredProperties:
        pytest.skip(
            "Cannot form compatible entity space from measurement space with no required propertiers"
        )

    es = measurement_space_from_single_parameterized_experiment.compatibleEntitySpace()
    assert (ent := sample_random_entity_from_space(es))
    assert isinstance(ent, Entity)
