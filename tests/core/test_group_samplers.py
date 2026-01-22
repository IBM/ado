# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from typing import Any

import pytest

from orchestrator.core.discoveryspace.group_samplers import (
    ExplicitEntitySpaceGroupedGridSampleGenerator,
    RandomGroupSampleSelector,
    SequentialGroupSampleSelector,
    _build_groups_dict,
    _get_space_matching_points,
)
from orchestrator.core.discoveryspace.samplers import (
    GroupSampler,
    WalkModeEnum,
)
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.operators.discovery_space_manager import (
    DiscoverySpaceManager,
)
from orchestrator.schema.entityspace import EntitySpaceRepresentation
from orchestrator.schema.property import ConstitutiveProperty


@pytest.fixture(params=[WalkModeEnum.RANDOM, WalkModeEnum.SEQUENTIAL])
def walk_mode(request: pytest.FixtureRequest) -> WalkModeEnum:

    return request.param


@pytest.fixture
def explicit_entity_space(
    constitutive_property_configuration_general: list[ConstitutiveProperty],
) -> EntitySpaceRepresentation:

    return EntitySpaceRepresentation(constitutive_property_configuration_general)


def check_group_order(
    sampler: GroupSampler,
    group_order: list[frozenset[tuple[str, Any]]],
    space: DiscoverySpace,
    group: list[str],
) -> None:

    # For selectors the sequential order depends on the order returned by the samplestore which may differ
    # for the same data across samplestores e.g. MySQL based SQL store performs sorts on primary keys, sqlite does not
    # For generators the sequential order depends on EntitySpaceRepresentation.sequential_point_iterator

    if isinstance(sampler, ExplicitEntitySpaceGroupedGridSampleGenerator):
        ids = [cp.identifier for cp in space.entitySpace.constitutiveProperties]
        points = [
            dict(zip(ids, p, strict=True))
            for p in space.entitySpace.sequential_point_iterator()
        ]
        groups = _build_groups_dict(points=points, group=group)
        expected_group_order = list(groups.keys())
        if sampler.mode == WalkModeEnum.RANDOM:
            assert group_order != expected_group_order
        else:

            assert group_order == expected_group_order
    else:
        points = _get_space_matching_points(discovery_space=space)
        groups = _build_groups_dict(points=points, group=group)
        expected_group_order = list(groups.keys())
        if isinstance(sampler, SequentialGroupSampleSelector):
            assert group_order == expected_group_order
        else:
            assert group_order != expected_group_order


@pytest.fixture(
    params=[
        "sequential_selector",
        "random_selector",
        "random_generator",
        "sequential_generator",
    ]
)
def group_sampler_ml_multi_cloud_space(
    request: pytest.FixtureRequest,
) -> (
    SequentialGroupSampleSelector
    | RandomGroupSampleSelector
    | ExplicitEntitySpaceGroupedGridSampleGenerator
    | None
):

    samp = None
    if request.param == "sequential_selector":
        samp = SequentialGroupSampleSelector(group=["nodes", "cpu_family"])
    if request.param == "random_selector":
        samp = RandomGroupSampleSelector(group=["nodes", "cpu_family"])
    if request.param == "sequential_generator":
        samp = ExplicitEntitySpaceGroupedGridSampleGenerator(
            group=["nodes", "cpu_family"], mode=WalkModeEnum.SEQUENTIAL
        )
    if request.param == "random_generator":
        samp = ExplicitEntitySpaceGroupedGridSampleGenerator(
            group=["nodes", "cpu_family"], mode=WalkModeEnum.RANDOM
        )

    return samp


def test_group_sampler_local(
    group_sampler_ml_multi_cloud_space: GroupSampler,
    ml_multi_cloud_space: DiscoverySpace,
) -> None:

    sampler = group_sampler_ml_multi_cloud_space
    space = ml_multi_cloud_space
    assert (
        space.sample_store.numberOfEntities == 42
    ), "Expected 42 entities in ml cloud sample store"

    count = 0
    i = 0
    group = []
    # Test Local Group iterator
    group_order = []
    for i, group in enumerate(sampler.entityGroupIterator(space)):
        count += len(group)
        for entity in group:
            print(i, count, entity)

        node_value = {
            (e.valueForConstitutivePropertyIdentifier("nodes").value) for e in group
        }
        cpu_value = {
            (e.valueForConstitutivePropertyIdentifier("cpu_family").value)
            for e in group
        }

        assert (
            len(node_value) == 1
        ), "Expected all entities in group to have same value for nodes property"
        assert (
            len(cpu_value) == 1
        ), "Expected all entities in group to have same value for cpu_family property"

        group_order.append(
            frozenset([("nodes", node_value.pop()), ("cpu_family", cpu_value.pop())])
        )

    # Generators versus Selectors: There will be a different number of entities iterated
    if isinstance(sampler, ExplicitEntitySpaceGroupedGridSampleGenerator):
        assert (
            count == space.entitySpace.size
        ), "Expected for generators that the number of entities iterated is equal to size of entity space"
    else:
        assert count == len(
            space.matchingEntities()
        ), "Expected for selectors that the number of entities iterated is equal to number matching entities in source"

    check_group_order(
        sampler=sampler,
        group_order=group_order,
        space=space,
        group=["nodes", "cpu_family"],
    )

    # Note: different to test_group_sampler_remote assertion as here i is from enumerate() so starts at 0
    assert i == 8 - 1, "Expected 8 groups over nodes+cpu_family"


def test_group_sampler_sequential_local(
    group_sampler_ml_multi_cloud_space: GroupSampler,
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    sampler = group_sampler_ml_multi_cloud_space
    space = ml_multi_cloud_space
    assert (
        space.sample_store.numberOfEntities == 42
    ), "Expected 42 entities in ml cloud sample store"

    count = 0
    entities = []
    all_entities = []
    for entities in sampler.entityIterator(space, batchsize=5):
        count += len(entities)
        all_entities.extend(entities)

    assert len(entities) != 5, "Expected the last batch not to be equal to batchsize"

    assert len(list({e.identifier for e in all_entities})) == len(
        [e.identifier for e in all_entities]
    ), "Expected entities iterated over to be unique"

    # Generators versus Selectors: There will be a different number of entities iterated
    if isinstance(sampler, ExplicitEntitySpaceGroupedGridSampleGenerator):
        assert (
            count == space.entitySpace.size
        ), "Expected for generators that the number of entities iterated is equal to size of entity space"
    else:
        assert count == len(
            space.matchingEntities()
        ), "Expected for selectors that the number of entities iterated is equal to number matching entities in source"


@pytest.mark.asyncio
async def test_group_sampler_remote(
    group_sampler_ml_multi_cloud_space: GroupSampler,
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    sampler = group_sampler_ml_multi_cloud_space
    space = ml_multi_cloud_space
    assert (
        space.sample_store.numberOfEntities == 42
    ), "Expected 42 entities in ml cloud sample store"

    # Test Remote Sequential Iterator
    queue = MeasurementQueue()
    manager = DiscoverySpaceManager.remote(space=space, queue=queue)
    assert sampler.samplerCompatibleWithDiscoverySpaceRemote(manager)

    iterator = await sampler.remoteEntityGroupIterator(remoteDiscoverySpace=manager)

    count = 0
    group_count = 0
    group = []
    group_order = []
    async for group in iterator:
        count += len(group)
        group_count += 1
        node_value = {
            (e.valueForConstitutivePropertyIdentifier("nodes").value) for e in group
        }
        cpu_value = {
            (e.valueForConstitutivePropertyIdentifier("cpu_family").value)
            for e in group
        }

        assert (
            len(node_value) == 1
        ), "Expected all entities in group to have same value for nodes property"
        assert (
            len(cpu_value) == 1
        ), "Expected all entities in group to have same value for cpu_family property"

        group_order.append(
            frozenset([("nodes", node_value.pop()), ("cpu_family", cpu_value.pop())])
        )

    # Generators versus Selectors: There will be a different number of entities iterated
    if isinstance(sampler, ExplicitEntitySpaceGroupedGridSampleGenerator):
        assert (
            count == space.entitySpace.size
        ), "Expected for generators that the number of entities iterated is equal to size of entity space"
    else:
        assert count == len(
            space.matchingEntities()
        ), "Expected for selectors that the number of entities iterated is equal to number matching entities in source"

    assert group_count == 8, "Expected 8 groups over nodes+cpu_family"

    check_group_order(
        sampler=sampler,
        group_order=group_order,
        space=space,
        group=["nodes", "cpu_family"],
    )


@pytest.mark.asyncio
async def test_group_sampler_sequential_remote(
    group_sampler_ml_multi_cloud_space: GroupSampler,
    ml_multi_cloud_space: DiscoverySpace,
) -> None:

    sampler = group_sampler_ml_multi_cloud_space
    space = ml_multi_cloud_space
    assert (
        space.sample_store.numberOfEntities == 42
    ), "Expected 42 entities in ml cloud sample store"

    # Test Remote Sequential Iterator
    queue = MeasurementQueue()
    manager = DiscoverySpaceManager.remote(space=space, queue=queue)
    assert RandomGroupSampleSelector.samplerCompatibleWithDiscoverySpaceRemote(manager)

    iterator = await sampler.remoteEntityIterator(
        remoteDiscoverySpace=manager,
        batchsize=5,
    )

    count = 0
    all_entities = []
    entities = []
    async for entities in iterator:
        assert len(entities) <= 5
        all_entities.extend(entities)
        count += len(entities)

    assert len(entities) != 5, "Expected the last batch not to be equal to batchsize"
    assert len(list({e.identifier for e in all_entities})) == len(
        [e.identifier for e in all_entities]
    ), "Expected entities iterated over to be unique"

    # Generators versus Selectors: There will be a different number of entities iterated
    if isinstance(sampler, ExplicitEntitySpaceGroupedGridSampleGenerator):
        assert (
            count == space.entitySpace.size
        ), "Expected for generators that the number of entities iterated is equal to size of entity space"
    else:
        assert count == len(
            space.matchingEntities()
        ), "Expected for selectors that the number of entities iterated is equal to number matching entities in source"


@pytest.mark.asyncio
async def test_group_sample_generator_fail_on_continuous_space() -> None:

    pytest.xfail(
        "We don't have a fixture for a discoveryspace with a continuous dimension"
    )
