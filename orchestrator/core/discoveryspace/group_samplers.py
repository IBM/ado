# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any

import numpy as np

from orchestrator.core.discoveryspace.samplers import (
    ExplicitEntitySpaceGridSampleGenerator,
    GroupSampler,
    WalkModeEnum,
)
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager
from orchestrator.schema.entity import Entity

moduleLog = logging.getLogger("groupsamplers")


def _get_space_points(discovery_space: DiscoverySpace) -> list[dict]:
    """
    Building list of points for a discovery space

    :param discovery_space: discovery space
    :return: list of points
    """
    entity_space = discovery_space.entitySpace
    property_names = [c.identifier for c in entity_space.constitutiveProperties]
    return [
        dict(zip(property_names, point, strict=True))
        for point in entity_space.sequential_point_iterator()
    ]


def _get_space_matching_points(discovery_space: DiscoverySpace) -> list[dict]:
    """
    Building list of points from the matching entities for a discovery space

    :param discovery_space: discovery space
    :return: list of points
    """

    return [
        {v.property.identifier: v.value for v in entity.constitutive_property_values}
        for entity in discovery_space.matchingEntities()
    ]


def _build_point_group_values(
    point: dict, group: list[str]
) -> frozenset[tuple[str, Any]]:
    """
    :return: A frozen set of (key,value) pairs
    """

    return frozenset({(k, v) for k, v in point.items() if k in group})


def _build_groups_dict(
    points: list[dict], group: list[str]
) -> dict[frozenset[tuple[str, Any]], list[dict]]:
    """
    builds a dict of lists of entities, combining entities based on group definitions
    :param group: group definition
    :return: A dictionary whose keys are groups and whose values are list of entities
    """
    groups = defaultdict(list)
    for point in points:
        grp = _build_point_group_values(point=point, group=group)
        groups[grp].append(point)

    return groups


def _build_groups_list(points: list[dict], group: list[str]) -> list[list[dict]]:
    """
    builds a list of lists of points, combining entities based on group definitions
    :param points: list of points
    :param group: group definition
    :return:
    """
    return list(_build_groups_dict(points=points, group=group).values())


async def _get_grouped_sample_async(
    generator: AsyncGenerator[list[Entity], None],
) -> list[Entity] | None:
    try:
        return await anext(generator)
    except (StopAsyncIteration, StopIteration):
        return None


def _get_grouped_sample(
    generator: Generator[list[Entity], None, None],
) -> list[Entity] | None:
    try:
        return next(generator)
    except (StopAsyncIteration, StopIteration):
        return None


async def _sequential_iterator_async(
    points: list[dict],
    group: list[str],
    remote_discovery_space: DiscoverySpaceManager,
) -> AsyncGenerator[list[Entity], None]:
    """
    Sequential iterator through discovery space with grouping
    :param points: list of points
    :param group: group definition
    :return:
    """
    group_list = _build_groups_list(points=points, group=group)
    for i in range(len(group_list)):
        entity_list_refs = [
            remote_discovery_space.entity_for_point.remote(point)
            for point in group_list[i]
        ]
        entity_list = await asyncio.gather(*entity_list_refs)
        yield entity_list


def _sequential_iterator(
    points: list[dict],
    group: list[str],
    discovery_space: DiscoverySpace,
) -> Generator[list[Entity], None, None]:
    """
    Sequential iterator through discovery space with grouping
    :param points: list of points
    :param group: group definition
    :return:
    """
    group_list = _build_groups_list(points=points, group=group)
    for i in range(len(group_list)):
        entity_list = [
            discovery_space.entity_for_point(point) for point in group_list[i]
        ]
        yield entity_list


async def _random_iterator_async(
    points: list[dict],
    group: list[str],
    remote_discovery_space: DiscoverySpaceManager,
) -> AsyncGenerator[list[Entity], None]:
    """
    Random iterator through discovery space with grouping
    :param points: list of points
    :param group: group definition
    :return:
    """
    group_list = _build_groups_list(points=points, group=group)
    randomized = np.random.default_rng().choice(
        a=range(len(group_list)), size=len(group_list), replace=False
    )
    for i in range(len(randomized)):
        entity_list_refs = [
            remote_discovery_space.entity_for_point.remote(point)
            for point in group_list[randomized[i]]
        ]
        entity_list = await asyncio.gather(*entity_list_refs)
        yield entity_list


def _random_iterator(
    points: list[dict],
    group: list[str],
    discovery_space: DiscoverySpace,
) -> Generator[list[Entity], None, None]:
    """
    Random iterator through discovery space with grouping
    :param points: list of points
    :param group: group definition
    :return:
    """
    group_list = _build_groups_list(points=points, group=group)
    randomized = np.random.default_rng().choice(
        a=range(len(group_list)), size=len(group_list), replace=False
    )
    for i in range(len(randomized)):
        entity_list = [
            discovery_space.entity_for_point(point)
            for point in group_list[randomized[i]]
        ]
        yield entity_list


def _sequential_group_iterator(
    generator: Generator[list[dict], None, None],
    batch_size: int,
) -> Generator[list[Entity], None, None]:
    """
    Sequential group iterator
    :param generator: grouped iterator
    :param batch_size: batch size
    :return:
    """
    sample = []
    batch = []
    done = False
    # loop while not done
    while not done:
        # loop through the batch size
        for _ in range(batch_size):
            if len(sample) == 0:
                # get the new group
                sample = _get_grouped_sample(generator=generator)
                if sample is None:
                    # no more data
                    # mark that we are done and break
                    done = True
                    break

            # append a new entity to the batch
            entity = sample.pop(0)
            batch.append(entity)
        # submit a batch and clean it up
        # The last batch may be empty - if so don't return it
        if batch:
            yield batch
        batch.clear()


async def _sequential_group_iterator_async(
    generator: AsyncGenerator[list[dict], None],
    batch_size: int,
) -> AsyncGenerator[list[Entity], None]:
    """
    Async sequential group iterator
    :param generator: grouped iterator
    :param batch_size: batch size
    :return:
    """
    sample = []
    batch = []
    done = False
    # loop while not done
    while not done:
        # loop through the batch size
        for _ in range(batch_size):
            if len(sample) == 0:
                # get the new group
                sample = await _get_grouped_sample_async(generator=generator)
                if sample is None:
                    # no more data
                    # mark that we are done
                    done = True
                    break

            # append a new entity to the batch
            entity = sample.pop(0)
            batch.append(entity)
        # submit a batch and clean it up
        # The last batch may be empty - if so don't return it
        if batch:
            yield batch
        batch.clear()


class SequentialGroupSampleSelector(GroupSampler):
    """
    This class sequentially selects groups of entities, that can/should be processed together
    """

    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls, remoteDiscoverySpace: DiscoverySpaceManager
    ) -> bool:
        return True

    def __init__(self, group: list[str]) -> None:
        """
        Creates sampler based on group of variables that should have the same values
        :param group: List of variable names that should have the same values
        """
        self.group = group

    def entityGroupIterator(
        self,
        discoverySpace: DiscoverySpace,
    ) -> Generator[list[Entity], None, None]:
        """Returns an iterator that samples groups of entities from a discovery space

        The group definition should be specified on initializing an instance of a subclass of this class

        Note: The number of entities returned on each call to the iterator can vary as it depends on
        the number of members of the associated group

        Parameters:
            discoverySpace: An orchestrator.model.space.DiscoverySpace instance
        """
        points = _get_space_matching_points(discovery_space=discoverySpace)
        return _sequential_iterator(
            points=points, group=self.group, discovery_space=discoverySpace
        )

    async def remoteEntityGroupIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager
    ) -> AsyncGenerator[list[Entity], None]:
        async def iterator_closure() -> (
            Callable[[], AsyncGenerator[list[Entity], None]]
        ):
            discovery_space = await remoteDiscoverySpace.discoverySpace.remote()
            points = _get_space_matching_points(discovery_space=discovery_space)
            return _sequential_iterator_async(
                points=points,
                group=self.group,
                remote_discovery_space=remoteDiscoverySpace,
            )

        return await iterator_closure()

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize: int = 1
    ) -> Generator[list[Entity], None, None]:
        grouped_iterator = self.entityGroupIterator(discoverySpace=discoverySpace)
        return _sequential_group_iterator(
            generator=grouped_iterator,
            batch_size=batchsize,
        )

    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize: int = 1
    ) -> AsyncGenerator[list[Entity], None]:
        grouped_iterator = await self.remoteEntityGroupIterator(
            remoteDiscoverySpace=remoteDiscoverySpace
        )
        return _sequential_group_iterator_async(
            generator=grouped_iterator,
            batch_size=batchsize,
        )


class RandomGroupSampleSelector(GroupSampler):
    """
    This class sequentially selects groups of entities, that can/should be processed together
    """

    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls, remoteDiscoverySpace: DiscoverySpaceManager
    ) -> bool:
        return True

    def __init__(self, group: list[str]) -> None:
        """
        Creates sampler based on group of variables that should have the same values
        :param group: List of variable names that should have the same values
        """
        self.group = group

    def entityGroupIterator(
        self,
        discoverySpace: DiscoverySpace,
    ) -> Generator[list[Entity], None, None]:
        """Returns an iterator that samples groups of entities from a discovery space

        The group definition should be specified on initializing an instance of a subclass of this class

        Note: The number of entities returned on each call to the iterator can vary as it depends on
        the number of members of the associated group

        Parameters:
            discoverySpace: An orchestrator.model.space.DiscoverySpace instance
        """
        points = _get_space_matching_points(discovery_space=discoverySpace)
        return _random_iterator(
            points=points, group=self.group, discovery_space=discoverySpace
        )

    async def remoteEntityGroupIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager
    ) -> AsyncGenerator[list[Entity], None]:
        async def iterator_closure(
            remote_discovery_space: DiscoverySpaceManager,
        ) -> Callable[[], AsyncGenerator[list[Entity], None]]:
            discovery_space = await remote_discovery_space.discoverySpace.remote()
            points = _get_space_matching_points(discovery_space=discovery_space)
            return _random_iterator_async(
                points=points,
                group=self.group,
                remote_discovery_space=remoteDiscoverySpace,
            )

        return await iterator_closure(remote_discovery_space=remoteDiscoverySpace)

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize: int = 1
    ) -> Generator[list[Entity], None, None]:
        grouped_iterator = self.entityGroupIterator(discoverySpace=discoverySpace)
        return _sequential_group_iterator(
            generator=grouped_iterator,
            batch_size=batchsize,
        )

    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize: int = 1
    ) -> AsyncGenerator[list[Entity], None]:
        grouped_iterator = await self.remoteEntityGroupIterator(
            remoteDiscoverySpace=remoteDiscoverySpace
        )
        return _sequential_group_iterator_async(
            generator=grouped_iterator,
            batch_size=batchsize,
        )


class ExplicitEntitySpaceGroupedGridSampleGenerator(
    ExplicitEntitySpaceGridSampleGenerator, GroupSampler
):
    """Samples an explicit entity space as a grid

    Grid means the probability distribution associated with the dimensions is not used
    Here we are only overwriting remoteEntityIterator of the base implementation
    """

    def __init__(self, mode: WalkModeEnum, group: list[str]) -> None:
        """
        Initialization
        :param mode: operation mode - sequential, random, grouped
        :param group: The group
        """
        super().__init__(mode)
        self.group = group
        print(
            f"Initializing ExplicitEntitySpaceGroupedGridSampleGenerator, group: {group}"
        )

    def entityGroupIterator(
        self,
        discoverySpace: DiscoverySpace,
    ) -> Generator[list[Entity], None, None]:
        """Returns an iterator that samples groups of entities from a discovery space

        Note: The number of entities returned on each call to the iterator can vary as it depends on
        the number of members of the associated group

        Parameters:
            discoverySpace: An orchestrator.model.space.DiscoverySpace instance
        """

        entity_space = discoverySpace.entitySpace

        if not ExplicitEntitySpaceGroupedGridSampleGenerator.samplerCompatibleWithEntitySpace(
            entity_space
        ):
            raise ValueError(
                f"Cannot use ExplicitEntitySpaceGroupedGridSampleGenerator with {entity_space}"
            )

        points = _get_space_points(discovery_space=discoverySpace)

        def iterator_closure() -> Generator[list[Entity], None, None]:
            def sequential_iterator() -> Generator[list[Entity], None, None]:
                return _sequential_iterator(
                    points=points, group=self.group, discovery_space=discoverySpace
                )

            def random_iterator() -> Generator[list[Entity], None, None]:
                return _random_iterator(
                    points=points, group=self.group, discovery_space=discoverySpace
                )

            if self.mode == WalkModeEnum.SEQUENTIAL:
                return sequential_iterator()
            return random_iterator()

        return iterator_closure()

    async def remoteEntityGroupIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager
    ) -> AsyncGenerator[list[Entity], None]:
        """Returns an async iterator that returns groups of entities as defined by the instances group property"""

        async def iterator_closure(
            remote_discovery_space: DiscoverySpaceManager,
        ) -> AsyncGenerator[list[Entity], None]:

            # noinspection PyUnresolvedReferences
            entity_space = await remote_discovery_space.entitySpace.remote()
            discovery_space = await remote_discovery_space.discoverySpace.remote()
            points = _get_space_points(discovery_space=discovery_space)

            if not ExplicitEntitySpaceGroupedGridSampleGenerator.samplerCompatibleWithEntitySpace(
                entity_space
            ):
                raise ValueError(
                    f"Cannot use ExplicitEntitySpaceGroupedGridSampleGenerator with {entity_space}"
                )

            def sequential_iterator() -> AsyncGenerator[list[Entity], None]:
                return _sequential_iterator_async(
                    points=points,
                    group=self.group,
                    remote_discovery_space=remoteDiscoverySpace,
                )

            def random_iterator() -> AsyncGenerator[list[Entity], None]:
                return _random_iterator_async(
                    points=points,
                    group=self.group,
                    remote_discovery_space=remoteDiscoverySpace,
                )

            if self.mode == WalkModeEnum.SEQUENTIAL:
                return sequential_iterator()
            return random_iterator()

        return await iterator_closure(remoteDiscoverySpace)

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize: int = 1
    ) -> Generator[list[Entity], None, None]:
        """Returns an iterator over a sequence of entities ordered by group"""
        grouped_iterator = self.entityGroupIterator(discoverySpace=discoverySpace)
        return _sequential_group_iterator(
            generator=grouped_iterator,
            batch_size=batchsize,
        )

    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize: int = 1
    ) -> AsyncGenerator[list[Entity], None]:
        grouped_iterator = await self.remoteEntityGroupIterator(
            remoteDiscoverySpace=remoteDiscoverySpace
        )
        return _sequential_group_iterator_async(
            generator=grouped_iterator,
            batch_size=batchsize,
        )
