# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
from collections.abc import AsyncGenerator, Generator
from typing import Any

import numpy as np
import ray

from orchestrator.core.discoveryspace.samplers import (
    ExplicitEntitySpaceGridSampleGenerator,
    GroupSampler,
    WalkModeEnum,
)
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager
from orchestrator.schema.entity import Entity

moduleLog = logging.getLogger("groupsamplers")


def _build_entity_group_values(
    entity: Entity, group: list[str]
) -> frozenset[tuple[str, Any]]:
    """
    :return: A frozen set of (key,value) paris
    """
    # build a dictionary of entity values given the group
    return frozenset(
        {
            (v.property.identifier, v.value)
            for v in entity.constitutive_property_values
            if v.property.identifier in group
        }
    )


def _build_groups_dict(
    entities: list[Entity], group: list[str]
) -> dict[frozenset[tuple[str, Any]], list[Entity]]:
    """
    builds a dict of lists of entities, combining entities based on group definitions
    :param entities: list of entities
    :param group: group definition
    :return: A dictionary whose keys are groups and whose values are list of entities
    """
    groups = {}
    for element in entities:
        grp = _build_entity_group_values(entity=element, group=group)
        lst = groups.get(grp, [])
        lst.append(element)
        groups[grp] = lst

    return groups


def _build_groups_list(entities: list[Entity], group: list[str]) -> list[list[Entity]]:
    """
    builds a list of lists of entities, combining entities based on group definitions
    :param entities: list of entities
    :param group: group definition
    :return:
    """

    return list(_build_groups_dict(entities=entities, group=group).values())


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
    entities: list[Entity], group: list[str]
) -> AsyncGenerator[list[Entity], None]:
    """
    Sequential iterator through discovery space with grouping
    :param entities: list of entities
    :param group: group definition
    :return:
    """
    group_list = _build_groups_list(entities=entities, group=group)
    for i in range(len(group_list)):
        lst = group_list[i]
        yield lst


def _sequential_iterator(
    entities: list[Entity], group: list[str]
) -> Generator[list[Entity], None, None]:
    """
    Sequential iterator through discovery space with grouping
    :param entities: list of entities
    :param group: group definition
    :return:
    """
    group_list = _build_groups_list(entities=entities, group=group)
    for i in range(len(group_list)):
        lst = group_list[i]
        yield lst


async def _random_iterator_async(
    entities: list[Entity], group: list[str]
) -> AsyncGenerator[list[Entity], None]:
    """
    Random iterator through discovery space with grouping
    :param entities: list of entities
    :param group: group definition
    :return:
    """
    group_list = _build_groups_list(entities=entities, group=group)
    randomized = np.random.choice(
        a=range(len(group_list)), size=len(group_list), replace=False
    )
    for i in range(len(randomized)):
        lst = group_list[randomized[i]]
        yield lst


def _random_iterator(
    entities: list[Entity], group: list[str]
) -> Generator[list[Entity], None, None]:
    """
    Random iterator through discovery space with grouping
    :param entities: list of entities
    :param group: group definition
    :return:
    """
    group_list = _build_groups_list(entities=entities, group=group)
    randomized = np.random.choice(
        a=range(len(group_list)), size=len(group_list), replace=False
    )
    for i in range(len(randomized)):
        lst = group_list[randomized[i]]
        yield lst


def _sequential_group_iterator(
    generator: Generator[list[Entity], None, None], batchsize: int
) -> Generator[list[Entity], None, None]:
    """
    Sequential group iterator
    :param generator: grouped iterator
    :param batchsize: batch size
    :return:
    """
    sample = []
    batch = []
    done = False
    # loop while not done
    while not done:
        # loop through the batch size
        for _ in range(batchsize):
            if len(sample) == 0:
                # get the new group
                sample = _get_grouped_sample(generator=generator)
                if sample is None:
                    # no more data
                    # mark that we are done and break
                    done = True
                    break
            # append a new entity to batch
            batch.append(sample[0])
            # remove entity from samples
            sample = sample[1:]
        # submit a batch and clean it up
        # The last batch may be empty - if so don't return it
        if batch:
            yield batch
        batch.clear()


async def _sequential_group_iterator_async(
    generator: AsyncGenerator[list[Entity], None], batchsize: int
) -> AsyncGenerator[list[Entity], None]:
    """
    Async sequential group iterator
    :param generator: grouped iterator
    :param batchsize: batch size
    :return:
    """
    sample = []
    batch = []
    done = False
    # loop while not done
    while not done:
        # loop through the batch size
        for _ in range(batchsize):
            if len(sample) == 0:
                # get the new group
                sample = await _get_grouped_sample_async(generator=generator)
                # print(f"getting a new group of length {len(sample)}")
                if sample is None:
                    # no more data
                    # mark that we are done
                    done = True
                    break
            # append a new entity to batch
            batch.append(sample[0])
            # remove entity from samples
            sample = sample[1:]
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
    ):
        return True

    def __init__(self, group: list[str]):
        """
        Creates sampler based on group of variables that should have the same values
        :param group: List of variable names that should have the same values
        """
        self.group = group

    def entityGroupIterator(
        self,
        discoverySpace: DiscoverySpace,
    ) -> Generator[list[Entity], None, None]:
        """Returns an iterator  that samples groups of entities from a discoveryspace

        The group definition should be specified on initializing an instance of a subclass of this class

        Note: The number of entities returned on each call to the iterator can vary as it depends on
        the number of members of the associated group

        Parameters:
            discoverySpace: An orchestrator.model.space.DiscoverySpace instance
        """
        entities = discoverySpace.matchingEntities()
        return _sequential_iterator(entities=entities, group=self.group)

    async def remoteEntityGroupIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager
    ) -> AsyncGenerator[list[Entity], None]:
        async def iterator_closure():
            entities = await remoteDiscoverySpace.matchingEntitiesInSource.remote()
            return _sequential_iterator_async(
                entities=entities,
                group=self.group,
            )

        return await iterator_closure()

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize=1
    ) -> Generator[list[Entity], None, None]:
        grouped_iterator = self.entityGroupIterator(discoverySpace=discoverySpace)
        return _sequential_group_iterator(
            generator=grouped_iterator, batchsize=batchsize
        )

    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize=1
    ) -> AsyncGenerator[list[Entity], None]:
        grooped_iterator = await self.remoteEntityGroupIterator(
            remoteDiscoverySpace=remoteDiscoverySpace
        )
        return _sequential_group_iterator_async(
            generator=grooped_iterator, batchsize=batchsize
        )


class RandomGroupSampleSelector(GroupSampler):
    """
    This class sequentially selects groups of entities, that can/should be processed together
    """

    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls, remoteDiscoverySpace: DiscoverySpaceManager
    ):
        return True

    def __init__(self, group: list[str]):
        """
        Creates sampler based on group of variables that should have the same values
        :param group: List of variable names that should have the same values
        """
        self.group = group

    def entityGroupIterator(
        self,
        discoverySpace: DiscoverySpace,
    ) -> Generator[list[Entity], None, None]:
        """Returns an iterator  that samples groups of entities from a discoveryspace

        The group definition should be specified on initializing an instance of a subclass of this class

        Note: The number of entities returned on each call to the iterator can vary as it depends on
        the number of members of the associated group

        Parameters:
            discoverySpace: An orchestrator.model.space.DiscoverySpace instance
        """
        entities = discoverySpace.matchingEntities()
        return _random_iterator(entities=entities, group=self.group)

    async def remoteEntityGroupIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager
    ) -> AsyncGenerator[list[Entity], None]:
        async def iterator_closure(
            stateHandle: DiscoverySpaceManager,
        ):
            entities = await stateHandle.matchingEntitiesInSource.remote()
            return _random_iterator_async(entities=entities, group=self.group)

        return await iterator_closure(stateHandle=remoteDiscoverySpace)

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize=1
    ) -> Generator[list[Entity], None, None]:
        grouped_iterator = self.entityGroupIterator(discoverySpace=discoverySpace)
        return _sequential_group_iterator(
            generator=grouped_iterator, batchsize=batchsize
        )

    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize=1
    ) -> AsyncGenerator[list[Entity], None]:
        grouped_iterator = await self.remoteEntityGroupIterator(
            remoteDiscoverySpace=remoteDiscoverySpace
        )
        return _sequential_group_iterator_async(
            generator=grouped_iterator, batchsize=batchsize
        )


class ExplicitEntitySpaceGroupedGridSampleGenerator(
    ExplicitEntitySpaceGridSampleGenerator, GroupSampler
):
    """Samples an explicit entity space as a grid

    Grid means the probability distribution associated with the dimensions is not used
    Here we are only overwriting remoteEntityIterator of the base implementation
    """

    def __init__(self, mode: WalkModeEnum, group: list[str]):
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

    def _get_remote_space_entities(
        self, discoverySpaceActor: DiscoverySpaceManager
    ) -> list[Entity]:
        """
        Building list of entities for a discovery space

        :param discoverySpaceActor: discovery space actor
        :return: list of entities
        """
        # get discovery space
        # noinspection PyUnresolvedReferences
        dspace = ray.get(discoverySpaceActor.discoverySpace.remote())
        # build list of entities
        return self._get_space_entities(discoverySpace=dspace)

    def _get_space_entities(self, discoverySpace: DiscoverySpace) -> list[Entity]:
        """
        Building list of entities for a discovery space

        :param discoverySpace: discovery space
        :return: list of entities
        """
        # get entity space
        entity_space = discoverySpace.entitySpace
        # create sampler generator
        self.samplerCompatibleWithEntitySpace(entity_space)
        # create iterator
        iterator = super().entityIterator(discoverySpace, batchsize=1)
        # get entities
        entity_list = []
        for e in iterator:
            entity_list.append(e[0])
        return entity_list

    def entityGroupIterator(
        self,
        discoverySpace: DiscoverySpace,
    ) -> Generator[list[Entity], None, None]:
        """Returns an iterator  that samples groups of entities from a discoveryspace

        Note: The number of entities returned on each call to the iterator can vary as it depends on
        the number of members of the associated group

        Parameters:
            discoverySpace: An orchestrator.model.space.DiscoverySpace instance
        """

        def iterator_closure() -> Generator[list[Entity], None, None]:
            def sequential_iterator() -> Generator[list[Entity], None, None]:
                entities = self._get_space_entities(discoverySpace=discoverySpace)
                return _sequential_iterator(entities=entities, group=self.group)

            def random_iterator() -> Generator[list[Entity], None, None]:
                entities = self._get_space_entities(discoverySpace=discoverySpace)
                return _random_iterator(entities=entities, group=self.group)

            if self.mode == WalkModeEnum.SEQUENTIAL:
                return sequential_iterator()
            return random_iterator()

        return iterator_closure()

    async def remoteEntityGroupIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager
    ) -> AsyncGenerator[list[Entity], None]:
        """Returns an async iterator that returns groups of entities as defined by the instances group property"""

        async def iterator_closure(
            spaceActor: DiscoverySpaceManager,
        ) -> AsyncGenerator[list[Entity], None]:

            # noinspection PyUnresolvedReferences
            entitySpace = await spaceActor.entitySpace.remote()
            # noinspection PyUnresolvedReferences
            measurementSpace = await spaceActor.measurementSpace.remote()

            if not ExplicitEntitySpaceGroupedGridSampleGenerator.samplerCompatibleWithEntitySpace(
                entitySpace=entitySpace
            ):
                raise ValueError(
                    f"Cannot use ExplicitEntitySpaceGroupedGridSampleGenerator with {entitySpace}"
                )

            observedProperties = []
            for experiment in measurementSpace.experiments:
                observedProperties.extend(experiment.observedProperties)

            def sequential_iterator() -> AsyncGenerator[list[Entity], None]:
                entities = self._get_remote_space_entities(
                    discoverySpaceActor=spaceActor
                )
                return _sequential_iterator_async(
                    entities=entities,
                    group=self.group,
                )

            def random_iterator() -> AsyncGenerator[list[Entity], None]:
                entities = self._get_remote_space_entities(
                    discoverySpaceActor=spaceActor
                )
                return _random_iterator_async(
                    entities=entities,
                    group=self.group,
                )

            if self.mode == WalkModeEnum.SEQUENTIAL:
                return sequential_iterator()
            return random_iterator()

        return await iterator_closure(remoteDiscoverySpace)

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize=1
    ) -> Generator[list[Entity], None, None]:
        """Returns an iterator over a sequence of entities ordered by group"""
        grouped_iterator = self.entityGroupIterator(discoverySpace=discoverySpace)
        return _sequential_group_iterator(
            generator=grouped_iterator, batchsize=batchsize
        )

    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize=1
    ) -> AsyncGenerator[list[Entity], None]:
        grouped_iterator = await self.remoteEntityGroupIterator(
            remoteDiscoverySpace=remoteDiscoverySpace
        )
        return _sequential_group_iterator_async(
            generator=grouped_iterator, batchsize=batchsize
        )
