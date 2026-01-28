# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import abc
import enum
import logging
import typing

import numpy as np
import pydantic
import ray

from orchestrator.core.discoveryspace.space import (
    DiscoverySpace,
)
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager
from orchestrator.schema.entity import Entity
from orchestrator.schema.entityspace import EntitySpaceRepresentation

"""Samplers are used to sample entities from a discovery space.

There are two classes of samplers - SampleSelectors and SampleGenerators

SampleSelectors: Select samples from the discovery space's sample store (already measured entities in the space).
SampleGenerators: These generate new entities from the entity space

All discovery spaces are compatible with SampleSelectors.
The ability to use a SampleGenerator depends on specific features of the discovery space e.g. TransformerModelSampleGenerator
"""

moduleLog = logging.getLogger("samplers")


class WalkModeEnum(enum.Enum):
    RANDOM = "random"  # Randomly select points in a space
    SEQUENTIAL = (
        "sequential"  # Iterate through points in a deterministic sequence if possible
    )


class SamplerTypeEnum(enum.Enum):
    SELECTOR = "selector"
    GENERATOR = "generator"


class BaseSamplerParameters(pydantic.BaseModel):

    mode: WalkModeEnum
    model_config = pydantic.ConfigDict(extra="forbid")


class BaseSampler(abc.ABC):
    """
    Samplers sample entities from a discovery space.

    A discoveryspace has two sources of entities that can be sampled from.
    A given sampler will use one of these which defines its type:

    - the entityspace of the discovery space (generators)
    - the samplestore (set of already measured entities) of the discoveryspace (selectors)
    """

    @classmethod
    @abc.abstractmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls, remoteDiscoverySpace: DiscoverySpaceManager
    ) -> bool:  # pragma: nocover
        """Return True if this remoteEntityIterator can be used with the given DiscoverySpace"""

    @abc.abstractmethod
    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize: int = 1
    ) -> typing.Generator[list[Entity], None, None]:  # pragma: nocover
        """Returns an iterator that samples entities from the discovery space in batchsize groups

        Parameters:
            discoverySpace: An orchestrator.model.space.DiscoverySpace instance
            batchsize: The iterators will return entities in batches of this size

        """

    @abc.abstractmethod
    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize: int = 1
    ) -> typing.AsyncGenerator[list[Entity], None]:  # pragma: nocover
        """Returns an async iterator that samples entities from an InternalState actor in batchsize groups

        Parameters:
            remoteDiscoverySpace: An InternalState actor instance providing access to the DiscoverySpace
            batchsize: The iterator will return entities in batches of this size

        """

    @classmethod
    def parameters_model(cls) -> type[pydantic.BaseModel] | None:
        """Returns a pydantic model for the init parameters of the class, if any"""

        return None


class GroupSampler(BaseSampler):
    """Provides additional capability to group entities and sample those groups"""

    @abc.abstractmethod
    def entityGroupIterator(
        self,
        discoverySpace: DiscoverySpace,
    ) -> typing.Generator[list[Entity], None, None]:  # pragma: nocover
        """Returns an iterator  that samples groups of entities from a discoveryspace

        The group definition should be specified on initializing an instance of a subclass of this class

        Note: The number of entities returned on each call to the iterator can vary as it depends on
        the number of members of the associated group

        Parameters:
            discoverySpace: An orchestrator.model.space.DiscoverySpace instance
        """

    @abc.abstractmethod
    async def remoteEntityGroupIterator(
        self,
        remoteDiscoverySpace: DiscoverySpaceManager,
    ) -> typing.AsyncGenerator[list[Entity], None]:  # pragma: nocover
        """Returns an async iterator  that samples groups of entities from an InternalState actor.

        The group definition should be specified on initializing an instance of a subclass of this class

        Note: The number of entities returned on each call to the iterator can vary as it depends on
        the number of members of the associated group

        Parameters:
           remoteDiscoverySpace: An InternalState actor instance providing access to the DiscoverySpace

        """


class RandomSampleSelector(BaseSampler):

    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls, remoteDiscoverySpace: DiscoverySpaceManager
    ) -> bool:
        return True

    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize: int = 1
    ) -> typing.AsyncGenerator[list[Entity], None]:
        """Returns an iterator that returns entities in a random order"""

        async def iterator_closure(
            stateHandle: DiscoverySpaceManager,
        ) -> typing.Callable[[], typing.AsyncGenerator[list[Entity], None]]:
            # noinspection PyUnresolvedReferences
            numberEntities = await stateHandle.numberOfMatchingEntitiesInSource.remote()
            walk = np.random.default_rng().choice(
                range(numberEntities), numberEntities, replace=False
            )

            async def iterator() -> typing.AsyncGenerator[list[Entity], None]:
                # Note: This does not suffer the same problem as the SequentialSampler when numberEntities may
                # not be equal to the actuator number of entities return by "entities"
                # Instead in this situation remoteEntityIterator method will raise IndexError when an element
                # between (actual number entities) and (expected number of entities) is walked to
                # This should stop the walk, although unexpectedly
                for walkIndex in range(0, numberEntities, batchsize):
                    selected = walk[walkIndex : walkIndex + batchsize]
                    # noinspection PyUnresolvedReferences
                    yield await stateHandle.matchingEntitiesInSource.remote(
                        selection=selected
                    )

            return iterator

        func = await iterator_closure(remoteDiscoverySpace)
        return func()

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize: int = 1
    ) -> typing.Generator[list[Entity], None, None]:
        """Returns an iterator that returns entities in a random order"""

        def iterator_closure(
            space: DiscoverySpace,
        ) -> typing.Callable[[], typing.Generator[list[Entity], None, None]]:
            entities = space.matchingEntities()
            numberEntities = len(space.matchingEntities())
            walk = np.random.default_rng().choice(
                range(numberEntities), numberEntities, replace=False
            )

            def iterator() -> typing.Generator[list[Entity], None, None]:
                # see note in remote iterator
                for walkIndex in range(0, numberEntities, batchsize):
                    selection = walk[walkIndex : walkIndex + batchsize]
                    selected = [entities[i] for i in selection]
                    yield selected

            return iterator

        func = iterator_closure(discoverySpace)
        return func()


class SequentialSampleSelector(BaseSampler):

    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls, remoteDiscoverySpace: DiscoverySpaceManager
    ) -> bool:
        return True

    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize: int = 1
    ) -> typing.AsyncGenerator[list[Entity], None]:
        """Returns an remoteEntityIterator that returns entities in order"""

        async def iterator_closure(
            stateHandle: DiscoverySpaceManager,
        ) -> typing.Callable[[], typing.AsyncGenerator[list[Entity], None]]:

            # Note: We rely on the return value of numberOfMatchingEntitiesInSource is the size of `matchingEntities`
            # However this may not be the case if, for example, some entities could not be retrieved from a
            # backend db.
            # noinspection PyUnresolvedReferences
            numberEntities = await stateHandle.numberOfMatchingEntitiesInSource.remote()

            async def iterator() -> typing.AsyncGenerator[list[Entity], None]:
                for i in range(0, numberEntities, batchsize):
                    # noinspection PyUnresolvedReferences
                    entities = await stateHandle.entitiesSlice.remote(
                        start=i, stop=i + batchsize
                    )
                    if len(entities) == 0:  # pragma: nocover
                        # This will happen if there are fewer entities available than numberEntities
                        # For example with DBs sometimes all the entities it can see are present cannot be retrieved
                        # That is, a COUNT statement returns all the rows, but the SELECT statement fails on some.
                        # If we do not check here then the remoteEntityIterator would return
                        # empty lists until numberEntities was reached
                        # Code expecting no empty lists from this method then break.
                        # We could enforce in the interface of sample_store that numberEntities must be the
                        # number of element in entities but this still may not be true so a check seems worth it.
                        break
                    else:
                        yield entities

            return iterator

        retval = await iterator_closure(remoteDiscoverySpace)
        return retval()

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize: int = 1
    ) -> typing.Generator[list[Entity], None, None]:
        """Returns an remoteEntityIterator that returns entities in order"""

        def iterator_closure(
            space: DiscoverySpace,
        ) -> typing.Callable[[], typing.Generator[list[Entity], None, None]]:

            # Note: We rely on the return value of numberOfMatchingEntitiesInSource is the size of `matchingEntities`
            # However this may not be the case if, for example, some entities could not be retrieved from a
            # backend db.
            numberEntities = len(space.matchingEntities())
            entities = space.matchingEntities()

            def iterator() -> typing.Generator[list[Entity], None, None]:
                for i in range(0, numberEntities, batchsize):
                    batch = entities[i : i + batchsize]
                    if len(batch) == 0:  # pragma: nocover
                        # This will happen if there are fewer entities available than numberEntities
                        # For example with DBs sometimes all the entities it can see are present cannot be retrieved
                        # That is, a COUNT statement returns all the rows, but the SELECT statement fails on some.
                        # If we do not check here then the remoteEntityIterator would return
                        # empty lists until numberEntities was reached
                        # Code expecting no empty lists from this method then break.
                        # We could enforce in the interface of sample_store that numberEntities must be the
                        # number of element in entities but this still may not be true so a check seems worth it.
                        break
                    else:
                        yield batch

            return iterator

        retval = iterator_closure(discoverySpace)
        return retval()


class ExplicitEntitySpaceGridSampleGenerator(BaseSampler):
    """Samples an explicit entity space as a grid

    Grid means the probability distribution associated with the dimensions is not used
    """

    @classmethod
    def samplerCompatibleWithEntitySpace(
        cls, entitySpace: EntitySpaceRepresentation
    ) -> bool:

        return bool(
            entitySpace is not None
            and isinstance(entitySpace, EntitySpaceRepresentation)
            and entitySpace.isDiscreteSpace
        )

    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls, remoteDiscoverySpace: DiscoverySpaceManager
    ) -> bool:

        # noinspection PyUnresolvedReferences
        entitySpace = ray.get(remoteDiscoverySpace.entitySpace.remote())
        return cls.samplerCompatibleWithEntitySpace(entitySpace)

    @classmethod
    def samplerCompatibleWithDiscoverySpace(
        cls, discoverySpace: DiscoverySpace
    ) -> bool:

        # noinspection PyUnresolvedReferences
        return cls.samplerCompatibleWithEntitySpace(discoverySpace.entitySpace)

    @classmethod
    def parameters_model(cls) -> type[pydantic.BaseModel] | None:

        return BaseSamplerParameters

    def __init__(self, mode: WalkModeEnum | BaseSamplerParameters) -> None:

        if isinstance(mode, BaseSamplerParameters):
            self.mode = mode.mode
        else:
            self.mode = mode

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize: int = 1
    ) -> typing.Generator[list[Entity], None, None]:
        """Returns an iterator over the entity space of the discovery space

        If Entities exist in the sample store of the discovery space they are returned.
        Otherwise, a new entity is created.

        The entity space must be an ExplicitEntitySpaceRepresentation"""

        def iterator_closure(
            discoverySpace: DiscoverySpace,
        ) -> typing.Callable[[], typing.Generator[list[Entity], None, None]]:

            entitySpace = discoverySpace.entitySpace

            if not ExplicitEntitySpaceGridSampleGenerator.samplerCompatibleWithEntitySpace(
                entitySpace=entitySpace
            ):
                raise ValueError(
                    f"Cannot use ExplicitEntitySpaceGridSampleGenerator with {entitySpace}"
                )

            def sequential_iterator() -> typing.Generator[list[Entity], None, None]:
                names = [c.identifier for c in entitySpace.constitutiveProperties]
                batch = []
                for point in entitySpace.sequential_point_iterator():
                    entity = discoverySpace.entity_for_point(
                        dict(zip(names, point, strict=True))
                    )
                    batch.append(entity)
                    if len(batch) == batchsize:
                        yield batch
                        batch = []

                if len(batch) != 0:
                    yield batch

            def random_iterator() -> typing.Generator[list[Entity], None, None]:

                names = [c.identifier for c in entitySpace.constitutiveProperties]
                batch = []
                for point in entitySpace.random_point_iterator():
                    entity = discoverySpace.entity_for_point(
                        dict(zip(names, point, strict=True))
                    )
                    batch.append(entity)
                    if len(batch) == batchsize:
                        yield batch
                        batch = []

                if len(batch) != 0:
                    yield batch

            return (
                sequential_iterator
                if self.mode == WalkModeEnum.SEQUENTIAL
                else random_iterator
            )

        retval = iterator_closure(discoverySpace)
        return retval()

    def entitySpaceIterator(
        self,
        entitySpace: EntitySpaceRepresentation,
        batchsize: int = 1,
    ) -> typing.Generator[list[Entity], None, None]:
        """Returns an iterator that iterates over an explicit entity space.

        Note: this does not return any measured entities as the entitySpace object does not contain this information.
        """

        if not ExplicitEntitySpaceGridSampleGenerator.samplerCompatibleWithEntitySpace(
            entitySpace=entitySpace
        ):
            raise ValueError(
                f"Cannot use ExplicitEntitySpaceGridSampleGenerator with {entitySpace}"
            )

        def iterator_closure(
            entitySpace: EntitySpaceRepresentation,
        ) -> typing.Callable[[], typing.Generator[list[Entity], None, None]]:

            def sequential_iterator() -> typing.Generator[list[Entity], None, None]:

                names = [c.identifier for c in entitySpace.constitutiveProperties]
                batch = []
                for point in entitySpace.sequential_point_iterator():
                    entity = entitySpace.entity_for_point(
                        dict(zip(names, point, strict=True))
                    )
                    batch.append(entity)
                    if len(batch) == batchsize:
                        yield batch
                        batch = []

                # If the number of points is not divisible by batch size we will have leftovers
                if len(batch) != 0:
                    yield batch

            def random_iterator() -> typing.Generator[list[Entity], None, None]:

                names = [c.identifier for c in entitySpace.constitutiveProperties]
                batch = []
                for point in entitySpace.random_point_iterator():
                    entity = entitySpace.entity_for_point(
                        dict(zip(names, point, strict=True))
                    )
                    batch.append(entity)
                    if len(batch) == batchsize:
                        yield batch
                        batch = []

                if len(batch) != 0:
                    yield batch

            return (
                sequential_iterator
                if self.mode == WalkModeEnum.SEQUENTIAL
                else random_iterator
            )

        retval = iterator_closure(entitySpace)
        return retval()

    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize: int = 1
    ) -> typing.AsyncGenerator[list[Entity], None]:
        """Returns an remoteEntityIterator that returns entities in order"""

        async def iterator_closure(
            discoverySpaceActor: DiscoverySpaceManager,
        ) -> typing.Callable[[], typing.AsyncGenerator[list[Entity], None]]:

            # noinspection PyUnresolvedReferences
            entitySpace = await discoverySpaceActor.entitySpace.remote()

            if not ExplicitEntitySpaceGridSampleGenerator.samplerCompatibleWithEntitySpace(
                entitySpace=entitySpace
            ):
                raise ValueError(
                    f"Cannot use ExplicitEntitySpaceGridSampleGenerator with {entitySpace}"
                )

            async def sequential_iterator() -> (
                typing.AsyncGenerator[list[Entity], None]
            ):

                names = [c.identifier for c in entitySpace.constitutiveProperties]
                batch = []
                for point in entitySpace.sequential_point_iterator():
                    entity = await discoverySpaceActor.entity_for_point.remote(
                        point=dict(zip(names, point, strict=True))
                    )
                    batch.append(entity)
                    if len(batch) == batchsize:
                        yield batch
                        batch = []

                if len(batch) != 0:
                    yield batch

            async def random_iterator() -> typing.AsyncGenerator[list[Entity], None]:

                names = [c.identifier for c in entitySpace.constitutiveProperties]
                batch = []
                for point in entitySpace.random_point_iterator():
                    entity = await discoverySpaceActor.entity_for_point.remote(
                        point=dict(zip(names, point, strict=True))
                    )
                    batch.append(entity)
                    if len(batch) == batchsize:
                        yield batch
                        batch = []

                if len(batch) != 0:
                    yield batch

            return (
                sequential_iterator
                if self.mode == WalkModeEnum.SEQUENTIAL
                else random_iterator
            )

        retval = await iterator_closure(remoteDiscoverySpace)
        return retval()


def sample_random_entity_from_space(
    es: EntitySpaceRepresentation,
) -> Entity | None:

    # Sample an entity from the entity space
    s = ExplicitEntitySpaceGridSampleGenerator(mode=WalkModeEnum.RANDOM)
    entity = None
    for e in s.entitySpaceIterator(es):
        entity = e[0]
        break

    return entity
