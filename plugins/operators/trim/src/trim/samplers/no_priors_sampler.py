# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import asyncio
import logging
import typing

from pydantic import BaseModel

from orchestrator.core.discoveryspace.samplers import BaseSampler
from orchestrator.core.discoveryspace.space import DiscoverySpace, Entity
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager
from trim.samplers.missing_target_utils import (
    entity_measured_target,
    record_missing_and_check_budget,
)
from trim.samplers.no_priors_parameters import MissingTargetMode, NoPriorsParameters
from trim.samplers.no_priors_utils import (
    get_list_of_entities_from_df_and_space,
    get_source_and_target,
    order_df_for_sampling_with_no_priors,
)

logger_no_priors = logging.getLogger(__name__)


# NOTE: to repeat the operation on the same space I can delete the operation if the output of this operation
# are not used by another operation
class NoPriorsSampleSelector(BaseSampler):
    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls,
        remoteDiscoverySpace: DiscoverySpaceManager,  # type: ignore[name-defined]
    ) -> bool:
        return True

    def _core_iterator_logic(
        self,
        discoverySpace: DiscoverySpace,
        list_of_entities: list[Entity],
    ) -> typing.Generator[list[Entity], None, None]:
        """Core iterator logic shared between sync and async implementations.

        This generator is driven by ``async_wrapper`` in ``remoteEntityIterator``,
        which interleaves with the Ray actor event loop. The sequence per entity is:

        Checking last_entity at the top of the next iteration (rather than immediately
        after yield) is correct because only yielding the event loop AND refreshing
        the sampleStore is it safe to check if the entity measured the targetOutput.

        In the sync path (``entityIterator`` / tests) there is no Ray boundary:
        measurements are written into the same in-memory object that the generator
        holds, so no refresh is needed.

        Args:
            list_of_entities: Ordered list of candidate entities to sample from.
            discoverySpace: The active discovery space used for measurement checks.
                In the async path this object is kept fresh by ``refresh()`` calls
                in ``async_wrapper`` between every yield.

        Yields:
            Single-element lists containing the next entity to measure.
        """
        self._missing_count = 0

        quota_count = 0
        quota = self.params.samples
        last_entity: Entity | None = None

        for entity in list_of_entities:
            if quota_count >= quota:
                break

            # Check whether the entity we yielded in the previous iteration produced
            # a target measurement. By this point async_wrapper has already called
            # discoverySpace.sample_store.refresh(), so the snapshot is current.
            if last_entity is not None:
                hit, _ = entity_measured_target(
                    last_entity, discoverySpace, self.params.targetOutput
                )
                if hit:
                    quota_count += 1
                else:
                    mode = self.params.missingTargetVariables.mode
                    self._missing_count = record_missing_and_check_budget(
                        params=self.params,
                        entity_id=last_entity.identifier,  # type: ignore[arg-type]
                        missing_count=self._missing_count,
                        discoverySpace=discoverySpace,
                        additional_info=(
                            f"Detected during no-priors characterization "
                            f"(quota {quota_count}/{quota})."
                        ),
                    )
                    if mode == MissingTargetMode.InjectDefaultValue:
                        # Entity will get a synthetic row in the TRIM phase;
                        # it still counts towards the quota.
                        quota_count += 1
                    # MissingTargetMode.Skip: quota_count not incremented.

                if quota_count >= quota:
                    break

            last_entity = entity
            yield [entity]

        # Handle the very last yielded entity; it was never checked in the loop.
        if last_entity is not None and quota_count < quota:
            hit, _ = entity_measured_target(
                last_entity, discoverySpace, self.params.targetOutput
            )
            if not hit:
                mode = self.params.missingTargetVariables.mode
                self._missing_count = record_missing_and_check_budget(
                    params=self.params,
                    entity_id=last_entity.identifier,  # type: ignore[arg-type]
                    missing_count=self._missing_count,
                    discoverySpace=discoverySpace,
                    additional_info=(
                        f"Detected during no-priors characterization "
                        f"(quota {quota_count}/{quota})."
                    ),
                )

        if quota_count < quota:
            logger_no_priors.warning(
                f"No-priors pool exhausted after {quota_count}/{quota} entities "
                "with target measurements. The operator will handle the shortfall."
            )

        logger_no_priors.info("\n\nCharacterization with no-priors finished.\n")

    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize: int = 1
    ) -> typing.AsyncGenerator[list[Entity], None]:
        """Generate entities for no-priors characterization sampling (async).

        Orders the full target space using a high-dimensional sampling strategy
        (e.g., CLHS, Sobol) without relying on prior model knowledge.  Applies
        the ``missingTargetVariables`` policy when an entity does not produce a
        target measurement:

        - ``RaiseError``: raises immediately.
        - ``InjectDefaultValue``: counts towards the quota.
        - ``Skip``: does **not** count towards the quota.

        Args:
            remoteDiscoverySpace: Manager for the discovery space state.
            batchsize: Number of entities to yield per iteration (must be 1).

        Yields:
            Single-element lists containing the next entity to measure.
        """
        if batchsize != 1:
            raise ValueError(
                f"NoPriorsSampleSelector.remoteEntityIterator expects batchsize=1, got {batchsize}"
            )

        logger_no_priors.info("Characterization with no-priors starts.\n")
        logger_no_priors.info(f"Parameters are:\n{self.params}\n\n")

        discoverySpace = await remoteDiscoverySpace.discoverySpace.remote()
        source_df, target_df = get_source_and_target(
            discoverySpace, self.params.targetOutput
        )
        logger_no_priors.info(f"Target dataframe has length {len(target_df)}")
        logger_no_priors.info(
            f"Space has {len(source_df)} measured entities. "
            f"Sampling {self.params.samples} new entities as requested."
        )

        full_pool_df = order_df_for_sampling_with_no_priors(
            target_df,
            [cp.identifier for cp in discoverySpace.entitySpace.constitutiveProperties],
            len(target_df),
            strategy=self.params.sampling_strategy,
        )
        list_of_entities = get_list_of_entities_from_df_and_space(
            df=full_pool_df, space=discoverySpace
        )
        logger_no_priors.info(
            f"No-priors pool: {len(list_of_entities)} candidates (quota={self.params.samples}).\n"
        )

        async def async_wrapper() -> typing.AsyncGenerator[list[Entity], None]:
            await asyncio.sleep(0.001)
            for entity_batch in self._core_iterator_logic(
                discoverySpace, list_of_entities
            ):
                yield entity_batch
                await asyncio.sleep(0.001)  # Allow other async tasks to run
                # Pull new measurement results written by the actuator actor into
                # the local discoverySpace copy without re-fetching the full object.
                discoverySpace.sample_store.refresh()

        return async_wrapper()

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize: int = 1
    ) -> typing.Generator[list[Entity], None, None]:
        """Generate entities for no-priors characterization sampling (synchronous).

        Orders the full target space using a high-dimensional sampling strategy
        (e.g., CLHS, Sobol) without relying on prior model knowledge.  Applies
        the ``missingTargetVariables`` policy when an entity does not produce a
        target measurement.

        Args:
            discoverySpace: The discovery space to sample from.
            batchsize: Number of entities to yield per iteration (must be 1).

        Yields:
            Single-element lists containing the next entity to measure.
        """
        if batchsize != 1:
            raise ValueError(
                f"NoPriorsSampleSelector.entityIterator expects batchsize=1, got {batchsize}"
            )

        logger_no_priors.info("Characterization with no-priors starts.\n")
        logger_no_priors.info(f"Parameters are:\n{self.params}\n\n")

        source_df, target_df = get_source_and_target(
            discoverySpace, self.params.targetOutput
        )
        logger_no_priors.info(f"Target dataframe has length {len(target_df)}")
        logger_no_priors.info(
            f"Space has {len(source_df)} measured entities. "
            f"Sampling {self.params.samples} new entities as requested."
        )

        full_pool_df = order_df_for_sampling_with_no_priors(
            target_df,
            [cp.identifier for cp in discoverySpace.entitySpace.constitutiveProperties],
            len(target_df),
            strategy=self.params.sampling_strategy,
        )
        list_of_entities = get_list_of_entities_from_df_and_space(
            df=full_pool_df, space=discoverySpace
        )
        logger_no_priors.info(
            f"No-priors pool: {len(list_of_entities)} candidates (quota={self.params.samples}).\n"
        )

        return self._core_iterator_logic(discoverySpace, list_of_entities)

    @classmethod
    def parameters_model(cls) -> type[BaseModel] | None:
        return NoPriorsParameters

    def __init__(self, parameters: NoPriorsParameters) -> None:
        # Sampler configuration parameters.
        self.params = parameters
        # Running count of entities that did not produce a target measurement.
        self._missing_count: int = 0
