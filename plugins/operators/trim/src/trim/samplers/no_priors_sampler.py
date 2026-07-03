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
        cls, remoteDiscoverySpace: DiscoverySpaceManager  # type: ignore[name-defined]
    ) -> bool:
        return True

    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize: int = 1
    ) -> typing.AsyncGenerator[list[Entity], None]:
        """Generate entities for no-priors characterization sampling.

        Orders the full target space using a high-dimensional sampling strategy
        (e.g., CLHS, Sobol) without relying on prior model knowledge.  Applies
        the ``missingTargetVariables`` policy when an entity does not produce a
        target measurement:

        - ``RaiseError``: raises immediately.
        - ``InjectDefaultValue``: counts towards the quota (the TRIM phase will
          inject the synthetic row).
        - ``Skip``: does **not** count towards the quota.  The pool is large
          enough to keep drawing until the quota is met or the pool is exhausted.

        Args:
            remoteDiscoverySpace: Manager for the discovery space state.
            batchsize: Number of entities to yield per iteration (must be 1).

        Yields:
            List of Entity objects to be measured, in the determined order.
        """
        if batchsize != 1:
            raise ValueError(
                f"NoPriorsSampleSelector.remoteEntityIterator expects batchsize=1, got {batchsize}"
            )

        async def iterator_closure(
            stateHandle: DiscoverySpaceManager,  # type: ignore[name-defined]
        ) -> typing.Callable[[], typing.AsyncGenerator[list[Entity], None]]:

            logger_no_priors.info("Characterization with no-priors starts.\n")
            logger_no_priors.info(f"Parameters are:\n{self.params}\n\n")

            discoverySpace = await stateHandle.discoverySpace.remote()
            source_df, target_df = get_source_and_target(
                discoverySpace, self.params.targetOutput
            )
            logger_no_priors.info(f"Target dataframe has length {len(target_df)}")

            logger_no_priors.info(
                f"Space has {len(source_df)} measured entities. "
                f"Sampling {self.params.samples} new entities as requested."
            )
            # Order the full pool (all unsampled candidates) so Skip mode can
            # draw more than `samples` without running out early.
            full_pool_df = order_df_for_sampling_with_no_priors(
                target_df,
                [
                    cp.identifier
                    for cp in discoverySpace.entitySpace.constitutiveProperties
                ],
                len(target_df),
                strategy=self.params.sampling_strategy,
            )
            full_pool = get_list_of_entities_from_df_and_space(
                df=full_pool_df, space=discoverySpace
            )

            pool = full_pool

            logger_no_priors.info(
                f"No-priors pool: {len(pool)} candidates "
                f"(quota={self.params.samples}).\n"
            )

            async def iterator() -> typing.AsyncGenerator[list[Entity], None]:  # type: ignore[name-defined]
                logger_no_priors.info(
                    "\n\nIteration over sorted entities for no priors characterization starts.\n"
                )
                await asyncio.sleep(0.1)

                quota_count = 0
                quota = self.params.samples

                for entity in pool:
                    if quota_count >= quota:
                        break

                    yield [entity]
                    await asyncio.sleep(0.001)

                    ds_after = await stateHandle.discoverySpace.remote()
                    hit, _ = entity_measured_target(
                        entity, ds_after, self.params.targetOutput
                    )

                    if hit:
                        quota_count += 1
                    else:
                        mode = self.params.missingTargetVariables.mode
                        self._missing_count = record_missing_and_check_budget(
                            params=self.params,
                            entity_id=entity.identifier,  # type: ignore[arg-type]
                            missing_count=self._missing_count,
                            discoverySpace=ds_after,
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

                if quota_count < quota:
                    logger_no_priors.warning(
                        f"No-priors pool exhausted after {quota_count}/{quota} entities "
                        "with target measurements. The operator will handle the shortfall."
                    )

                logger_no_priors.info("\n\nCharacterization with no-priors finished.\n")

            return iterator

        retval = await iterator_closure(remoteDiscoverySpace)
        return retval()

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
            List of Entity objects to be measured, in the determined order.
        """
        if batchsize != 1:
            raise ValueError(
                f"NoPriorsSampleSelector.entityIterator expects batchsize=1, got {batchsize}"
            )

        def iterator_closure(
            space: DiscoverySpace,
        ) -> typing.Callable[[], typing.Generator[list[Entity], None, None]]:

            logger_no_priors.info("Characterization with no-priors starts.\n")
            logger_no_priors.info(f"Parameters are:\n{self.params}\n\n")

            source_df, target_df = get_source_and_target(
                space, self.params.targetOutput
            )
            logger_no_priors.info(f"Target dataframe has length {len(target_df)}")

            logger_no_priors.info(
                f"Space has {len(source_df)} measured entities. "
                f"Sampling {self.params.samples} new entities as requested."
            )
            # Order the full pool so Skip mode can draw more than `samples`.
            full_pool_df = order_df_for_sampling_with_no_priors(
                target_df,
                [cp.identifier for cp in space.entitySpace.constitutiveProperties],
                len(target_df),
                strategy=self.params.sampling_strategy,
            )
            full_pool = get_list_of_entities_from_df_and_space(
                df=full_pool_df, space=space
            )

            pool = full_pool

            logger_no_priors.info(
                f"No-priors pool: {len(pool)} candidates "
                f"(quota={self.params.samples}).\n"
            )

            def iterator() -> typing.Generator[list[Entity], None, None]:
                logger_no_priors.info(
                    "\n\nIteration over sorted entities for no priors characterization starts.\n"
                )
                quota_count = 0
                quota = self.params.samples

                for entity in pool:
                    if quota_count >= quota:
                        break

                    yield [entity]

                    hit, _ = entity_measured_target(
                        entity, space, self.params.targetOutput
                    )

                    if hit:
                        quota_count += 1
                    else:
                        mode = self.params.missingTargetVariables.mode
                        self._missing_count = record_missing_and_check_budget(
                            params=self.params,
                            entity_id=entity.identifier,  # type: ignore[arg-type]
                            missing_count=self._missing_count,
                            discoverySpace=space,
                            additional_info=(
                                f"Detected during no-priors characterization "
                                f"(quota {quota_count}/{quota})."
                            ),
                        )
                        if mode == MissingTargetMode.InjectDefaultValue:
                            # Counts towards quota; TRIM phase injects the row.
                            quota_count += 1
                        # MissingTargetMode.Skip: quota_count not incremented.

                if quota_count < quota:
                    logger_no_priors.warning(
                        f"No-priors pool exhausted after {quota_count}/{quota} entities "
                        "with target measurements. The operator will handle the shortfall."
                    )

                logger_no_priors.info("\n\nCharacterization with no-priors finished.\n")

            return iterator

        retval = iterator_closure(discoverySpace)
        return retval()

    @classmethod
    def parameters_model(cls) -> type[BaseModel] | None:
        return NoPriorsParameters

    def __init__(self, parameters: NoPriorsParameters) -> None:
        self.params = parameters
        self._missing_count: int = 0
