# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import asyncio
import logging
import typing

from pydantic import BaseModel

from ado.core.discoveryspace.samplers import BaseSampler
from ado.core.discoveryspace.space import DiscoverySpace, Entity
from ado.modules.operators.discovery_space_manager import DiscoverySpaceManager
from ado.utilities.logging import configure_logging
from trim.missing_target import record_unmeasured_entity
from trim.samplers.no_priors_utils import (
    get_list_of_entities_from_df_and_space,
    get_source_and_target,
    order_df_for_sampling_with_no_priors,
)
from trim.trim_pydantic import NoPriorsParametersInternal


# NOTE: to repeat the operation on the same space I can delete the operation if the output of this operation
# are not used by another operation
class NoPriorsSampleSelector(BaseSampler):
    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls,
        remoteDiscoverySpace: DiscoverySpaceManager,  # type: ignore[name-defined]
    ) -> bool:
        return True

    def generate_sorted_entities(self, discovery_space: DiscoverySpace) -> list[Entity]:
        source_df, target_df = get_source_and_target(
            discovery_space, self.params.targetOutput
        )

        # The 'samples' parameter specifies the number of NEW entities to sample,
        # regardless of how many entities have already been measured in the space
        # they also must measure the targetOutput. However, entities may be unable to
        # measure the targetOutput, this here we generate *all* entities in the order
        # that we would visit them if we had to exhaust the entire unmeasured space.
        self.log.info(
            f"Space has {len(source_df)} measured and {len(target_df)} unmeasured entities. "
            f"Sampling {self.params.samples} new entities as requested."
        )
        target_df = order_df_for_sampling_with_no_priors(
            target_df,
            [
                cp.identifier
                for cp in discovery_space.entitySpace.constitutiveProperties
            ],
            n=len(target_df),
            strategy=self.params.sampling_strategy,
        )

        return get_list_of_entities_from_df_and_space(
            df=target_df, space=discovery_space
        )

    def noprior_iterator(
        self, discovery_space: DiscoverySpace
    ) -> typing.Generator[list[Entity], None, None]:
        self.log.info("Characterization with no-priors starts.\n")
        self.log.info(f"Parameters are:\n{self.params}\n\n")

        sorted_entities = self.generate_sorted_entities(discovery_space)
        self.log.warning(
            f"\n\nIteration over sorted entities for no priors characterization starts for {self.params.samples} "
            f"points and {len(sorted_entities)} entities.\n"
        )
        total_unmeasured = 0
        total_measured = 0

        previous_source_df, _ = get_source_and_target(
            discovery_space, self.params.targetOutput
        )

        for i in range(len(sorted_entities)):
            entity = sorted_entities[i]
            yield [entity]

            # VV: When iterated via remoteEntityIterator the code will only get to this point
            # when the RandomWalk operator asks for the next Entity
            current_source_df, _ = get_source_and_target(
                discovery_space, self.params.targetOutput
            )
            if len(current_source_df) == len(previous_source_df):
                total_unmeasured += 1
                record_unmeasured_entity(
                    entity_identifier=entity.identifier,
                    missing_target_measurements=self.params.missingTargetMeasurements,
                    total_unmeasured=total_unmeasured,
                    target_output=self.params.targetOutput,
                    logger=self.log,
                    additional_info="",
                )
            else:
                total_measured += 1

                if total_measured >= self.params.samples:
                    break

            previous_source_df = current_source_df

        self.log.info(
            f"Characterization with no-priors finished. Yielded {total_measured + total_unmeasured} "
            f"samples out of {len(sorted_entities)} entities. "
            f"Of those, {total_measured} measured the targetOutput. Starting Iterative Modeling."
        )

    async def remoteEntityIterator(
        self,
        remoteDiscoverySpace: DiscoverySpaceManager,
        batchsize: int = 1,
    ) -> typing.AsyncGenerator[list[Entity], None]:
        """
        Generate entities for no-priors characterization sampling.

        Orders the target space using a high-dimensional sampling strategy (e.g., CLHS, Sobol)
        without relying on prior model knowledge or feature importance.

        Args:
            remoteDiscoverySpace: Manager for the discovery space state
            batchsize: Number of entities to yield per iteration

        Yields:
            List of Entity objects to be measured, in the determined order
        """
        if batchsize != 1:
            raise ValueError(
                f"NoPriorsSampleSelector requires batchsize=1, got {batchsize}"
            )

        discovery_space = await remoteDiscoverySpace.discoverySpace.remote()

        async def async_wrapper() -> typing.AsyncGenerator[list[Entity], None]:
            await asyncio.sleep(0.001)
            for entity in self.noprior_iterator(discovery_space):
                yield entity

                # VV: This is crucial, we want to release the CPU for RandomWalk to
                # attempt measuring the entity we just yielded. noprior_iterator() needs to
                # know whether the Entity measured the targetOutput or not.
                await asyncio.sleep(0.001)

        return async_wrapper()

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize: int = 1
    ) -> typing.Generator[list[Entity], None, None]:
        """
        Generate entities for no-priors characterization sampling (synchronous version).

        Orders the target space using a high-dimensional sampling strategy (e.g., CLHS, Sobol)
        without relying on prior model knowledge or feature importance.

        Args:
            discoverySpace: The discovery space to sample from
            batchsize: Number of entities to yield per iteration

        Yields:
            List of Entity objects to be measured, in the determined order
        """
        if batchsize != 1:
            raise ValueError(
                f"NoPriorsSampleSelector requires batchsize=1, got {batchsize}"
            )

        return self.noprior_iterator(discoverySpace)

    @classmethod
    def parameters_model(cls) -> type[BaseModel] | None:
        return NoPriorsParametersInternal

    def __init__(self, parameters: NoPriorsParametersInternal) -> None:
        self.params = parameters

        configure_logging()

        self.log = logging.getLogger(__name__)
