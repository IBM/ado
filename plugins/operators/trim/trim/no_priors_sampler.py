# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import asyncio
import logging
import typing

from pydantic import BaseModel

from orchestrator.core.discoveryspace.samplers import BaseSampler
from orchestrator.core.discoveryspace.space import DiscoverySpace, Entity
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager
from trim.no_priors_pydantic import NoPriorsParameters
from trim.utils.order import order_df_for_sampling_with_no_priors
from trim.utils.space_df_connector import (
    get_list_of_entities_from_df_and_space,
    get_source_and_target,
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

        async def iterator_closure(
            stateHandle: DiscoverySpaceManager,  # type: ignore[name-defined]
        ) -> typing.Callable[[], typing.AsyncGenerator[list[Entity], None]]:

            logger_no_priors.info("Characterization with no-priors starts.\n")
            logger_no_priors.info(f"PARAMETERS ARE:\n{self.params}\n\n")

            discoverySpace = await stateHandle.discoverySpace.remote()
            source_df, target_df = get_source_and_target(
                discoverySpace, self.params.targetOutput
            )
            logger_no_priors.info(f"Target dataframe has length {len(target_df)}")
            target_df = order_df_for_sampling_with_no_priors(
                target_df,
                [
                    cp.identifier
                    for cp in discoverySpace.entitySpace.constitutiveProperties
                ],
                self.params.samples - len(source_df),
                strategy=self.params.sampling_strategy,
            )
            list_of_entities_for_no_prior_characterization = (
                get_list_of_entities_from_df_and_space(
                    df=target_df, space=discoverySpace
                )
            )

            logger_no_priors.info(
                "\n\nCharacterization with no-priors finished. Starting Iterative Modeling.\n"
            )

            async def iterator() -> typing.AsyncGenerator[list[Entity], None]:  # type: ignore[name-defined]
                logger_no_priors.info(
                    "\n\nIteration over sorted entities for no priors characterization starts.\n"
                )
                await asyncio.sleep(0.1)
                for i in range(
                    0, len(list_of_entities_for_no_prior_characterization), batchsize
                ):
                    entities = list_of_entities_for_no_prior_characterization[
                        i : i + batchsize
                    ]
                    if len(entities) == 0:
                        logger_no_priors.info(
                            "\n\nCharacterization with no-priors finished.\n"
                        )
                        break
                    else:
                        yield entities
                logger_no_priors.info("\n\nCharacterization with no-priors finished.\n")

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

            # list_of_entities = list(...)  # type: ignore[name-defined]
            # numberEntities = len(list_of_entities)

            def iterator() -> typing.Generator[list[Entity], None, None]:  # type: ignore[name-defined]
                raise NotImplementedError
                # ...for i in range(0, numberEntities, batchsize):

            return iterator

        retval = iterator_closure(discoverySpace)
        return retval()

    @classmethod
    def parameters_model(cls) -> type[BaseModel] | None:
        return NoPriorsParameters

    def __init__(self, parameters: NoPriorsParameters) -> None:
        self.params = parameters
