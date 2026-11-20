# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""ADO sampler for sequential forest-based selection in finite discovery spaces."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

from sklearn.ensemble import RandomForestRegressor

from active_learning.data import (
    candidate_entities,
    measured_labels,
    validate_target_output,
)
from ado.core.discoveryspace.samplers import (
    BaseSampler,
    ExplicitEntitySpaceGridSampleGenerator,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator

    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer

    from active_learning.forest_sampling import ForestSamplingState
    from active_learning.parameters import ActiveLearningParameters
    from ado.core.discoveryspace.space import DiscoverySpace, Entity
    from ado.modules.operators.discovery_space_manager import DiscoverySpaceManager


def build_forest(
    params: ActiveLearningParameters, random_state: int | None
) -> RandomForestRegressor:
    """Return an unfitted Random Forest with the configured hyperparameters."""
    return RandomForestRegressor(
        n_estimators=params.nEstimators,
        min_samples_leaf=params.minSamplesLeaf,
        max_features=params.maxFeatures,
        criterion=params.criterion,
        random_state=random_state,
        n_jobs=params.nJobs,
    )


class ForestSampleSelector(BaseSampler):
    """Feed each completed target measurement back to the forest sampling state."""

    state_factory: ClassVar[Callable[..., ForestSamplingState]]
    parameter_class: ClassVar[type[ActiveLearningParameters]]

    def __init__(self, params: ActiveLearningParameters) -> None:
        """Set the forest and measurement parameters."""
        self.params = params

    @staticmethod
    @abstractmethod
    def target_label(entity: Entity, target_name: str) -> float | None:
        """Read the training label for one observed property."""

    @staticmethod
    @abstractmethod
    def encode_feature_matrix(
        discovery_space: DiscoverySpace, entities: list[Entity]
    ) -> tuple[pd.DataFrame, np.ndarray, ColumnTransformer]:
        """Return the training frame, encoded features and fitted encoder."""

    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls,
        remoteDiscoverySpace: DiscoverySpaceManager,
    ) -> bool:
        """Verify that the discovery space is finite and compatible."""
        return ExplicitEntitySpaceGridSampleGenerator.samplerCompatibleWithDiscoverySpaceRemote(
            remoteDiscoverySpace
        )

    @classmethod
    def parameters_model(cls) -> type[ActiveLearningParameters]:
        """Return the parameter model validated by Random Walk."""
        return cls.parameter_class

    def entityIterator(
        self,
        discoverySpace: DiscoverySpace,
        batchsize: int = 1,
    ) -> Generator[list[Entity], None, None]:
        """Require a remote iterator to read each new measurement."""
        raise NotImplementedError(
            f"{type(self).__name__} only supports remoteEntityIterator"
        )

    async def remoteEntityIterator(
        self,
        remoteDiscoverySpace: DiscoverySpaceManager,
        batchsize: int = 1,
    ) -> AsyncGenerator[list[Entity], None]:
        """Select one entity at a time from a finite discovery space."""
        if batchsize != 1:
            raise ValueError(
                f"{type(self).__name__} requires batchsize=1, got {batchsize}"
            )

        return self._entity_stream(remoteDiscoverySpace)

    async def _entity_stream(
        self,
        remoteDiscoverySpace: DiscoverySpaceManager,
    ) -> AsyncGenerator[list[Entity], None]:
        """Build the initial pool and yield entities as their measurements arrive."""
        discovery_space: DiscoverySpace = (
            await remoteDiscoverySpace.discoverySpace.remote()
        )
        target_id = validate_target_output(
            self.params.targetOutput, discovery_space
        ).identifier
        existing_labels = measured_labels(discovery_space, target_id, self.target_label)
        entities = candidate_entities(discovery_space)
        if not entities:
            return

        _, features, _ = self.encode_feature_matrix(discovery_space, entities)

        initial_indices = [
            idx
            for idx, entity in enumerate(entities)
            if entity.identifier in existing_labels
        ]
        initial_labels = [
            existing_labels[entities[idx].identifier] for idx in initial_indices
        ]

        state = type(self).state_factory(
            features,
            initial_measured_indices=initial_indices,
            initial_labels=initial_labels,
            params=self.params,
        )

        for batch in state.sample_with_feedback(
            entities, discovery_space, target_id, self.target_label
        ):
            yield batch
