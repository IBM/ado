# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Predictive Kernel Herding (PKH) sample selector for finite Discovery Spaces."""

import logging
from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor

from active_learning.forest_sampling import ForestSamplingState
from active_learning.pkh.data import (
    candidate_entities,
    encode_feature_matrix,
    measured_labels,
    target_label,
    validate_target_output,
)
from active_learning.pkh.parameters import PKHParameters
from ado.core.discoveryspace.samplers import (
    BaseSampler,
    ExplicitEntitySpaceGridSampleGenerator,
)
from ado.core.discoveryspace.space import DiscoverySpace, Entity

if TYPE_CHECKING:
    from ado.modules.operators.discovery_space_manager import DiscoverySpaceManager

logger = logging.getLogger(__name__)


def build_forest(
    params: PKHParameters, random_state: int | None
) -> RandomForestRegressor:
    """Return an unfitted Random Forest configured from the PKH parameters."""
    return RandomForestRegressor(
        n_estimators=params.nEstimators,
        min_samples_leaf=params.minSamplesLeaf,
        max_features=params.maxFeatures,
        criterion=params.criterion,
        random_state=random_state,
        n_jobs=params.nJobs,
    )


class PKHState(ForestSamplingState):
    """Core mathematical state and greedy MMD leaf-histogram selection for PKH."""

    def __init__(
        self,
        features: np.ndarray,
        *,
        initial_measured_indices: list[int],
        initial_labels: list[float],
        params: PKHParameters,
    ) -> None:
        """Initialize PKH selection state for a finite candidate feature pool.

        Args:
            features: 2D array of candidate entity features shape (N, D).
            initial_measured_indices: Indices of entities with prior measurements.
            initial_labels: Measurement target values for initial_measured_indices.
            params: PKH sampler hyperparameters.
        """
        super().__init__(
            features,
            initial_measured_indices,
            initial_labels,
            pilot_size=params.pilotSize,
            seed=params.seed,
        )
        self.params = params
        self._measured_at_last_refit = 0
        self._pool_leaf_mass: np.ndarray | None = None

    def fit_forest_and_histograms(self) -> None:
        """Fit the Random Forest on measured points and construct pool leaf histograms."""
        if len(self.measured_indices) < 2:
            return

        seed = (
            (self.params.seed + len(self.measured_indices)) % 2**32
            if self.params.seed is not None
            else None
        )
        self._pool_leaf_mass = self.fit_forest(build_forest(self.params, seed))
        self._pool_leaf_mass /= self.n_candidates
        self._measured_at_last_refit = len(self.measured_indices)
        logger.info(
            "PKH forest refit: measured=%d mmd2=%.6f",
            len(self.measured_indices),
            self.mmd_squared(),
        )

    def _leaf_discrepancy(self) -> np.ndarray | None:
        """Subtract the measured sample's leaf probabilities from the pool's."""
        if self._pool_leaf_mass is None or self._sample_leaf_counts is None:
            return None
        sample_mass = self._sample_leaf_counts / len(self.measured_indices)
        return self._pool_leaf_mass - sample_mass

    def mmd_squared(self) -> float | None:
        """Return MMD^2 between the measured entities and the whole pool.

        With Phi scaled by 1/sqrt(T), ``||mu_P - mean(Phi(x_s))||^2`` equals
        ``(1/T) * sum_t ||pool_mass_t - sample_mass_t||^2`` in the leaf space.
        Return None before the first forest fit. Each forest defines its own
        kernel. Compare values within the same epoch.
        """
        if (discrepancy := self._leaf_discrepancy()) is None:
            return None
        return float(np.sum(discrepancy**2) / self.params.nEstimators)

    def _select_after_pilot(self, unmeasured: np.ndarray) -> int:
        """Select the candidate with the largest mean leaf discrepancy."""
        new_measurements = len(self.measured_indices) - self._measured_at_last_refit
        if self.forest is None or new_measurements >= self.params.epochLength:
            self.fit_forest_and_histograms()

        if (discrepancy := self._leaf_discrepancy()) is None:
            return int(unmeasured[0])

        scores = np.zeros(len(unmeasured), dtype=float)

        # Score candidates: average under-representation across all trees in the ensemble
        for tree_idx in range(self.params.nEstimators):
            candidate_leaves = self._leaf_assignments[unmeasured, tree_idx]
            scores += discrepancy[tree_idx, candidate_leaves]

        scores /= self.params.nEstimators

        best_positions = np.flatnonzero(
            np.isclose(scores, scores.max(), rtol=0, atol=1e-12)
        )
        return int(unmeasured[self._rng.choice(best_positions)])


class PKHSampleSelector(BaseSampler):
    """ADO sample selector implementing Predictive Kernel Herding."""

    def __init__(self, params: PKHParameters) -> None:
        """Initialize the PKH sample selector."""
        self.params = params

    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls,
        remoteDiscoverySpace: "DiscoverySpaceManager",
    ) -> bool:
        """Verify that the discovery space is finite and compatible."""
        return ExplicitEntitySpaceGridSampleGenerator.samplerCompatibleWithDiscoverySpaceRemote(
            remoteDiscoverySpace
        )

    @classmethod
    def parameters_model(cls) -> type[BaseModel] | None:
        """Return the parameter model validated by RandomWalk for this sampler.

        The operator forwards its full parameter set, including ``targetOutput``
        which the sampler reads to collect measured labels.
        """
        return PKHParameters

    def entityIterator(
        self,
        discoverySpace: DiscoverySpace,
        batchsize: int = 1,
    ) -> Generator[list[Entity], None, None]:
        """Use ``remoteEntityIterator`` to select entities with live measurement feedback.

        Raises:
            NotImplementedError: PKH requires the asynchronous iterator.
        """
        raise NotImplementedError(
            "PKHSampleSelector only supports remoteEntityIterator"
        )

    async def remoteEntityIterator(
        self,
        remoteDiscoverySpace: "DiscoverySpaceManager",
        batchsize: int = 1,
    ) -> AsyncGenerator[list[Entity], None]:
        """Return an async iterator of entities driven by PKH leaf-histogram herding.

        Args:
            remoteDiscoverySpace: Handle to the remote Discovery Space manager.
            batchsize: Must be 1 for sequential active learning feedback.

        Returns:
            Async generator yielding single-element lists of Entity objects to measure.
        """
        if batchsize != 1:
            raise ValueError(f"PKHSampleSelector requires batchsize=1, got {batchsize}")

        return self._entity_stream(remoteDiscoverySpace)

    async def _entity_stream(
        self,
        remoteDiscoverySpace: "DiscoverySpaceManager",
    ) -> AsyncGenerator[list[Entity], None]:
        """Yield candidate entities sequentially driven by PKH leaf-histogram herding."""
        discovery_space: DiscoverySpace = (
            await remoteDiscoverySpace.discoverySpace.remote()
        )
        target_id = validate_target_output(
            self.params.targetOutput, discovery_space
        ).identifier
        existing_labels = measured_labels(discovery_space, target_id)
        entities = candidate_entities(discovery_space)
        if not entities:
            return

        _, features, _ = encode_feature_matrix(discovery_space, entities)

        initial_indices = [
            idx
            for idx, entity in enumerate(entities)
            if entity.identifier in existing_labels
        ]
        initial_labels = [
            existing_labels[entities[idx].identifier] for idx in initial_indices
        ]

        state = PKHState(
            features,
            initial_measured_indices=initial_indices,
            initial_labels=initial_labels,
            params=self.params,
        )

        for batch in state.sample_with_feedback(
            entities, discovery_space, target_id, target_label
        ):
            yield batch
