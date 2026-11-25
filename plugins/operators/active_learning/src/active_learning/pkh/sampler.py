# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Predictive Kernel Herding (PKH) sample selector for finite Discovery Spaces."""

import logging
from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING

import numpy as np
import ray
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor

from active_learning.forest_sampling import ForestSamplingState
from active_learning.pkh.data import (
    candidate_coordinates,
    candidate_entities,
    encode_feature_matrix,
    feature_frame,
    measured_labels,
    validate_candidate_space,
    validate_target_output,
    walk_measurement_labels,
)
from active_learning.pkh.parameters import PKHParameters
from ado.core.discoveryspace.samplers import BaseSampler
from ado.core.discoveryspace.space import DiscoverySpace, Entity

if TYPE_CHECKING:
    from ado.modules.operators.discovery_space_manager import DiscoverySpaceManager

logger = logging.getLogger(__name__)


class PKHState(ForestSamplingState):
    """Core mathematical state and greedy MMD leaf-histogram selection for PKH."""

    def __init__(
        self,
        features: np.ndarray | csr_matrix,
        *,
        initial_measured_indices: list[int],
        initial_labels: list[float],
        params: PKHParameters,
    ) -> None:
        """Initialize sampling from the finite pool and any prior measurements."""
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
        """Fit the forest on measured points and build its pool leaf histograms."""
        if len(self.measured_indices) < 2:
            return

        seed = (
            (self.params.seed + len(self.measured_indices)) % 2**32
            if self.params.seed is not None
            else None
        )
        forest = RandomForestRegressor(
            n_estimators=self.params.nEstimators,
            min_samples_leaf=self.params.minSamplesLeaf,
            max_features=self.params.maxFeatures,
            criterion=self.params.criterion,
            random_state=seed,
            n_jobs=self.params.nJobs,
        )
        self._pool_leaf_mass = self.fit_forest(forest)
        self._pool_leaf_mass /= self.n_candidates
        self._measured_at_last_refit = len(self.measured_indices)
        logger.info(
            "PKH forest refit with %d measurements", self._measured_at_last_refit
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
            return int(self._rng.choice(unmeasured))

        scores = np.zeros(len(unmeasured), dtype=float)

        # Accumulate one tree at a time to bound temporary storage on large grids.
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
        entity_space = ray.get(remoteDiscoverySpace.entitySpace.remote())
        try:
            validate_candidate_space(entity_space)
        except ValueError:
            return False
        return True

    @classmethod
    def parameters_model(cls) -> type[PKHParameters]:
        """Return the parameter model RandomWalk validates for this sampler."""
        return PKHParameters

    def entityIterator(
        self,
        discoverySpace: DiscoverySpace,
        batchsize: int = 1,
    ) -> Generator[list[Entity], None, None]:
        """Require the remote iterator for live measurement feedback."""
        raise NotImplementedError(
            "PKHSampleSelector only supports remoteEntityIterator"
        )

    async def remoteEntityIterator(
        self,
        remoteDiscoverySpace: "DiscoverySpaceManager",
        batchsize: int = 1,
    ) -> AsyncGenerator[list[Entity], None]:
        """Request one entity at a time so each measurement informs the next choice."""
        if batchsize != 1:
            raise ValueError(f"PKHSampleSelector requires batchsize=1, got {batchsize}")

        # The RandomWalk actor supplies the same identifier on its measurement requests.
        walk = ray.get_runtime_context().current_actor
        operation_id = await walk.operationIdentifier.remote()
        return self._entity_stream(remoteDiscoverySpace, operation_id)

    async def _entity_stream(
        self,
        remote_discovery_space: "DiscoverySpaceManager",
        operation_id: str,
    ) -> AsyncGenerator[list[Entity], None]:
        """Yield candidates according to their forest leaf discrepancies."""
        discovery_space: DiscoverySpace = (
            await remote_discovery_space.discoverySpace.remote()
        )
        target = validate_target_output(self.params.targetOutput, discovery_space)
        entities = candidate_entities(discovery_space)
        existing_labels = measured_labels(discovery_space, target, entities)
        if len(existing_labels) == len(entities):
            return
        initial_indices = []
        initial_labels = []
        for index, entity in enumerate(entities):
            if (label := existing_labels.get(entity.identifier)) is not None:
                initial_indices.append(index)
                initial_labels.append(label)

        state = PKHState(
            encode_feature_matrix(
                feature_frame(discovery_space, candidate_coordinates(entities))
            ),
            initial_measured_indices=initial_indices,
            initial_labels=initial_labels,
            params=self.params,
        )

        def read_label(identifier: str) -> int | float | None:
            """Read the current walk's result for the selected entity."""
            return walk_measurement_labels(
                discovery_space, target, operation_id, entity_identifier=identifier
            ).get(identifier)

        for batch in state.sample_with_feedback(entities, target, read_label):
            yield batch
