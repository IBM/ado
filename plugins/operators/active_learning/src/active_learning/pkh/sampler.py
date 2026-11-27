# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Predictive Kernel Herding (PKH) sample selector for finite Discovery Spaces."""

import logging

import numpy as np

from active_learning.forest_sampling import ForestSamplingState
from active_learning.pkh.data import encode_feature_matrix, target_label
from active_learning.pkh.parameters import PKHParameters
from active_learning.sampler import ForestSampleSelector, build_forest

logger = logging.getLogger(__name__)


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


class PKHSampleSelector(ForestSampleSelector):
    """ADO sample selector implementing Predictive Kernel Herding."""

    state_factory = PKHState
    parameter_class = PKHParameters
    target_label = staticmethod(target_label)
    encode_feature_matrix = staticmethod(encode_feature_matrix)
