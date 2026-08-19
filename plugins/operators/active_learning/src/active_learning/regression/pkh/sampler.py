# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Predictive kernel herding (PKH): a random-forest leaf-histogram sampler."""

from __future__ import annotations

import typing

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from active_learning.regression._shared import (
    ForestCriterion,
    _encode_leaf_signature,
    _PredictiveSampleSelector,
    _SequentialSelectionState,
)
from active_learning.regression.pkh.parameters import PKHParameters

if typing.TYPE_CHECKING:
    import pydantic


class _PKHSelectionState(_SequentialSelectionState):
    """Exact random-forest leaf-histogram state for PKH."""

    def __init__(
        self,
        features: np.ndarray,
        initial_indices: np.ndarray,
        initial_labels: np.ndarray,
        *,
        n_estimators: int = 100,
        min_samples_leaf: int = 2,
        epoch_length: int = 10,
        seed: int = 42,
        criterion: ForestCriterion = "squared_error",
        n_jobs: int = 1,
    ) -> None:
        """Create PKH state for a complete finite feature pool."""

        if epoch_length <= 0:
            raise ValueError("epoch_length must be positive")
        super().__init__(
            features,
            initial_indices,
            initial_labels,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=seed,
            criterion=criterion,
            n_jobs=n_jobs,
        )
        self.epoch_length = int(epoch_length)
        self._forest_fitted = False
        self._steps_since_refit = 0

    def _fit_epoch(self) -> None:
        """Fit the forest and rebuild the current leaf histograms."""

        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        selected = np.asarray(self._selected, dtype=np.int64)
        forest.fit(self.features_[selected], self._selected_labels())
        leaf_signature, leaf_counts = _encode_leaf_signature(
            forest.apply(self.features_)
        )

        max_leaves = max(leaf_counts)
        pool_leaf_mass = np.zeros(
            (self.n_estimators, max_leaves),
            dtype=float,
        )
        selected_leaf_count = np.zeros_like(pool_leaf_mass)
        for tree_index, tree_leaf_count in enumerate(leaf_counts):
            pool_counts = np.bincount(
                leaf_signature[:, tree_index],
                minlength=tree_leaf_count,
            )
            pool_leaf_mass[tree_index, :tree_leaf_count] = pool_counts / float(
                self.features_.shape[0]
            )
            selected_leaf_count[tree_index, :tree_leaf_count] = np.bincount(
                leaf_signature[selected, tree_index],
                minlength=tree_leaf_count,
            )

        self.forest_ = forest
        self.leaf_signature_ = leaf_signature
        self.pool_leaf_mass_ = pool_leaf_mass
        self.selected_leaf_count_ = selected_leaf_count
        self._forest_fitted = True
        self._steps_since_refit = 0

    def _choose_candidate(self, candidates: np.ndarray) -> int:
        """Choose the largest random-forest leaf-distribution deficit."""

        if not self._forest_fitted or self._steps_since_refit >= self.epoch_length:
            self._fit_epoch()

        tree_indices = np.arange(self.leaf_signature_.shape[1])
        candidate_leaves = self.leaf_signature_[candidates]
        deficits = self.pool_leaf_mass_[tree_indices, candidate_leaves] - (
            self.selected_leaf_count_[tree_indices, candidate_leaves]
            / float(len(self._selected))
        )
        scores = np.mean(deficits, axis=1)
        maximisers = candidates[scores == np.max(scores)]
        return int(np.min(maximisers))

    def _on_update(self, index: int) -> None:
        """Count the newly labelled point in every forest leaf it occupies."""

        if not self._forest_fitted:
            return
        tree_indices = np.arange(self.leaf_signature_.shape[1])
        np.add.at(
            self.selected_leaf_count_,
            (tree_indices, self.leaf_signature_[index]),
            1.0,
        )
        self._steps_since_refit += 1


class PKHSampleSelector(_PredictiveSampleSelector):
    """ADO sample selector implementing predictive kernel herding.

    References
    ----------
    .. [1] E. Aydar, C. Pinto, S. Venugopal, and D. Chatzopoulos, "Sampling
           Where It Matters: Predicting LLM Serving Performance with
           Predictive Kernel Herding," in Proceedings of the Sixth European
           Workshop on Machine Learning and Systems (EuroMLSys '26), ACM,
           2026, pp. 13-22. doi: 10.1145/3805621.3807633.
    """

    def __init__(self, parameters: PKHParameters) -> None:
        """Create a PKH sampler from validated parameters."""

        super().__init__(parameters)
        self.params = parameters

    @classmethod
    def parameters_model(cls) -> type[pydantic.BaseModel]:
        """Return the PKH custom-sampler parameter model."""

        return PKHParameters

    def _selection_state(
        self,
        features: np.ndarray,
        initial_indices: np.ndarray,
        initial_labels: np.ndarray,
    ) -> _PKHSelectionState:
        """Create PKH state for the encoded discovery-space pool."""

        return _PKHSelectionState(
            features,
            initial_indices,
            initial_labels,
            n_estimators=self.params.nEstimators,
            min_samples_leaf=self.params.minSamplesLeaf,
            epoch_length=self.params.epochLength,
            seed=self.params.seed,
            criterion=self.params.criterion,
            n_jobs=self.params.nJobs,
        )


__all__ = ["PKHSampleSelector", "_PKHSelectionState"]
