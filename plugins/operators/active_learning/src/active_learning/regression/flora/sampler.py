# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""FLORA: a pointwise random-forest disagreement acquisition sampler."""

from __future__ import annotations

import math
import typing

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from active_learning.regression._shared import (
    ForestCriterion,
    _encode_leaf_signature,
    _PredictiveSampleSelector,
    _SequentialSelectionState,
)
from active_learning.regression.flora.parameters import FLORAParameters

if typing.TYPE_CHECKING:
    import pydantic


def _flora_leaf_gain(
    pool_count: np.ndarray | float,
    selected_count: np.ndarray | float,
    disagreement: np.ndarray | float,
    pool_size: int,
) -> np.ndarray | float:
    """Return FLORA's one-label risk reduction for covered leaves.

    Args:
        pool_count: Complete-pool population of each leaf.
        selected_count: Number of labelled selections in each leaf.
        disagreement: Mean pointwise forest disagreement in each leaf.
        pool_size: Number of entities in the complete target pool.

    Returns:
        A gain array matching the inputs, or a float for scalar inputs.

    Raises:
        ValueError: If the arrays have different shapes, contain negative
            statistics, or ``pool_size`` is not positive.
    """

    pool_count_array = np.asarray(pool_count, dtype=float)
    selected_count_array = np.asarray(selected_count, dtype=float)
    disagreement_array = np.asarray(disagreement, dtype=float)
    if not (
        pool_count_array.shape == selected_count_array.shape == disagreement_array.shape
    ):
        raise ValueError("leaf statistic arrays must have identical shapes")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    if (
        np.any(pool_count_array < 0.0)
        or np.any(selected_count_array < 0.0)
        or np.any(disagreement_array < 0.0)
    ):
        raise ValueError("leaf statistics cannot be negative")

    covered = selected_count_array > 0.0
    transfer = np.divide(
        pool_count_array,
        selected_count_array * (selected_count_array + 1.0),
        out=np.zeros_like(pool_count_array),
        where=covered,
    )
    gain = disagreement_array / float(pool_size) * (transfer + covered.astype(float))
    if gain.ndim == 0:
        return float(gain)
    return gain


class _FLORASelectionState(_SequentialSelectionState):
    """Finite-pool FLORA acquisition state."""

    def __init__(
        self,
        features: np.ndarray,
        initial_indices: np.ndarray,
        initial_labels: np.ndarray,
        *,
        n_estimators: int = 100,
        min_samples_leaf: int = 2,
        seed: int = 42,
        criterion: ForestCriterion = "squared_error",
        n_jobs: int = 1,
    ) -> None:
        """Create FLORA state for a complete finite feature pool."""

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
        self._rng = np.random.default_rng(self.seed)
        self._forest_fitted = False
        self._next_refit_at = np.iinfo(np.int64).max

    @staticmethod
    def _next_refit_size(selected_size: int) -> int:
        """Return the next near-geometric forest-refit size."""

        if selected_size <= 0:
            raise ValueError("selected_size must be positive")
        increment = math.ceil(selected_size / math.log(selected_size + 1.0))
        return selected_size + max(increment, 1)

    def _fit_epoch(self) -> None:
        """Fit the forest and compile pointwise disagreement by leaf."""

        forest_seed = int(self._rng.integers(np.iinfo(np.int32).max))
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            bootstrap=True,
            random_state=forest_seed,
            n_jobs=self.n_jobs,
        )
        selected = np.asarray(self._selected, dtype=np.int64)
        forest.fit(self.features_[selected], self._selected_labels())
        leaf_signature, leaf_counts = _encode_leaf_signature(
            forest.apply(self.features_)
        )

        tree_predictions = np.vstack(
            [tree.predict(self.features_) for tree in forest.estimators_]
        )
        point_disagreement = np.var(tree_predictions, axis=0, ddof=0)
        max_leaves = max(leaf_counts)
        pool_leaf_count = np.zeros(
            (self.n_estimators, max_leaves),
            dtype=float,
        )
        selected_leaf_count = np.zeros_like(pool_leaf_count)
        leaf_disagreement = np.zeros_like(pool_leaf_count)
        for tree_index, tree_leaf_count in enumerate(leaf_counts):
            codes = leaf_signature[:, tree_index]
            counts = np.bincount(codes, minlength=tree_leaf_count).astype(float)
            disagreement_sum = np.bincount(
                codes,
                weights=point_disagreement,
                minlength=tree_leaf_count,
            )
            pool_leaf_count[tree_index, :tree_leaf_count] = counts
            leaf_disagreement[tree_index, :tree_leaf_count] = np.divide(
                disagreement_sum,
                counts,
                out=np.zeros_like(disagreement_sum),
                where=counts > 0.0,
            )
            selected_leaf_count[tree_index, :tree_leaf_count] = np.bincount(
                codes[selected],
                minlength=tree_leaf_count,
            )

        self.forest_ = forest
        self.leaf_signature_ = leaf_signature
        self.point_disagreement_ = point_disagreement
        self.pool_leaf_count_ = pool_leaf_count
        self.selected_leaf_count_ = selected_leaf_count
        self.leaf_disagreement_ = np.maximum(leaf_disagreement, 0.0)
        self._forest_fitted = True
        self._next_refit_at = self._next_refit_size(len(self._selected))

    def _choose_candidate(self, candidates: np.ndarray) -> int:
        """Choose the largest risk-reduction acquisition score."""

        if not self._forest_fitted or len(self._selected) >= self._next_refit_at:
            self._fit_epoch()

        candidate_leaves = self.leaf_signature_[candidates]
        tree_indices = np.arange(self.leaf_signature_.shape[1])
        leaf_gain = typing.cast(
            "np.ndarray",
            _flora_leaf_gain(
                self.pool_leaf_count_,
                self.selected_leaf_count_,
                self.leaf_disagreement_,
                self.features_.shape[0],
            ),
        )
        risk_scores = np.mean(leaf_gain[tree_indices, candidate_leaves], axis=1)
        maximisers = candidates[risk_scores == np.max(risk_scores)]

        if maximisers.size == 1:
            return int(maximisers[0])
        return int(maximisers[self._rng.integers(maximisers.size)])

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


class FLORASampleSelector(_PredictiveSampleSelector):
    """ADO sample selector implementing pointwise RF disagreement acquisition."""

    def __init__(self, parameters: FLORAParameters) -> None:
        """Create a FLORA sampler from validated parameters."""

        super().__init__(parameters)
        self.params = parameters

    @classmethod
    def parameters_model(cls) -> type[pydantic.BaseModel]:
        """Return the FLORA custom-sampler parameter model."""

        return FLORAParameters

    def _selection_state(
        self,
        features: np.ndarray,
        initial_indices: np.ndarray,
        initial_labels: np.ndarray,
    ) -> _FLORASelectionState:
        """Create FLORA state for the encoded discovery-space pool."""

        return _FLORASelectionState(
            features,
            initial_indices,
            initial_labels,
            n_estimators=self.params.nEstimators,
            min_samples_leaf=self.params.minSamplesLeaf,
            seed=self.params.seed,
            criterion=self.params.criterion,
            n_jobs=self.params.nJobs,
        )


__all__ = ["FLORASampleSelector", "_FLORASelectionState", "_flora_leaf_gain"]
