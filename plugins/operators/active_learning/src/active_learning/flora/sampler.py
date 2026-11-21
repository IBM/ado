# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Finite-pool FLORA selection using pointwise disagreement between forest trees."""

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from active_learning.data import feature_frame, fit_feature_encoder
from active_learning.flora.parameters import FLORAParameters
from active_learning.forest_sampling import ForestSamplingState
from active_learning.sampler import ForestSampleSelector, build_forest
from ado.core.discoveryspace.space import DiscoverySpace, Entity
from ado.schema.domain import VariableTypeEnum

# Separate deterministic streams keep forest refits independent of tie-breaking draws.
FOREST_SEED_STREAM = 0x464F5245
TIE_BREAK_SEED_STREAM = 0x54494553


def encode_forest_features(
    frame: pd.DataFrame, *, categories: dict[str, pd.CategoricalDtype]
) -> np.ndarray:
    """Encode raw rows with the same category levels and column order used at fit time."""
    for column, dtype in categories.items():
        values = frame[column]
        if (values.notna() & ~values.isin(dtype.categories)).any():
            raise ValueError(f"Unknown category in feature '{column}'.")
    encoded = pd.get_dummies(
        frame.astype(categories), columns=list(categories), dummy_na=True, dtype=float
    )
    features = encoded.to_numpy()
    if np.iscomplexobj(features):
        raise ValueError("FLORA features must be real numbers.")
    features = np.asarray(features, dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("FLORA features must be finite.")
    return features


class FLORASelectionState(ForestSamplingState):
    """Select entities by their estimated one-label reduction in forest disagreement."""

    def __init__(
        self,
        features: np.ndarray,
        *,
        initial_measured_indices: list[int],
        initial_labels: list[float],
        params: FLORAParameters,
    ) -> None:
        """Initialize the random pilot and independent forest and tie-breaking streams."""
        super().__init__(
            features,
            initial_measured_indices,
            initial_labels,
            pilot_size=params.pilotSize,
            seed=params.seed,
        )
        self.params = params
        self._next_refit_label_count = 0
        self._forest_rng = np.random.default_rng([params.seed, FOREST_SEED_STREAM])
        self._tie_rng = np.random.default_rng([params.seed, TIE_BREAK_SEED_STREAM])

    def fit_forest_and_histograms(self) -> None:
        """Fit the forest and average pointwise tree disagreement within each leaf."""
        forest_seed = int(self._forest_rng.integers(np.iinfo(np.int32).max))
        forest = build_forest(self.params, forest_seed)
        self._pool_leaf_counts = self.fit_forest(forest)
        prediction_variance = np.var(
            [tree.predict(self.features) for tree in forest.estimators_], axis=0
        )
        self._mean_leaf_disagreement = np.zeros_like(self._pool_leaf_counts)
        for tree_index, leaf_indices in enumerate(self._leaf_assignments.T):
            pool_counts = self._pool_leaf_counts[tree_index]
            total_disagreement = np.bincount(
                leaf_indices, weights=prediction_variance, minlength=len(pool_counts)
            )
            np.divide(
                total_disagreement,
                pool_counts,
                out=self._mean_leaf_disagreement[tree_index],
                where=pool_counts > 0,
            )
        measured_count = len(self.measured_indices)
        self._next_refit_label_count = measured_count + math.ceil(
            measured_count / math.log(measured_count + 1)
        )

    def _select_after_pilot(self, unmeasured: np.ndarray) -> int:
        """Select the largest one-label gain, drawing uniformly among exact ties."""
        if len(self.measured_indices) >= self._next_refit_label_count:
            self.fit_forest_and_histograms()
        measured_leaf_counts = self._sample_leaf_counts
        coverage_gain = np.divide(
            self._pool_leaf_counts,
            measured_leaf_counts * (measured_leaf_counts + 1),
            out=np.zeros_like(measured_leaf_counts),
            where=measured_leaf_counts > 0,
        )
        # Each candidate reaches a fitted leaf containing at least one training label.
        leaf_gain = (
            self._mean_leaf_disagreement / self.n_candidates * (coverage_gain + 1)
        )
        candidate_leaves = self._leaf_assignments[unmeasured]
        scores = leaf_gain[np.arange(self.params.nEstimators), candidate_leaves].mean(
            axis=1
        )
        best_candidates = unmeasured[scores == scores.max()]
        return int(self._tie_rng.choice(best_candidates))


class FLORASampleSelector(ForestSampleSelector):
    """Feed measurements from ADO to FLORA's selection state."""

    state_factory = FLORASelectionState
    parameter_class = FLORAParameters

    @staticmethod
    def target_label(entity: Entity, target_name: str) -> float | None:
        """Average all finite scalar measurements of the observed property."""
        values = entity.valuesForObservedPropertyIdentifier(target_name)
        if not values:
            return None
        labels = np.asarray([value.value for value in values], dtype=float)
        if labels.ndim != 1 or not np.isfinite(labels).all():
            raise ValueError(
                f"Target '{target_name}' requires finite scalar measurements."
            )
        return float(labels.mean())

    @staticmethod
    def encode_feature_matrix(
        discovery_space: DiscoverySpace, entities: list[Entity]
    ) -> tuple[pd.DataFrame, np.ndarray, ColumnTransformer]:
        """One-hot encode declared categories, including numeric levels and missing values."""
        declared_categories = (
            prop.identifier
            for prop in discovery_space.entitySpace.constitutiveProperties
            if prop.propertyDomain.variableType
            in (
                VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
                VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE,
            )
        )
        frame, categories = feature_frame(
            discovery_space, entities, declared_categories
        )
        transformer = FunctionTransformer(
            encode_forest_features,
            kw_args={"categories": frame[categories].dtypes.to_dict()},
        )
        encoder = ColumnTransformer([("features", transformer, frame.columns.tolist())])
        return fit_feature_encoder(frame, encoder)
