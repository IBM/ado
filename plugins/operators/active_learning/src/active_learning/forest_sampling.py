# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Random pilot sampling and forest leaf counts for a finite candidate pool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from scipy.sparse import csr_matrix
    from sklearn.ensemble import RandomForestRegressor

    from ado.core.discoveryspace.space import DiscoverySpace, Entity
    from ado.schema.observed_property import ObservedProperty


class ForestSamplingState(ABC):
    """Track sampled candidates and the forest leaves they occupy."""

    def __init__(
        self,
        features: np.ndarray | csr_matrix,
        initial_measured_indices: list[int],
        initial_labels: list[float],
        *,
        pilot_size: int,
        seed: int | None,
    ) -> None:
        """Draw the rest of the random pilot after accounting for prior measurements."""
        self.features = features
        self.n_candidates = features.shape[0]
        self.measured_indices = list(initial_measured_indices)
        self.measured_labels = list(initial_labels)
        self.measured_set = set(initial_measured_indices)
        self._rng = np.random.default_rng(seed)
        unmeasured = self._unmeasured_indices()
        remaining = min(len(unmeasured), max(0, pilot_size - len(self.measured_set)))
        self._pilot_indices = iter(
            self._rng.choice(unmeasured, size=remaining, replace=False).tolist()
        )
        self.forest: RandomForestRegressor | None = None
        self._leaf_assignments: np.ndarray | None = None
        self._sample_leaf_counts: np.ndarray | None = None

    def _unmeasured_indices(self) -> np.ndarray:
        """Return unmeasured indices in candidate order."""
        return np.fromiter(
            (i for i in range(self.n_candidates) if i not in self.measured_set),
            dtype=np.intp,
        )

    def select_next(self) -> int | None:
        """Draw from the pilot, then apply the selection rule until the pool is exhausted."""
        if (index := next(self._pilot_indices, None)) is not None:
            return index
        unmeasured = self._unmeasured_indices()
        return self._select_after_pilot(unmeasured) if unmeasured.size else None

    @abstractmethod
    def _select_after_pilot(self, unmeasured: np.ndarray) -> int:
        """Choose an unmeasured candidate using the operator's refit and scoring rules."""

    def record_measurement(self, index: int, label: float) -> None:
        """Record a new target value and update its leaf count in each fitted tree."""
        if index in self.measured_set:
            return
        self.measured_indices.append(index)
        self.measured_labels.append(label)
        self.measured_set.add(index)
        if self._sample_leaf_counts is not None:
            leaves = self._leaf_assignments[index]
            self._sample_leaf_counts[np.arange(len(leaves)), leaves] += 1

    def fit_forest(self, forest: RandomForestRegressor) -> np.ndarray:
        """Fit the forest, update measured leaf counts, and return full-pool counts."""
        labels = np.asarray(self.measured_labels, dtype=float)
        forest.fit(self.features[self.measured_indices], labels)
        self.forest = forest
        self._leaf_assignments = forest.apply(self.features).astype(np.int32)
        histogram_size = int(self._leaf_assignments.max()) + 1
        pool_counts = np.zeros((forest.n_estimators, histogram_size), dtype=float)
        self._sample_leaf_counts = np.zeros_like(pool_counts)
        for tree_index, leaves in enumerate(self._leaf_assignments.T):
            pool_counts[tree_index] = np.bincount(leaves, minlength=histogram_size)
            self._sample_leaf_counts[tree_index] = np.bincount(
                leaves[self.measured_indices], minlength=histogram_size
            )
        return pool_counts

    def sample_with_feedback(
        self,
        entities: list[Entity],
        discovery_space: DiscoverySpace,
        target: ObservedProperty,
        read_label: Callable[[Entity, ObservedProperty], float | None],
    ) -> Iterator[list[Entity]]:
        """Yield one entity and record its target measurement before selecting the next."""
        while (index := self.select_next()) is not None:
            entity = entities[index]
            yield [entity]

            stored_entities = discovery_space.sample_store.get_entities(
                identifiers=entity.identifier, require_measurements=True, refresh=True
            )
            label = read_label(stored_entities[0], target) if stored_entities else None
            if label is None:
                raise ValueError(
                    f"Missing finite target measurement for '{target}' "
                    f"on entity {entity.identifier}"
                )
            self.record_measurement(index, label)
