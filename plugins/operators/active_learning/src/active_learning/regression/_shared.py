# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Machinery shared by ADO's regression-based active-learning strategies.

A regression-based strategy like PKH fits a random-forest regressor to a
finite, explicit discovery-space pool and selects one entity at a time based
on that forest. This module holds everything such strategies share:
parameter defaults, finite-pool encoding, the sequential-selection state
machine, and the ADO operator/RandomWalk wiring.
``active_learning.regression.pkh`` adds only the acquisition logic that is
specific to its strategy.
"""

from __future__ import annotations

import abc
import asyncio
import time
import typing
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import pydantic
import ray

from ado.core.discoveryspace.samplers import BaseSampler
from ado.core.operation.config import FunctionOperationInfo
from ado.core.operation.operation import OperationOutput
from ado.schema.domain import VariableTypeEnum
from ado.schema.entityspace import EntitySpaceRepresentation

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from ado.core.discoveryspace.space import DiscoverySpace
    from ado.modules.operators.discovery_space_manager import DiscoverySpaceManager
    from ado.modules.operators.randomwalk import RandomWalkParameters
    from ado.schema.entity import Entity
    from ado.schema.measurementspace import MeasurementSpace
    from ado.schema.observed_property import ObservedProperty

ForestCriterion = Literal[
    "squared_error",
    "absolute_error",
    "friedman_mse",
    "poisson",
]


class _PredictiveParameters(pydantic.BaseModel):
    """Settings shared by the adaptive random-forest samplers."""

    targetOutput: Annotated[
        str,
        pydantic.Field(
            min_length=1,
            description="Scalar target property used to fit the random forest.",
        ),
    ]
    nEstimators: Annotated[
        int,
        pydantic.Field(
            ge=1,
            description="Number of trees in each random forest.",
        ),
    ] = 100
    minSamplesLeaf: Annotated[
        int,
        pydantic.Field(
            ge=1,
            description="Minimum number of labelled samples in a tree leaf.",
        ),
    ] = 2
    seed: Annotated[
        int,
        pydantic.Field(description="Seed controlling reproducible forest fitting."),
    ] = 42
    criterion: Annotated[
        ForestCriterion,
        pydantic.Field(description="Regression-tree split criterion."),
    ] = "squared_error"
    nJobs: Annotated[
        int,
        pydantic.Field(
            description=(
                "Number of parallel random-forest workers. Use -1 for all "
                "available workers."
            ),
        ),
    ] = -1
    labelWaitTimeoutSeconds: Annotated[
        float,
        pydantic.Field(
            gt=0.0,
            description="Maximum wait for the previously requested target value.",
        ),
    ] = 300.0
    labelPollIntervalSeconds: Annotated[
        float,
        pydantic.Field(
            gt=0.0,
            description="Interval between checks for a requested target value.",
        ),
    ] = 0.1

    model_config = pydantic.ConfigDict(extra="forbid")

    @pydantic.field_validator("targetOutput")
    @classmethod
    def validate_target_output(cls, value: str) -> str:
        """Reject target identifiers containing only whitespace."""

        value = value.strip()
        if not value:
            raise ValueError("targetOutput cannot be blank")
        return value

    @pydantic.field_validator("nJobs")
    @classmethod
    def validate_n_jobs(cls, value: int) -> int:
        """Reject zero, which scikit-learn does not accept for ``n_jobs``."""

        if value == 0:
            raise ValueError("nJobs cannot be zero")
        return value


def _encode_features(
    frame: pd.DataFrame,
    categorical_columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Encode a feature frame as a finite numeric matrix.

    Args:
        frame: Constitutive-property values for the complete finite pool.
        categorical_columns: Columns that must be represented with one-hot
            indicators. Object and pandas categorical columns are added
            automatically.

    Returns:
        A two-dimensional floating-point feature matrix.

    Raises:
        ValueError: If the frame has no rows or features, names a missing
            categorical column, or cannot be represented by finite numbers.
    """

    if frame.empty:
        raise ValueError("the finite pool must contain at least one entity")
    if frame.shape[1] == 0:
        raise ValueError("the finite pool must contain at least one feature")

    explicit_categorical = list(categorical_columns or [])
    missing_columns = sorted(set(explicit_categorical) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"categorical columns are absent from the feature frame: {missing_columns}"
        )

    inferred_categorical = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    encoded_columns = list(
        dict.fromkeys([*explicit_categorical, *inferred_categorical])
    )
    encoded_frame = pd.get_dummies(
        frame,
        columns=encoded_columns,
        dummy_na=True,
        dtype=float,
    )

    try:
        features = encoded_frame.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "constitutive properties must be numerically encodable"
        ) from error

    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("feature encoding produced an empty matrix")
    if not np.all(np.isfinite(features)):
        raise ValueError("feature encoding produced a non-finite value")
    return np.asarray(features, dtype=float, order="C")


def _encode_leaf_signature(raw_leaves: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Map each tree's raw node identifiers to compact local leaf codes."""

    raw_leaves = np.asarray(raw_leaves)
    if raw_leaves.ndim != 2 or raw_leaves.shape[1] == 0:
        raise ValueError("a forest must produce a leaf for every tree")

    signature = np.empty(raw_leaves.shape, dtype=np.int32)
    leaf_counts: list[int] = []
    for tree_index in range(raw_leaves.shape[1]):
        _, inverse = np.unique(
            raw_leaves[:, tree_index],
            return_inverse=True,
        )
        signature[:, tree_index] = inverse.astype(np.int32, copy=False)
        leaf_counts.append(int(inverse.max()) + 1)
    return signature, leaf_counts


class _SequentialSelectionState(abc.ABC):
    """State shared by strictly sequential finite-pool selectors."""

    def __init__(
        self,
        features: np.ndarray,
        initial_indices: np.ndarray,
        initial_labels: np.ndarray,
        *,
        n_estimators: int,
        min_samples_leaf: int,
        seed: int,
        criterion: ForestCriterion = "squared_error",
        n_jobs: int = 1,
    ) -> None:
        """Validate and retain the complete pool and revealed labels."""

        features = np.asarray(features, dtype=float, order="C")
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must be a nonempty two-dimensional array")
        if features.shape[1] == 0:
            raise ValueError("features must contain at least one column")
        if not np.all(np.isfinite(features)):
            raise ValueError("features contain a non-finite value")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")

        indices = np.asarray(initial_indices, dtype=np.int64).reshape(-1)
        labels = np.asarray(initial_labels, dtype=float).reshape(-1)
        if indices.size != labels.size:
            raise ValueError(
                "initial_indices and initial_labels must have the same length"
            )
        if np.unique(indices).size != indices.size:
            raise ValueError("initial_indices cannot contain duplicates")
        if np.any(indices < 0) or np.any(indices >= features.shape[0]):
            raise ValueError("initial_indices contain an out-of-range index")
        if not np.all(np.isfinite(labels)):
            raise ValueError("initial_labels contain a non-finite value")

        self.features_ = features
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)
        self.criterion = criterion
        self.n_jobs = int(n_jobs)
        self._selected = [int(index) for index in indices]
        self._labels = {
            int(index): float(label)
            for index, label in zip(indices, labels, strict=True)
        }
        self._remaining_mask = np.ones(features.shape[0], dtype=bool)
        self._remaining_mask[indices] = False
        self._pending_index: int | None = None

    @property
    def selected_indices(self) -> tuple[int, ...]:
        """Return indices whose target values have been supplied."""

        return tuple(self._selected)

    @property
    def pending_index(self) -> int | None:
        """Return the selected index awaiting its target value."""

        return self._pending_index

    @property
    def has_candidates(self) -> bool:
        """Return whether at least one unqueried entity remains."""

        return bool(np.any(self._remaining_mask))

    def _candidate_indices(self, candidate_indices: np.ndarray | None) -> np.ndarray:
        """Validate and return a sorted candidate set."""

        if candidate_indices is None:
            candidates = np.flatnonzero(self._remaining_mask)
        else:
            candidates = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
            if np.unique(candidates).size != candidates.size:
                raise ValueError("candidate_indices cannot contain duplicates")
            if np.any(candidates < 0) or np.any(candidates >= self.features_.shape[0]):
                raise ValueError("candidate_indices contain an out-of-range index")
            if np.any(~self._remaining_mask[candidates]):
                raise ValueError("candidate_indices contain a selected entity")
            candidates = np.sort(candidates)
        if candidates.size == 0:
            raise ValueError("no unqueried candidates remain")
        return candidates.astype(np.int64, copy=False)

    def query_one(self, candidate_indices: np.ndarray | None = None) -> int:
        """Select one candidate without accessing its target value.

        Args:
            candidate_indices: Optional subset of currently unqueried rows.

        Returns:
            The selected complete-pool row index.

        Raises:
            RuntimeError: If the preceding query has not been updated.
            ValueError: If the candidate set is invalid or empty.
        """

        if self._pending_index is not None:
            raise RuntimeError("the preceding query must be updated first")
        candidates = self._candidate_indices(candidate_indices)
        if not self._selected:
            choice = int(candidates[0])
        else:
            choice = self._choose_candidate(candidates)
        self._remaining_mask[choice] = False
        self._pending_index = choice
        return choice

    def update(self, index: int, label: float) -> None:
        """Supply the scalar target value for the pending query.

        Args:
            index: Index returned by the latest call to :meth:`query_one`.
            label: Finite scalar target value measured for that entity.

        Raises:
            RuntimeError: If no query is pending or ``index`` does not match it.
            ValueError: If ``label`` is not a finite scalar.
        """

        if self._pending_index is None:
            raise RuntimeError("update received no pending query")
        index = int(index)
        if index != self._pending_index:
            raise RuntimeError("update index does not match the pending query")
        label_array = np.asarray(label, dtype=float)
        if label_array.ndim != 0 or not np.isfinite(label_array.item()):
            raise ValueError("label must be one finite scalar")

        self._labels[index] = float(label_array.item())
        self._selected.append(index)
        self._pending_index = None
        self._on_update(index)

    def _selected_labels(self) -> np.ndarray:
        """Return labels in the selector's acquisition order."""

        return np.asarray([self._labels[index] for index in self._selected])

    @abc.abstractmethod
    def _choose_candidate(self, candidates: np.ndarray) -> int:
        """Return one index from a validated candidate set."""

    @abc.abstractmethod
    def _on_update(self, index: int) -> None:
        """Update cached leaf counts after a target value is supplied."""


@dataclass(frozen=True)
class _FinitePool:
    """Encoded entities and lookup data for one finite discovery space."""

    features: np.ndarray
    entities: tuple[Entity, ...]
    index_by_identifier: dict[str, int]


def _resolve_target(
    measurement_space: MeasurementSpace,
    target_output: str,
) -> ObservedProperty:
    """Resolve a configured target to exactly one observed property.

    Args:
        measurement_space: Measurement space used by the RandomWalk operation.
        target_output: Observed-property identifier, or an unambiguous target
            property identifier.

    Returns:
        The single observed property used to supply labels.

    Raises:
        ValueError: If no property matches, or a target property is measured by
            more than one experiment.
    """

    observed_matches = [
        prop
        for prop in measurement_space.observedProperties
        if prop.identifier == target_output
    ]
    matches = observed_matches or [
        prop
        for prop in measurement_space.observedProperties
        if prop.targetProperty.identifier == target_output
    ]
    if not matches:
        raise ValueError(
            f"targetOutput {target_output!r} is not in the measurement space"
        )
    if len(matches) > 1:
        identifiers = sorted(prop.identifier for prop in matches)
        raise ValueError(
            f"targetOutput {target_output!r} is measured by multiple experiments; "
            f"use one observed-property identifier from {identifiers}"
        )
    return matches[0]


def _finite_pool(discovery_space: DiscoverySpace) -> _FinitePool:
    """Enumerate and encode an explicit finite discovery space."""

    entity_space = discovery_space.entitySpace
    if not isinstance(entity_space, EntitySpaceRepresentation):
        raise ValueError("predictive selectors require an explicit entity space")
    if not entity_space.isDiscreteSpace:
        raise ValueError("predictive selectors require a discrete entity space")
    if entity_space.size <= 0:
        raise ValueError("predictive selectors require a nonempty entity space")

    properties = entity_space.constitutiveProperties
    if not properties:
        raise ValueError("predictive selectors require constitutive properties")
    property_names = [prop.identifier for prop in properties]
    categorical_types = {
        VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE,
    }
    categorical_columns = [
        prop.identifier
        for prop in properties
        if prop.propertyDomain.variableType in categorical_types
    ]

    stored_by_generated_identifier: dict[str, Entity] = {}
    stored_entities = discovery_space.sample_store.get_entities(
        require_measurements=False,
    )
    for stored_entity in stored_entities:
        if not entity_space.isEntityInSpace(stored_entity):
            continue
        stored_point = {
            value.property.identifier: value.value
            for value in stored_entity.constitutive_property_values
        }
        generated_identifier = entity_space.entity_for_point(stored_point).identifier
        if generated_identifier is None:
            raise ValueError("an entity in the finite pool has no identifier")
        if generated_identifier in stored_by_generated_identifier:
            raise ValueError(
                "multiple stored entities represent the same finite-pool point"
            )
        stored_by_generated_identifier[generated_identifier] = stored_entity

    points: list[dict[str, typing.Any]] = []
    entities: list[Entity] = []
    identifiers: dict[str, int] = {}
    for values in entity_space.sequential_point_iterator():
        point = dict(zip(property_names, values, strict=True))
        generated_entity = entity_space.entity_for_point(point)
        generated_identifier = generated_entity.identifier
        if generated_identifier is None:
            raise ValueError("an entity in the finite pool has no identifier")
        entity = stored_by_generated_identifier.get(
            generated_identifier,
            generated_entity,
        )
        if entity.identifier is None:
            raise ValueError("an entity in the finite pool has no identifier")
        if entity.identifier in identifiers:
            raise ValueError(
                f"duplicate entity identifier in finite pool: {entity.identifier}"
            )
        identifiers[entity.identifier] = len(entities)
        points.append(point)
        entities.append(entity)

    features = _encode_features(
        pd.DataFrame(points, columns=property_names),
        categorical_columns=categorical_columns,
    )
    return _FinitePool(
        features=features,
        entities=tuple(entities),
        index_by_identifier=identifiers,
    )


def _entity_target_values(
    entity: Entity,
    target_property: ObservedProperty,
) -> list[float]:
    """Extract finite scalar values matching a target or observed identifier."""

    values: list[float] = []
    for property_value in entity.observedPropertyValues:
        prop = property_value.property
        if prop.identifier != target_property.identifier:
            continue
        try:
            value = np.asarray(property_value.value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"targetOutput {target_property.identifier!r} must contain "
                "finite scalar values"
            ) from error
        if value.ndim != 0 or not np.isfinite(value.item()):
            raise ValueError(
                f"targetOutput {target_property.identifier!r} must contain "
                "finite scalar values"
            )
        values.append(float(value.item()))
    return values


def _measured_labels(
    discovery_space: DiscoverySpace,
    pool: _FinitePool,
    target_property: ObservedProperty,
    indices: Sequence[int] | None = None,
) -> dict[int, float]:
    """Read currently revealed target values from the discovery space."""

    if indices is None:
        identifiers = set(pool.index_by_identifier)
    else:
        identifiers = {
            typing.cast("str", pool.entities[int(index)].identifier)
            for index in indices
        }
    entities = discovery_space.sample_store.get_entities(
        identifiers=identifiers,
        require_measurements=True,
        refresh=True,
    )
    labels: dict[int, float] = {}
    for entity in entities:
        if entity.identifier is None:
            continue
        index = pool.index_by_identifier.get(entity.identifier)
        if index is None:
            continue
        values = _entity_target_values(
            entity,
            target_property,
        )
        if values:
            labels[index] = float(np.mean(values))
    return labels


class _PredictiveSampleSelector(BaseSampler, abc.ABC):
    """ADO adapter for one-query-at-a-time predictive selection state."""

    def __init__(self, parameters: _PredictiveParameters) -> None:
        """Retain the validated custom-sampler parameters."""

        self.params = parameters

    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls,
        remoteDiscoverySpace: DiscoverySpaceManager,
    ) -> bool:
        """Return whether the remote space is explicit, finite, and discrete."""

        entity_space = ray.get(remoteDiscoverySpace.entitySpace.remote())
        if not isinstance(entity_space, EntitySpaceRepresentation):
            return False
        if not entity_space.isDiscreteSpace:
            return False
        try:
            return entity_space.size > 0
        except AttributeError:
            return False

    @abc.abstractmethod
    def _selection_state(
        self,
        features: np.ndarray,
        initial_indices: np.ndarray,
        initial_labels: np.ndarray,
    ) -> _SequentialSelectionState:
        """Create the algorithm-specific state."""

    def _initial_state(
        self,
        discovery_space: DiscoverySpace,
    ) -> tuple[_FinitePool, ObservedProperty, _SequentialSelectionState]:
        """Build the finite pool and initialize it from revealed labels."""

        pool = _finite_pool(discovery_space)
        target_property = _resolve_target(
            discovery_space.measurementSpace,
            self.params.targetOutput,
        )
        labels = _measured_labels(
            discovery_space,
            pool,
            target_property,
        )
        indices = np.asarray(sorted(labels), dtype=np.int64)
        initial_labels = np.asarray([labels[index] for index in indices])
        state = self._selection_state(pool.features, indices, initial_labels)
        return pool, target_property, state

    def _update_pending(
        self,
        discovery_space: DiscoverySpace,
        pool: _FinitePool,
        target_property: ObservedProperty,
        state: _SequentialSelectionState,
    ) -> bool:
        """Update pending state if its target value is now available."""

        pending = state.pending_index
        if pending is None:
            return True
        labels = _measured_labels(
            discovery_space,
            pool,
            target_property,
            indices=[pending],
        )
        if pending not in labels:
            return False
        state.update(pending, labels[pending])
        return True

    async def _wait_for_pending(
        self,
        discovery_space: DiscoverySpace,
        pool: _FinitePool,
        target_property: ObservedProperty,
        state: _SequentialSelectionState,
    ) -> None:
        """Wait until the preceding query has a revealed target value."""

        deadline = time.monotonic() + self.params.labelWaitTimeoutSeconds
        while not self._update_pending(
            discovery_space,
            pool,
            target_property,
            state,
        ):
            if time.monotonic() >= deadline:
                pending = state.pending_index
                identifier = (
                    pool.entities[pending].identifier if pending is not None else None
                )
                raise TimeoutError(
                    f"timed out waiting for targetOutput "
                    f"{self.params.targetOutput!r} on entity {identifier!r}"
                )
            await asyncio.sleep(self.params.labelPollIntervalSeconds)

    def _sync_iterator(
        self,
        discovery_space: DiscoverySpace,
    ) -> typing.Generator[list[Entity], None, None]:
        """Yield entities synchronously, requiring an update between queries."""

        pool, target_property, state = self._initial_state(discovery_space)
        while state.pending_index is not None or state.has_candidates:
            if state.pending_index is not None and not self._update_pending(
                discovery_space,
                pool,
                target_property,
                state,
            ):
                pending = state.pending_index
                identifier = pool.entities[pending].identifier
                raise RuntimeError(
                    f"targetOutput {self.params.targetOutput!r} has not been "
                    f"measured for entity {identifier!r}"
                )
            if not state.has_candidates:
                break
            yield [pool.entities[state.query_one()]]

    async def _async_iterator(
        self,
        discovery_space: DiscoverySpace,
    ) -> typing.AsyncGenerator[list[Entity], None]:
        """Yield entities asynchronously after each previous label arrives."""

        pool, target_property, state = self._initial_state(discovery_space)
        while state.pending_index is not None or state.has_candidates:
            if state.pending_index is not None:
                await self._wait_for_pending(
                    discovery_space,
                    pool,
                    target_property,
                    state,
                )
            if not state.has_candidates:
                break
            yield [pool.entities[state.query_one()]]

    def entityIterator(
        self,
        discoverySpace: DiscoverySpace,
        batchsize: int = 1,
    ) -> typing.Generator[list[Entity], None, None]:
        """Return a synchronous iterator over adaptive single-entity queries."""

        if batchsize != 1:
            raise ValueError("predictive selectors require batchsize=1")
        return self._sync_iterator(discoverySpace)

    async def remoteEntityIterator(
        self,
        remoteDiscoverySpace: DiscoverySpaceManager,
        batchsize: int = 1,
    ) -> typing.AsyncGenerator[list[Entity], None]:
        """Return an asynchronous iterator over adaptive single-entity queries."""

        if batchsize != 1:
            raise ValueError("predictive selectors require batchsize=1")
        discovery_space = await remoteDiscoverySpace.discoverySpace.remote()
        return self._async_iterator(discovery_space)


def _random_walk_parameters(
    parameters: _PredictiveParameters,
    *,
    number_entities: int,
    sampler_class: type[_PredictiveSampleSelector],
) -> RandomWalkParameters:
    """Build the fixed sequential RandomWalk configuration for a selector."""

    from ado.modules.operators.randomwalk import (
        CustomSamplerConfiguration,
        EntityFilter,
        FilterModeEnum,
        RandomWalkParameters,
        SamplerModuleConf,
    )

    sampler_configuration = CustomSamplerConfiguration(
        module=SamplerModuleConf(
            moduleClass=sampler_class.__name__,
            moduleName=sampler_class.__module__,
        ),
        parameters=parameters.model_dump(exclude={"numberEntities"}),
    )
    return RandomWalkParameters(
        samplerConfig=sampler_configuration,
        numberEntities=number_entities,
        batchSize=1,
        singleMeasurement=True,
        filter=EntityFilter(filterMode=FilterModeEnum.noFilter),
    )


def _outer_operation_output(nested_output: OperationOutput) -> OperationOutput:
    """Expose a nested RandomWalk operation as a resource of the outer one."""

    return OperationOutput(
        other=[],
        resources=[nested_output.operation] if nested_output.operation else [],
        metadata={},
    )


def _run_predictive_operator(
    *,
    discovery_space: DiscoverySpace,
    operation_info: FunctionOperationInfo | None,
    parameters: _PredictiveParameters,
    number_entities: int,
    sampler_class: type[_PredictiveSampleSelector],
) -> OperationOutput:
    """Run RandomWalk with a plugin-local sequential adaptive sampler."""

    # Lazy import avoids a circular import while ADO loads operator plugins.
    from ado.modules.operators.collections import explore

    random_walk_parameters = _random_walk_parameters(
        parameters,
        number_entities=number_entities,
        sampler_class=sampler_class,
    )
    random_walk = explore.operators["random_walk"].function
    nested_operation_info = FunctionOperationInfo(
        actuatorConfigurationIdentifiers=list(
            operation_info.actuatorConfigurationIdentifiers
            if operation_info is not None
            else []
        )
    )
    nested_output = random_walk(
        discoverySpace=discovery_space,
        operationInfo=nested_operation_info,
        **random_walk_parameters.model_dump(),
    )
    return _outer_operation_output(nested_output)
