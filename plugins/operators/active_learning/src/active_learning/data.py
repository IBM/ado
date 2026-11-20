# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Entities, measurements, and feature matrices for finite discovery spaces."""

import contextlib
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from ado.core.discoveryspace.samplers import (
    ExplicitEntitySpaceGridSampleGenerator,
    WalkModeEnum,
)
from ado.core.discoveryspace.space import DiscoverySpace, Entity
from ado.schema.observed_property import ObservedProperty


def candidate_entities(discovery_space: DiscoverySpace) -> list[Entity]:
    """Return one entity per point of the finite entity space.

    Entities already in the sample store are reused and the others are created
    (not stored), so this also works on an empty sample store.
    """
    generator = ExplicitEntitySpaceGridSampleGenerator(mode=WalkModeEnum.SEQUENTIAL)
    entities = [
        entity
        for batch in generator.entityIterator(discovery_space, batchsize=1)
        for entity in batch
    ]
    if len({entity.identifier for entity in entities}) != len(entities):
        raise ValueError(
            "Active learning requires unique entity identifiers. Choose property values that "
            "produce distinct identifiers; numeric 1 and string '1' collide."
        )
    return entities


def validate_target_output(
    target_output: str, discovery_space: DiscoverySpace
) -> ObservedProperty:
    """Resolve a unique target in a measurement space containing one experiment."""
    number_experiments = len(discovery_space.measurementSpace.experimentReferences)
    if number_experiments != 1:
        raise ValueError(
            f"Active learning requires exactly 1 experiment; found {number_experiments}."
        )
    matches = [
        prop
        for prop in discovery_space.measurementSpace.observedProperties
        if target_output in (prop.identifier, prop.targetProperty.identifier)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"targetOutput '{target_output}' must identify exactly one observed property; "
            "use its qualified identifier when target names are ambiguous."
        )
    features = (
        discovery_space.entitySpace.constitutiveProperties
        if discovery_space.entitySpace
        else []
    )
    feature_names = {prop.identifier for prop in features}
    reserved = {"entityIdentifier", "is_predicted"}
    collisions = feature_names & reserved
    target_name = matches[0].targetProperty.identifier
    if target_name in feature_names | reserved:
        collisions.add(target_name)
    if collisions:
        raise ValueError(
            f"The output column names conflict: {sorted(collisions)}. Rename these properties."
        )
    return matches[0]


def target_label(entity: Entity, target_name: str) -> float | None:
    """Return the first finite scalar measurement of the exact observed property."""
    for value in entity.valuesForObservedPropertyIdentifier(target_name):
        with contextlib.suppress(ValueError, TypeError, OverflowError):
            if np.isfinite(label := float(value.value)):
                return label
    return None


def measured_labels(
    discovery_space: DiscoverySpace,
    target_name: str,
    read_label: Callable[[Entity, str], float | None] = target_label,
) -> dict[str, float]:
    """Refresh and return finite labels of the observed property, by entity identifier."""
    discovery_space.sample_store.get_entities(require_measurements=True, refresh=True)
    return {
        entity.identifier: label
        for entity in discovery_space.matchingEntities(require_measurements=True)
        if (label := read_label(entity, target_name)) is not None
    }


def feature_frame(
    discovery_space: DiscoverySpace,
    entities: list[Entity],
    categorical_columns: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Read features in declared order and mark the categorical dimensions."""
    frame = pd.DataFrame(
        [entity.seriesRepresentation(constitutiveOnly=True) for entity in entities],
        columns=[
            prop.identifier
            for prop in discovery_space.entitySpace.constitutiveProperties
        ],
    )
    categories = list(
        dict.fromkeys([*categorical_columns, *frame.select_dtypes(exclude="number")])
    )
    frame[categories] = frame[categories].astype("category")
    return frame, categories


def fit_feature_encoder(
    frame: pd.DataFrame, encoder: ColumnTransformer
) -> tuple[pd.DataFrame, np.ndarray, ColumnTransformer]:
    """Fit the encoder and return the raw rows, their numeric features, and the encoder."""
    features = np.ascontiguousarray(encoder.fit_transform(frame), dtype=float)
    return frame, features, encoder
