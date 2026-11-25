# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Entity pool, feature and label helpers shared by the PKH sampler and operator."""

import contextlib
import json
import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from ado.core.discoveryspace.samplers import (
    ExplicitEntitySpaceGridSampleGenerator,
    WalkModeEnum,
)
from ado.core.discoveryspace.space import DiscoverySpace, Entity
from ado.schema.domain import VariableTypeEnum
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.observed_property import ObservedProperty, ObservedPropertyValue
from ado.schema.property_value import ConstitutivePropertyValue, ValueTypeEnum
from ado.schema.result import ValidMeasurementResult

# SQLite accepts at least 999 bind parameters per statement on older builds.
_IDENTIFIER_CHUNK_SIZE = 500


def validate_candidate_space(entity_space: EntitySpaceRepresentation | None) -> int:
    """Return the size of a finite grid with coordinates that survive storage."""
    if not isinstance(entity_space, EntitySpaceRepresentation):
        raise ValueError("PKH requires an explicit finite entity space.")
    properties = entity_space.constitutiveProperties
    if not properties:
        raise ValueError("PKH requires at least one constitutive property.")

    count = 1
    for prop in properties:
        domain = prop.propertyDomain
        size = domain.size
        if size == math.inf:
            raise ValueError(
                "PKH requires a finite domain for constitutive property "
                f"'{prop.identifier}'."
            )
        for value in domain.values or ():
            try:
                coordinate = ConstitutivePropertyValue(property=prop, value=value)
                restored = ConstitutivePropertyValue.model_validate_json(
                    coordinate.model_dump_json(exclude_defaults=True)
                )
                if coordinate != restored:
                    raise ValueError("The coordinate changed during serialization.")
            except ValueError as error:
                raise ValueError(
                    f"PKH coordinate {value!r} for '{prop.identifier}' "
                    "cannot be stored unchanged."
                ) from error
        count *= int(size)
    return count


def coordinate_value_key(value: object) -> object:
    """Make a coordinate hashable, retaining vector order and mapping equality."""
    if isinstance(value, list):
        return tuple(coordinate_value_key(item) for item in value)
    if isinstance(value, dict):
        return frozenset(
            (key, coordinate_value_key(item)) for key, item in value.items()
        )
    return value


def _point_key(entity: Entity) -> frozenset[tuple[str, object]]:
    """Index an entity by named coordinates independently of property order."""
    return frozenset(
        (entry.property.identifier, coordinate_value_key(entry.value))
        for entry in entity.constitutive_property_values
    )


def candidate_entities(discovery_space: DiscoverySpace) -> list[Entity]:
    """Enumerate the grid, refreshing stored entities to retain their identifiers."""
    generator = ExplicitEntitySpaceGridSampleGenerator(mode=WalkModeEnum.SEQUENTIAL)
    candidates = [
        entity
        for batch in generator.entitySpaceIterator(discovery_space.entitySpace)
        for entity in batch
    ]
    points = [_point_key(candidate) for candidate in candidates]
    stored_by_point: dict[frozenset[tuple[str, object]], Entity | None] = dict.fromkeys(
        points
    )
    if len(stored_by_point) != len(points):
        raise ValueError("PKH requires unique grid coordinates.")
    for stored in discovery_space.sample_store.get_entities(
        require_measurements=False, refresh=True
    ):
        point = _point_key(stored)
        if point not in stored_by_point:
            continue
        if (previous := stored_by_point[point]) is not None:
            raise ValueError(
                "Multiple stored entities match one PKH grid point: "
                f"'{previous.identifier}' and '{stored.identifier}'."
            )
        stored_by_point[point] = stored

    entities = [
        stored_by_point[point] or candidate
        for candidate, point in zip(candidates, points, strict=True)
    ]
    if len({entity.identifier for entity in entities}) != len(entities):
        raise ValueError(
            "PKH requires unique entity identifiers. Choose property values that "
            "produce distinct identifiers; numeric 1 and string '1' collide."
        )
    return entities


def validate_target_output(
    target_output: str, discovery_space: DiscoverySpace
) -> ObservedProperty:
    """Resolve a qualified or unambiguous bare target and check result column names.

    Raises:
        ValueError: The space must have one experiment and one matching target.
            Feature and target names must be distinct and must leave
            ``entityIdentifier`` and ``is_predicted`` available for PKH output.
    """
    number_experiments = len(discovery_space.measurementSpace.experimentReferences)
    if number_experiments != 1:
        raise ValueError(
            f"PKH requires exactly 1 experiment; found {number_experiments}."
        )
    properties = discovery_space.measurementSpace.observedProperties
    matches = [prop for prop in properties if prop.identifier == target_output] or [
        prop for prop in properties if prop.targetProperty.identifier == target_output
    ]
    if len(matches) != 1:
        raise ValueError(
            f"targetOutput '{target_output}' must identify exactly one observed "
            "property; use its qualified identifier when target names are ambiguous."
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
            f"PKH output column names conflict: {sorted(collisions)}. "
            "Rename these properties."
        )
    return matches[0]


def target_label(
    measurements: Iterable[ObservedPropertyValue], target: ObservedProperty
) -> int | float | None:
    """Return the first finite numeric measurement of the exact observed property."""
    for value in measurements:
        if (
            value.property != target
            or value.valueType != ValueTypeEnum.NUMERIC_VALUE_TYPE
        ):
            continue
        with contextlib.suppress(ValueError, TypeError, OverflowError):
            if np.isfinite(label := float(value.value)):
                return value.value if isinstance(value.value, int) else label
    return None


def measured_labels(
    discovery_space: DiscoverySpace,
    target: ObservedProperty,
    entities: list[Entity] | None = None,
) -> dict[str, int | float]:
    """Refresh and return finite labels of the observed property, by entity identifier.

    Pass the current candidate pool to reuse its identifiers and coordinates.
    """
    if entities is None:
        entities = candidate_entities(discovery_space)
    labels = {}
    for start in range(0, len(entities), _IDENTIFIER_CHUNK_SIZE):
        chunk = {
            entity.identifier: entity
            for entity in entities[start : start + _IDENTIFIER_CHUNK_SIZE]
        }
        for entity in discovery_space.sample_store.get_entities(
            identifiers=set(chunk), require_measurements=True, refresh=True
        ):
            if _point_key(entity) != _point_key(chunk[entity.identifier]):
                raise ValueError(
                    f"Stored entity '{entity.identifier}' has different coordinates "
                    "from the PKH candidate."
                )
            if (
                label := target_label(entity.observedPropertyValues, target)
            ) is not None:
                labels[entity.identifier] = label
    return labels


def walk_measurement_labels(
    discovery_space: DiscoverySpace,
    target: ObservedProperty,
    operation_id: str,
    *,
    entity_identifier: str | None = None,
) -> dict[str, int | float | None]:
    """Read the walk's targets, with None for results lacking a finite target."""
    filters = (
        [{"entityIdentifier": json.dumps(entity_identifier)}]
        if entity_identifier is not None
        else None
    )
    labels = {}
    for result in discovery_space.sample_store.measurement_results_for_operation(
        operation_id, filters=filters
    ):
        labels.setdefault(result.entityIdentifier, None)
        if (
            isinstance(result, ValidMeasurementResult)
            and (label := target_label(result.measurements, target)) is not None
        ):
            labels[result.entityIdentifier] = label
    return labels


def candidate_coordinates(entities: list[Entity]) -> dict[str, list]:
    """Collect coordinates in candidate order, encoding blobs in ADO's JSON format."""
    columns: dict[str, list] = {}
    for entity in entities:
        for coordinate in entity.constitutive_property_values:
            value = coordinate.value
            if isinstance(value, bytes):
                value = coordinate.model_dump(
                    mode="json", include={"valueType", "value"}
                )
            columns.setdefault(coordinate.property.identifier, []).append(value)
    return columns


def feature_frame(
    discovery_space: DiscoverySpace,
    coordinates: dict[str, list],
) -> pd.DataFrame:
    """Build model inputs with one categorical code per distinct coordinate value."""
    columns = {}
    for prop in discovery_space.entitySpace.constitutiveProperties:
        values = coordinates[prop.identifier]
        if (
            prop.propertyDomain.variableType
            == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE
            or any(not isinstance(value, (int, float)) for value in values)
        ):
            codes: dict[object, int] = {}
            columns[prop.identifier] = pd.Categorical(
                [
                    codes.setdefault(coordinate_value_key(value), len(codes))
                    for value in values
                ]
            )
        else:
            columns[prop.identifier] = pd.Series(values)
    return pd.DataFrame(columns)


def encode_feature_matrix(frame: pd.DataFrame) -> np.ndarray | csr_matrix:
    """One-hot encode features for a forest, using sparse storage when possible."""
    categorical_columns = frame.select_dtypes(include="category").columns.tolist()
    encoder = ColumnTransformer(
        [("categorical", OneHotEncoder(), categorical_columns)],
        remainder="passthrough",
    )
    features = encoder.fit_transform(frame).astype(float, copy=False)
    # Scikit-learn's forest supports missing numeric values only in dense input.
    if isinstance(features, csr_matrix) and np.isnan(features.data).any():
        return features.toarray()
    return features
