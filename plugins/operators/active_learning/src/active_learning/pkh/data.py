# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Entity pool, feature and label helpers shared by the PKH sampler and operator."""

import contextlib
import math
from typing import TYPE_CHECKING

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
from ado.schema.observed_property import ObservedProperty

if TYPE_CHECKING:
    from ado.schema.property_value import ConstitutivePropertyValue


def validate_candidate_space(entity_space: EntitySpaceRepresentation | None) -> int:
    """Return the size of an explicit grid with finite property domains."""
    if not isinstance(entity_space, EntitySpaceRepresentation):
        raise ValueError("PKH requires an explicit finite entity space.")
    count = 1
    for prop in entity_space.constitutiveProperties:
        size = prop.propertyDomain.size
        if size == math.inf:
            raise ValueError(
                f"PKH requires a finite domain for constitutive property '{prop.identifier}'."
            )
        count *= int(size)
    return count


def candidate_entities(discovery_space: DiscoverySpace) -> list[Entity]:
    """Return one entity per point of the finite entity space.

    Reuse entities from the sample store and create the remaining candidates
    in memory.
    """
    generator = ExplicitEntitySpaceGridSampleGenerator(mode=WalkModeEnum.SEQUENTIAL)
    candidates = [
        entity
        for batch in generator.entitySpaceIterator(discovery_space.entitySpace)
        for entity in batch
    ]
    stored_by_point: dict[frozenset[ConstitutivePropertyValue], Entity | None] = {
        frozenset(candidate.constitutive_property_values): None
        for candidate in candidates
    }
    for stored in discovery_space.sample_store.get_entities(require_measurements=False):
        point = frozenset(stored.constitutive_property_values)
        if point not in stored_by_point:
            continue
        if (previous := stored_by_point[point]) is not None:
            raise ValueError(
                "Multiple stored entities match one PKH grid point: "
                f"'{previous.identifier}' and '{stored.identifier}'."
            )
        stored_by_point[point] = stored

    entities = [
        stored_by_point[frozenset(candidate.constitutive_property_values)] or candidate
        for candidate in candidates
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
    """Find the observed property that PKH will model.

    Args:
        target_output: Qualified observed-property identifier or bare target name.
        discovery_space: Discovery space containing the experiment and its inputs.

    Returns:
        The matching observed property.

    Raises:
        ValueError: The space contains zero or multiple experiments, or the
            target matches zero or multiple observed properties. This error
            also occurs when the target and a feature share a name, or when
            either uses the reserved name ``entityIdentifier`` or ``is_predicted``.
    """
    number_experiments = len(discovery_space.measurementSpace.experimentReferences)
    if number_experiments != 1:
        raise ValueError(
            f"PKH requires exactly 1 experiment; found {number_experiments}."
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
            f"PKH output column names conflict: {sorted(collisions)}. Rename these properties."
        )
    return matches[0]


def target_label(entity: Entity, target: ObservedProperty) -> float | None:
    """Return the first finite scalar measurement of the exact observed property."""
    for value in entity.observedPropertyValues:
        if value.property != target:
            continue
        with contextlib.suppress(ValueError, TypeError, OverflowError):
            if np.isfinite(label := float(value.value)):
                return label
    return None


def measured_labels(
    discovery_space: DiscoverySpace, target: ObservedProperty
) -> dict[str, float]:
    """Refresh and return finite labels of the observed property, by entity identifier."""
    discovery_space.sample_store.get_entities(require_measurements=True, refresh=True)
    return {
        entity.identifier: label
        for entity in discovery_space.matchingEntities(require_measurements=True)
        if (label := target_label(entity, target)) is not None
    }


def feature_frame(
    discovery_space: DiscoverySpace,
    entities: list[Entity],
) -> pd.DataFrame:
    """Build candidate features in the discovery space's declared property order."""
    columns = {}
    for prop in discovery_space.entitySpace.constitutiveProperties:
        values = [
            entity.valueForConstitutivePropertyIdentifier(prop.identifier).value
            for entity in entities
        ]
        if (
            prop.propertyDomain.variableType
            == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE
            or any(not isinstance(value, (int, float)) for value in values)
        ):
            columns[prop.identifier] = pd.Series(
                [str(value) for value in values], dtype="category"
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
