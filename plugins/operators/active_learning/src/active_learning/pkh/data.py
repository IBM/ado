# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Entity pool, feature and label helpers shared by the PKH sampler and operator."""

import contextlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from ado.core.discoveryspace.samplers import (
    ExplicitEntitySpaceGridSampleGenerator,
    WalkModeEnum,
)
from ado.core.discoveryspace.space import DiscoverySpace, Entity
from ado.schema.domain import VariableTypeEnum
from ado.schema.observed_property import ObservedProperty


def candidate_entities(discovery_space: DiscoverySpace) -> list[Entity]:
    """Return one entity per point of the finite entity space.

    Reuse entities from the sample store and create the remaining candidates
    in memory.
    """
    generator = ExplicitEntitySpaceGridSampleGenerator(mode=WalkModeEnum.SEQUENTIAL)
    entities = [
        entity
        for batch in generator.entityIterator(discovery_space, batchsize=1)
        for entity in batch
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


def target_label(entity: Entity, target_name: str) -> float | None:
    """Return the first finite scalar measurement of the exact observed property."""
    for value in entity.valuesForObservedPropertyIdentifier(target_name):
        with contextlib.suppress(ValueError, TypeError, OverflowError):
            if np.isfinite(label := float(value.value)):
                return label
    return None


def measured_labels(
    discovery_space: DiscoverySpace, target_name: str
) -> dict[str, float]:
    """Refresh and return finite labels of the observed property, by entity identifier."""
    discovery_space.sample_store.get_entities(require_measurements=True, refresh=True)
    return {
        entity.identifier: label
        for entity in discovery_space.matchingEntities(require_measurements=True)
        if (label := target_label(entity, target_name)) is not None
    }


def _format_categorical_value(value: object) -> str:
    """Format a categorical value as a string for consistent discrete encoding."""
    return value if isinstance(value, str) else str(value)


def encode_feature_matrix(
    discovery_space: DiscoverySpace,
    entities: list[Entity],
) -> tuple[pd.DataFrame, np.ndarray, ColumnTransformer]:
    """Encode constitutive values in the discovery space's declared feature order."""
    properties = discovery_space.entitySpace.constitutiveProperties
    raw_columns = {
        prop.identifier: [
            entity.valueForConstitutivePropertyIdentifier(prop.identifier).value
            for entity in entities
        ]
        for prop in properties
    }
    categorical_columns = [
        prop.identifier
        for prop in properties
        if prop.propertyDomain.variableType
        == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE
        or any(not isinstance(v, (int, float)) for v in raw_columns[prop.identifier])
    ]
    frame = pd.DataFrame(
        {
            prop.identifier: (
                pd.Series(
                    [
                        _format_categorical_value(v)
                        for v in raw_columns[prop.identifier]
                    ],
                    dtype="category",
                )
                if prop.identifier in categorical_columns
                else pd.Series(raw_columns[prop.identifier])
            )
            for prop in properties
        }
    )
    categorical_encoder = make_pipeline(
        FunctionTransformer(
            pd.DataFrame.map,
            kw_args={"func": _format_categorical_value},
            feature_names_out="one-to-one",
        ),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    encoder = ColumnTransformer(
        [("categorical", categorical_encoder, categorical_columns)],
        remainder="passthrough",
    )
    features = np.ascontiguousarray(encoder.fit_transform(frame), dtype=float)
    return frame, features, encoder
