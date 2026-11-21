# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""FLORA's parameter validation, label reading and feature encoding."""

import numpy as np
import pandas as pd
import pytest
from active_learning.data import candidate_entities
from active_learning.flora.operator import _require_autogluon_labels
from active_learning.flora.parameters import FLORAParameters
from active_learning.flora.sampler import FLORASampleSelector, encode_forest_features
from pydantic import ValidationError

from ado.core.discoveryspace.space import DiscoverySpace
from ado.schema.domain import PropertyDomain
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.experiment import Experiment
from ado.schema.observed_property import ObservedPropertyValue
from ado.schema.property import ConcretePropertyDescriptor, ConstitutiveProperty
from ado.schema.result import ValidMeasurementResult


@pytest.mark.parametrize(
    "overrides",
    [{"epochLength": 3}, {"seed": None}, {"pilotSize": 5.0}],
)
def test_parameters_reject_settings_flora_does_not_define(
    overrides: dict[str, object],
) -> None:
    """FLORA schedules its own refits and requires a seeded integer pilot."""
    with pytest.raises(ValidationError):
        FLORAParameters(targetOutput="response", numberEntities=3, **overrides)


def test_parameters_round_trip() -> None:
    """Dumped parameters recreate an equal model."""
    parameters = FLORAParameters(targetOutput="response", numberEntities=3, seed=7)
    assert FLORAParameters(**parameters.model_dump()) == parameters


def test_autogluon_requires_four_labels() -> None:
    """AutoGluon needs enough labels for its training and validation sets."""
    with pytest.raises(ValueError, match="at least four"):
        _require_autogluon_labels(3)


def test_target_label_averages_measurements_and_rejects_non_finite() -> None:
    """Repeated measurements are averaged and unusable values raise."""
    count = ConstitutiveProperty(
        identifier="count", propertyDomain=PropertyDomain(values=[0, 1])
    )
    experiment = Experiment(
        identifier="selected",
        actuatorIdentifier="test",
        requiredProperties=(count,),
        targetProperties=[ConcretePropertyDescriptor(identifier="response")],
    )
    observed = experiment.observedProperties[0]
    entity_space = EntitySpaceRepresentation(constitutiveProperties=[count])
    entity = entity_space.entity_for_point({"count": 0})
    assert FLORASampleSelector.target_label(entity, observed.identifier) is None
    for value in (1.0, 4.0):
        entity.add_measurement_result(
            ValidMeasurementResult(
                entityIdentifier=entity.identifier,
                measurements=[ObservedPropertyValue(property=observed, value=value)],
            )
        )
    assert FLORASampleSelector.target_label(entity, observed.identifier) == 2.5
    entity.add_measurement_result(
        ValidMeasurementResult(
            entityIdentifier=entity.identifier,
            measurements=[ObservedPropertyValue(property=observed, value=np.nan)],
        )
    )
    with pytest.raises(ValueError, match="finite scalar"):
        FLORASampleSelector.target_label(entity, observed.identifier)


def test_forest_features_one_hot_levels_and_reject_unknown_category() -> None:
    """Categories keep their fit-time columns, with a column for missing values."""
    categories = {"kind": pd.CategoricalDtype(["a", "b"])}
    frame = pd.DataFrame({"kind": ["b", None], "size": [1.0, 2.0]})
    encoded = encode_forest_features(frame, categories=categories)
    # Columns: size, kind=a, kind=b, kind=missing.
    assert encoded.tolist() == [[1.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="Unknown category in feature 'kind'"):
        encode_forest_features(
            pd.DataFrame({"kind": ["c"], "size": [1.0]}), categories=categories
        )


@pytest.mark.parametrize("categorical_space", [[10, 20, 30, 40]], indirect=True)
def test_numeric_categories_have_no_ordered_distance(
    categorical_space: DiscoverySpace,
) -> None:
    """Declared numeric categories are equally distant in the encoded feature space."""
    entities = candidate_entities(categorical_space)
    frame, features, encoder = FLORASampleSelector.encode_feature_matrix(
        categorical_space, entities
    )
    same_count = features[frame["count"] == 0]
    assert np.square(same_count[1:] - same_count[0]).sum(axis=1).tolist() == [
        2.0,
        2.0,
        2.0,
    ]
    reordered = frame.iloc[[6, 0]].loc[:, ["count", "kind"]]
    np.testing.assert_array_equal(encoder.transform(reordered), features[[6, 0]])
