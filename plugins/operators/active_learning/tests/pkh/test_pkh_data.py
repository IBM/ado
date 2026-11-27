# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""PKH candidate identity, measured-label lookup, and forest input encoding."""

import sqlite3
from collections.abc import Callable

import pytest
from active_learning.pkh.data import (
    candidate_coordinates,
    candidate_entities,
    encode_feature_matrix,
    feature_frame,
    measured_labels,
    validate_candidate_space,
    validate_target_output,
)
from scipy.sparse import issparse

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.samplestore.sql import SQLSampleStore
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.experiment import Experiment
from ado.schema.measurementspace import MeasurementSpace, MeasurementSpaceConfiguration
from ado.schema.observed_property import ObservedPropertyValue
from ado.schema.property import ConcretePropertyDescriptor, ConstitutiveProperty
from ado.schema.result import ValidMeasurementResult


def test_grid_requires_a_constitutive_property() -> None:
    """Reject a zero-dimensional grid before invoking RandomWalk."""
    space = EntitySpaceRepresentation(constitutiveProperties=[])
    with pytest.raises(ValueError, match="at least one constitutive property"):
        validate_candidate_space(space)


def test_grid_rejects_unbounded_domain() -> None:
    """Reject an unbounded domain before starting measurements."""
    prop = ConstitutiveProperty(
        identifier="temperature",
        propertyDomain=PropertyDomain(
            variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE, interval=1
        ),
    )
    space = EntitySpaceRepresentation(constitutiveProperties=[prop])
    with pytest.raises(ValueError, match=r"finite domain.*temperature"):
        validate_candidate_space(space)


def test_grid_rejects_empty_domain() -> None:
    """Reject an empty grid before starting measurements."""
    prop = ConstitutiveProperty(
        identifier="temperature", propertyDomain=PropertyDomain(values=[])
    )
    space = EntitySpaceRepresentation(constitutiveProperties=[prop])
    with pytest.raises(ValueError, match=r"empty domain.*temperature"):
        validate_candidate_space(space)


@pytest.mark.parametrize(
    "domain",
    [
        PropertyDomain(values=[1e38, -1e39]),
        PropertyDomain(domainRange=[0, 10**40], interval=10**39),
    ],
    ids=["explicit-values", "interval"],
)
def test_grid_rejects_coordinates_outside_forest_range(domain: PropertyDomain) -> None:
    """Reject numeric coordinates outside the sampling forest's supported range."""
    prop = ConstitutiveProperty(identifier="reading", propertyDomain=domain)
    space = EntitySpaceRepresentation(constitutiveProperties=[prop])
    with pytest.raises(ValueError, match=r"reading.*float32"):
        validate_candidate_space(space)


@pytest.mark.parametrize(
    "value",
    [float("nan"), [b"code"]],
    ids=["nan", "vector-blob"],
)
def test_grid_requires_coordinates_that_survive_storage(
    value: float | list[bytes],
) -> None:
    """Reject coordinates whose stored representation cannot retain their value."""
    prop = ConstitutiveProperty(
        identifier="code", propertyDomain=PropertyDomain(values=[1, value])
    )
    space = EntitySpaceRepresentation(constitutiveProperties=[prop])
    with pytest.raises(ValueError, match=r"'code'.*stored unchanged"):
        validate_candidate_space(space)


def test_grid_requires_a_unique_identifier_for_each_coordinate(
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> None:
    """Resolve colliding generated IDs when another writer supplies distinct IDs."""
    prop = ConstitutiveProperty(
        identifier="code",
        propertyDomain=PropertyDomain(values=[1, "1", [{"a": [2], "b": 3}]]),
    )
    space = make_pkh_space([prop])
    assert validate_candidate_space(space.entitySpace) == 3
    with pytest.raises(ValueError, match="unique entity identifiers"):
        candidate_entities(space)

    point = space.entitySpace.entity_for_point
    number = point({"code": 1}).model_copy(update={"identifier": "stored-number"})
    vector = point({"code": [{"b": 3, "a": [2]}]}).model_copy(
        update={"identifier": "stored-vector"}
    )
    writer = SQLSampleStore(
        identifier=space.sample_store.identifier,
        storageLocation=space.project_context.metadataStore,
        parameters={},
    )
    writer.addEntities([number, vector])
    assert [entity.identifier for entity in candidate_entities(space)] == [
        number.identifier,
        point({"code": "1"}).identifier,
        vector.identifier,
    ]
    writer.addEntities([number.model_copy(update={"identifier": "second-number"})])
    with pytest.raises(ValueError, match="Multiple stored entities"):
        candidate_entities(space)


def test_label_id_must_match_candidate_coordinates(
    categorical_space: DiscoverySpace,
) -> None:
    """A stored string must not supply the label for an integer with the same ID."""
    candidate = categorical_space.entitySpace.entity_for_point({"code": 1})
    target = validate_target_output("self", categorical_space)
    with pytest.raises(ValueError, match="different coordinates"):
        measured_labels(categorical_space, target, [candidate])


def test_grid_rejects_duplicate_coordinates_before_measurement(
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> None:
    """Distinct generated IDs cannot hide duplicate coordinates in an empty store."""
    prop = ConstitutiveProperty(
        identifier="code", propertyDomain=PropertyDomain(values=[1, 1.0])
    )
    with pytest.raises(ValueError, match="unique grid coordinates"):
        candidate_entities(make_pkh_space([prop]))


def test_qualified_target_takes_precedence(categorical_space: DiscoverySpace) -> None:
    """Resolve a qualified name that is also another property's bare name."""
    target = validate_target_output("selected-self", categorical_space)
    assert target.targetProperty.identifier == "self"


def test_rejects_reserved_target_name(categorical_space: DiscoverySpace) -> None:
    """Reserve the prediction marker column for PKH output."""
    with pytest.raises(ValueError, match="output column names conflict"):
        validate_target_output("is_predicted", categorical_space)


def test_rejects_duplicate_target_names(categorical_space: DiscoverySpace) -> None:
    """Reject multiple outputs with the same name in one experiment."""
    experiment = Experiment(
        identifier="selected",
        actuatorIdentifier="test",
        targetProperties=[ConcretePropertyDescriptor(identifier="pressure")] * 2,
    )
    space = DiscoverySpace(
        project_context=categorical_space.project_context,
        sample_store=categorical_space.sample_store,
        measurementSpace=MeasurementSpace(
            MeasurementSpaceConfiguration(experiments=[experiment])
        ),
    )
    with pytest.raises(ValueError, match="must identify exactly one observed property"):
        validate_target_output("pressure", space)


def test_numeric_text_leaves_candidate_unlabeled(
    categorical_space: DiscoverySpace,
) -> None:
    """Ignore text measurements even when they contain digits."""
    candidate = categorical_space.entitySpace.entity_for_point({"code": None})
    target = validate_target_output("self", categorical_space)
    categorical_space.sample_store.add_measurement_results(
        [
            ValidMeasurementResult(
                entityIdentifier=candidate.identifier,
                measurements=[ObservedPropertyValue(property=target, value="12")],
            )
        ],
        skip_relationship_to_request=True,
    )
    assert not measured_labels(categorical_space, target, [candidate])


def test_measured_labels_span_sqlite_bind_limit(
    categorical_space: DiscoverySpace,
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> None:
    """Read labels at both ends of a large grid and exclude stored points outside it."""
    prop = ConstitutiveProperty(
        identifier="code",
        propertyDomain=PropertyDomain(values=["1", *range(10, 1010), b"code"]),
    )
    space = make_pkh_space([prop])
    target = validate_target_output("self", categorical_space)
    with space.sample_store.engine.connect() as connection:
        connection.connection.driver_connection.setlimit(
            sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999
        )
    point = space.entitySpace.entity_for_point
    assert set(measured_labels(space, target)) == {
        point({"code": "1"}).identifier,
        point({"code": b"code"}).identifier,
    }


def test_forest_input_uses_sparse_storage_when_possible(
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> None:
    """Keep one-hot features for a categorical grid sparse."""
    prop = ConstitutiveProperty(
        identifier="category", propertyDomain=PropertyDomain(values=list("abcdefghij"))
    )
    space = make_pkh_space([prop])
    frame = feature_frame(space, candidate_coordinates(candidate_entities(space)))
    assert issparse(encode_feature_matrix(frame))


def test_large_integer_coordinates_are_numeric_model_inputs(
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> None:
    """Expose a large-integer grid as a numeric feature and retain exact coordinates."""
    values = [2**65 + 1, 2**65 + 2**52 + 1]
    prop = ConstitutiveProperty(
        identifier="reading", propertyDomain=PropertyDomain(values=values)
    )
    space = make_pkh_space([prop])
    coordinates = candidate_coordinates(candidate_entities(space))
    frame = feature_frame(space, coordinates)

    assert frame.select_dtypes(include="number").columns.tolist() == ["reading"]
    assert frame["reading"].nunique() == 2
    assert coordinates["reading"] == values
