# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
from collections.abc import Callable

import numpy.random
import pytest

from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from orchestrator.schema.property import AbstractPropertyDescriptor
from orchestrator.schema.property_value import ConstitutivePropertyValue
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.result import (
    InvalidMeasurementResult,
    MeasurementResult,
    MeasurementResultStateEnum,
    ValidMeasurementResult,
)
from orchestrator.schema.virtual_property import (
    PropertyAggregationMethod,
    VirtualObservedProperty,
)


def test_valid_measurement_result(
    entity: Entity,
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
) -> None:

    # Test init
    result = ValidMeasurementResult(
        entityIdentifier=entity.identifier, measurements=property_values
    )

    # Test reference
    assert property_values[0].property.experimentReference == result.experimentReference


def test_valid_measurement_result_mismatch_properties(
    entity: Entity,
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
) -> None:

    import pydantic

    # Create a fake property-value
    ap = AbstractPropertyDescriptor(identifier="test_prop")
    op = ObservedProperty(
        targetProperty=ap,
        experimentReference=ExperimentReference(
            experimentIdentifier="test_exp", actuatorIdentifier="test_act"
        ),
    )
    pv = ObservedPropertyValue(
        value=numpy.random.default_rng().integers(0, 50), property=op
    )

    # Test init with incorrect properties
    property_values.append(pv)

    with pytest.raises(pydantic.ValidationError):
        # Test init
        ValidMeasurementResult(
            entityIdentifier=entity.identifier, measurements=property_values
        )


def test_valid_measurement_result_no_properties(entity: Entity) -> None:

    import pydantic

    with pytest.raises(pydantic.ValidationError):
        # Test init
        ValidMeasurementResult(entityIdentifier=entity.identifier, measurements=[])


def test_invalid_measurement_record(entity: Entity) -> None:

    # Test init
    InvalidMeasurementResult(
        entityIdentifier=entity.identifier,
        reason="Insufficient memory",
        experimentReference=ExperimentReference(
            experimentIdentifier="testexp", actuatorIdentifier="testact"
        ),
    )


@pytest.fixture
def valid_measurement_result(
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
    entity: Entity,
) -> ValidMeasurementResult:

    return ValidMeasurementResult(
        entityIdentifier=entity.identifier, measurements=property_values
    )


@pytest.fixture
def invalid_measurement_result(
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
    entity: Entity,
) -> InvalidMeasurementResult:

    return InvalidMeasurementResult(
        entityIdentifier=entity.identifier,
        reason="Insufficient memory",
        experimentReference=ExperimentReference(
            experimentIdentifier="testexp", actuatorIdentifier="testact"
        ),
    )


def test_valid_measurement_result_series_representation(
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
    ml_multi_cloud_benchmark_performance_experiment: Experiment,
) -> None:

    number_entities = 1
    measurements_per_result = 1

    random_result: ValidMeasurementResult = (
        random_ml_multi_cloud_benchmark_performance_measurement_results(
            measurements_per_result=measurements_per_result,
            entity=random_ml_multi_cloud_benchmark_performance_entities(
                quantity=number_entities
            )[0],
            status=MeasurementResultStateEnum.VALID,
        )
    )

    #
    target_series_representation = random_result.series_representation(
        output_format="target"
    )
    observed_series_representation = random_result.series_representation(
        output_format="observed"
    )

    #
    expected_entity_identifier = random_result.entityIdentifier
    expected_experiment_identifier = random_result.experimentReference
    expected_validity = True
    expected_observed_property_identifier = random_result.measurements[
        0
    ].property.identifier
    expected_target_property_identifier = random_result.measurements[
        0
    ].property.targetProperty.identifier
    expected_property_value = random_result.measurements[0].value

    #
    for representation in [
        target_series_representation,
        observed_series_representation,
    ]:
        assert representation["identifier"] == expected_entity_identifier
        assert representation["experiment_id"] == str(expected_experiment_identifier)
        assert representation["valid"] == expected_validity

    assert expected_target_property_identifier in target_series_representation
    assert (
        target_series_representation[expected_target_property_identifier]
        == expected_property_value
    )
    assert expected_observed_property_identifier in observed_series_representation
    assert (
        observed_series_representation[expected_observed_property_identifier]
        == expected_property_value
    )

    # Test with a virtual observed property
    vp = VirtualObservedProperty(
        baseObservedProperty=ml_multi_cloud_benchmark_performance_experiment.observedProperties[
            0
        ],
        aggregationMethod=PropertyAggregationMethod(),  # defaults to mean
    )
    # Test target format
    rep = random_result.series_representation(
        output_format="target",
        virtual_target_property_identifiers=[vp.virtualTargetPropertyIdentifier],
    )
    assert rep.get(vp.virtualTargetPropertyIdentifier)

    # Test observed format
    rep = random_result.series_representation(
        output_format="observed",
        virtual_target_property_identifiers=[vp.virtualTargetPropertyIdentifier],
    )
    assert rep.get(vp.identifier)

    # Test behaviour with random string virtual target property identifier - should raise ValueError
    with pytest.raises(
        ValueError, match="random_string is not a valid virtual property identifier"
    ):
        random_result.series_representation(
            output_format="observed",
            virtual_target_property_identifiers=["random_string"],
        )

    # Test if vp identifier but doesn't match anything - nothing returned
    rep = random_result.series_representation(
        output_format="observed",
        virtual_target_property_identifiers=["some_prop-mean"],
    )
    assert rep.get("some_prop-mean") is None


def test_measurement_results_series_representation_invalid_method(
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
) -> None:

    random_result: ValidMeasurementResult = (
        random_ml_multi_cloud_benchmark_performance_measurement_results(
            measurements_per_result=3,
            entity=random_ml_multi_cloud_benchmark_performance_entities(quantity=1)[0],
            status=MeasurementResultStateEnum.VALID,
        )
    )

    with pytest.raises(
        ValueError, match="The only supported series representation output formats are"
    ):
        # Test passing an invalid value raises ValueError
        random_result.series_representation(output_format="random")


def test_invalid_measurement_result_series_representation(
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
) -> None:

    number_entities = 1
    measurements_per_result = 1

    random_result: InvalidMeasurementResult = (
        random_ml_multi_cloud_benchmark_performance_measurement_results(
            measurements_per_result=measurements_per_result,
            entity=random_ml_multi_cloud_benchmark_performance_entities(
                quantity=number_entities
            )[0],
            status=MeasurementResultStateEnum.INVALID,
        )
    )

    #
    target_series_representation = random_result.series_representation(
        output_format="target"
    )
    observed_series_representation = random_result.series_representation(
        output_format="observed"
    )

    #
    expected_entity_identifier = random_result.entityIdentifier
    expected_experiment_identifier = random_result.experimentReference
    expected_validity = False
    expected_reason = random_result.reason

    #
    for representation in [
        target_series_representation,
        observed_series_representation,
    ]:
        assert representation["identifier"] == expected_entity_identifier
        assert representation["experiment_id"] == str(expected_experiment_identifier)
        assert representation["valid"] == expected_validity
        assert representation["reason"] == expected_reason


def test_compressed_serialization_format(
    entity: Entity,
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
) -> None:
    """Verify new serialization format has experimentReference at top level"""
    result = ValidMeasurementResult(
        entityIdentifier=entity.identifier, measurements=property_values
    )
    serialized = result.model_dump()

    # Verify new format structure
    assert "experimentReference" in serialized
    assert "measurements" in serialized
    assert "uid" in serialized
    assert "entityIdentifier" in serialized

    # Verify measurements have 'property' field but without experimentReference
    assert "property" in serialized["measurements"][0]
    assert "experimentReference" not in serialized["measurements"][0]["property"]

    # Verify property contains targetProperty and metadata
    assert "targetProperty" in serialized["measurements"][0]["property"]
    assert "metadata" in serialized["measurements"][0]["property"]
    assert "value" in serialized["measurements"][0]
    assert "valueType" in serialized["measurements"][0]


def test_old_format_deserialization(entity: Entity) -> None:
    """Verify old format can still be deserialized"""
    from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum

    # Create old format JSON (with experimentReference in each measurement)
    old_format_json = {
        "uid": "12345678-1234-1234-1234-123456789012",
        "entityIdentifier": entity.identifier,
        "metadata": {},
        "measurements": [
            {
                "property": {
                    "experimentReference": {
                        "actuatorIdentifier": "test_actuator",
                        "experimentIdentifier": "test_experiment",
                        "parameterization": None,
                    },
                    "targetProperty": {
                        "identifier": "test_property",
                        "propertyType": "MEASURED_PROPERTY_TYPE",
                        "propertyDomain": PropertyDomain(
                            values=["a", "b"],
                            variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
                        ).model_dump(),
                    },
                    "metadata": {},
                },
                "value": 42.0,
                "valueType": "NUMERIC_VALUE_TYPE",
                "uncertainty": None,
            }
        ],
    }

    # Should deserialize successfully
    result = ValidMeasurementResult.model_validate(old_format_json)
    assert result.uid == old_format_json["uid"]
    assert result.entityIdentifier == old_format_json["entityIdentifier"]
    assert len(result.measurements) == 1
    assert result.measurements[0].value == 42.0


def test_new_format_deserialization(entity: Entity) -> None:
    """Verify new format deserializes correctly"""
    from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum

    # Create new format JSON (experimentReference at top level)
    new_format_json = {
        "uid": "12345678-1234-1234-1234-123456789012",
        "entityIdentifier": entity.identifier,
        "metadata": {},
        "experimentReference": {
            "actuatorIdentifier": "test_actuator",
            "experimentIdentifier": "test_experiment",
            "parameterization": None,
        },
        "measurements": [
            {
                "property": {
                    "targetProperty": {
                        "identifier": "test_property",
                        "propertyType": "MEASURED_PROPERTY_TYPE",
                        "propertyDomain": PropertyDomain(
                            values=["a", "b"],
                            variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
                        ).model_dump(),
                    },
                    "metadata": {},
                },
                "value": 42.0,
                "valueType": "NUMERIC_VALUE_TYPE",
                "uncertainty": None,
            }
        ],
    }

    # Store expected values before validation (validator modifies the dict)
    expected_actuator = new_format_json["experimentReference"]["actuatorIdentifier"]
    expected_experiment = new_format_json["experimentReference"]["experimentIdentifier"]

    # Should deserialize successfully
    result = ValidMeasurementResult.model_validate(new_format_json)
    assert result.uid == "12345678-1234-1234-1234-123456789012"
    assert result.entityIdentifier == entity.identifier
    assert len(result.measurements) == 1
    assert result.measurements[0].value == 42.0
    assert result.experimentReference.actuatorIdentifier == expected_actuator
    assert result.experimentReference.experimentIdentifier == expected_experiment


def test_serialization_deserialization_roundtrip(
    entity: Entity,
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
) -> None:
    """Verify round-trip: create → serialize → deserialize → verify"""
    original = ValidMeasurementResult(
        entityIdentifier=entity.identifier, measurements=property_values
    )

    # Serialize
    serialized = original.model_dump()

    # Deserialize
    deserialized = ValidMeasurementResult.model_validate(serialized)

    # Verify all fields match
    assert original.uid == deserialized.uid
    assert original.entityIdentifier == deserialized.entityIdentifier
    assert len(original.measurements) == len(deserialized.measurements)
    assert original.experimentReference == deserialized.experimentReference

    # Verify measurement values
    for orig_m, deser_m in zip(
        original.measurements, deserialized.measurements, strict=True
    ):
        assert orig_m.value == deser_m.value
        assert orig_m.valueType == deser_m.valueType
        assert (
            orig_m.property.targetProperty.identifier
            == deser_m.property.targetProperty.identifier
        )


def test_compression_achieved(
    entity: Entity,
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
) -> None:
    """Verify compression is achieved with new format"""
    import json

    # Create result with multiple measurements
    result = ValidMeasurementResult(
        entityIdentifier=entity.identifier, measurements=property_values
    )

    # Simulate old format size (each measurement includes full ExperimentReference)
    old_format_measurements = [
        {
            "property": m.property.model_dump(),
            "value": m.value,
            "valueType": m.valueType,
            "uncertainty": m.uncertainty,
        }
        for m in result.measurements
    ]
    old_format_data = {
        "uid": result.uid,
        "entityIdentifier": result.entityIdentifier,
        "metadata": result.metadata,
        "measurements": old_format_measurements,
    }
    old_size = len(json.dumps(old_format_data))

    # New format size (ExperimentReference stored once)
    new_size = len(result.model_dump_json())

    # Verify new format is smaller (only if we have multiple measurements)
    if len(result.measurements) > 1:
        assert new_size < old_size, "New format should be smaller than old format"

        # Compression increases with more measurements
        compression_ratio = (old_size - new_size) / old_size
        assert compression_ratio > 0, "Should achieve some compression"


def test_json_serialization_roundtrip(
    entity: Entity,
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
) -> None:
    """Verify JSON serialization and deserialization works correctly"""
    import json

    original = ValidMeasurementResult(
        entityIdentifier=entity.identifier, measurements=property_values
    )

    # Serialize to JSON string
    json_str = original.model_dump_json()

    # Parse JSON
    json_data = json.loads(json_str)

    # Verify it's in new format
    assert "experimentReference" in json_data
    assert "measurements" in json_data
    assert "property" in json_data["measurements"][0]
    assert "experimentReference" not in json_data["measurements"][0]["property"]

    # Deserialize from JSON string
    deserialized = ValidMeasurementResult.model_validate_json(json_str)

    # Verify match
    assert original.uid == deserialized.uid
    assert original.entityIdentifier == deserialized.entityIdentifier
    assert len(original.measurements) == len(deserialized.measurements)
