# Copyright (c) IBM Corporation
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
        assert representation["experiment_id"] == expected_experiment_identifier
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
        assert representation["experiment_id"] == expected_experiment_identifier
        assert representation["valid"] == expected_validity
        assert representation["reason"] == expected_reason
