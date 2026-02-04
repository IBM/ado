# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import re

import numpy as np
import pytest

from orchestrator.core.samplestore.csv import CSVSampleStore
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.entity import (
    CheckRequiredConstitutivePropertyValuesPresent,
    Entity,
)
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from orchestrator.schema.property import (
    AbstractPropertyDescriptor,
    ConcretePropertyDescriptor,
    ConstitutivePropertyDescriptor,
    MeasuredPropertyTypeEnum,
    NonMeasuredPropertyTypeEnum,
)
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.result import ValidMeasurementResult
from orchestrator.schema.virtual_property import (
    PropertyAggregationMethod,
    PropertyAggregationMethodEnum,
    VirtualObservedProperty,
)


def test_value_per_property(
    entity: Entity, abstract_properties: list[AbstractPropertyDescriptor]
) -> None:
    for ap in abstract_properties:
        value = entity.valuesForTargetProperty(ap)
        assert value is not None


def test_retrieve_properties_by_experiment(
    entity: Entity, experiment: Experiment
) -> None:
    assert experiment.reference in entity.experimentReferences

    pvs = entity.propertyValuesFromExperiment(experiment)
    assert len(pvs) != 0

    ops = [p.property for p in pvs]

    assert len(ops) == len(experiment.observedProperties)


def test_retrieve_properties_by_experiment_reference(
    entity: Entity, experiment_reference: ExperimentReference, experiment: Experiment
) -> None:
    assert experiment_reference in entity.experimentReferences

    pvs = entity.propertyValuesFromExperimentReference(experiment_reference)
    assert len(pvs) != 0

    ops = [p.property for p in pvs]

    assert len(ops) == len(experiment.observedProperties)


def test_property_types(entity: Entity) -> None:
    """Test that created properties have the correct type as per the enum"""

    assert len(entity.propertyValues) != 0

    for p in entity.observedProperties:
        assert p.propertyType == MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE

    for p in entity.constitutiveProperties:
        assert p.propertyType == NonMeasuredPropertyTypeEnum.CONSTITUTIVE_PROPERTY_TYPE


def test_number_and_type_of_property_method_return_values(entity: Entity) -> None:
    """Test that the property methods return what's expected in type and number"""

    # Add extra duplicate measurement result to entity to test it can handle it
    result = ValidMeasurementResult(
        entityIdentifier=entity.identifier,
        measurements=entity.measurement_results[0].measurements,
    )
    entity.add_measurement_result(result)

    assert len(entity.observedProperties) > 0, "Expected at least one observed property"
    assert (
        len(entity.constitutiveProperties) > 0
    ), "Expected at least one constitutive property"

    assert len(entity.propertyValues) > 1, "Expected at least two property values"

    # Length of constitutivePropertyValues is the length of constitutiveProperties
    assert len(entity.constitutiveProperties) == len(
        entity.constitutive_property_values
    ), "There must be ONE value for each constitutive property"

    # All observed properties have correct type

    assert {type(cp) for cp in entity.observedProperties} == {ObservedProperty}

    # All the observed properties are unique
    assert len({cp.identifier for cp in entity.observedProperties}) == len(
        entity.observedProperties
    ), "Error: Duplicate observed properties"

    # All constitutive properties have correct type
    assert {type(cp) for cp in entity.constitutiveProperties} == {
        ConstitutivePropertyDescriptor
    }
    assert len({cp.identifier for cp in entity.constitutiveProperties}) == len(
        entity.observedProperties
    ), "Error: Duplicate observed properties"

    # Properties is the sum of observed properties and constitutive properties
    assert len(entity.properties) == len(entity.observedProperties) + len(
        entity.constitutiveProperties
    ), "The total number of properties should be the sum of the number of observed and constitutive properties"

    # PropertyValues is the sum of observedPropertyValues and constitutivePropertyValues
    assert len(entity.properties) == len(entity.observedProperties) + len(
        entity.constitutiveProperties
    ), "The total number of properties should be the sum of the number of observed and constitutive properties"

    for p in entity.observedProperties:
        assert (
            p.propertyType == MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE
        ), "All observed properties should be type ObservedProperty"

    for p in entity.constitutiveProperties:
        assert (
            p.propertyType == NonMeasuredPropertyTypeEnum.CONSTITUTIVE_PROPERTY_TYPE
        ), "All constitutive properties should be type ConstitutiveProperty"


def test_multiple_values_per_observed_property(
    entity: Entity,
    experiment: Experiment,
    property_values: [],
) -> None:
    # Add a second set of values for each observed property to the entity

    numberValues = len(entity.observedPropertyValues)
    for v in property_values:
        entity.add_measurement_result(
            ValidMeasurementResult(entityIdentifier=entity.identifier, measurements=[v])
        )

    assert len(entity.observedPropertyValues) == 2 * numberValues
    assert len(entity.propertyValuesFromExperiment(experiment)) == 2 * len(
        experiment.observedProperties
    )

    testProperty = experiment.observedProperties[0]

    assert isinstance(
        entity.valueForProperty(testProperty),
        ObservedPropertyValue,
    )

    assert len(entity.valuesForTargetProperty(testProperty.targetProperty)) == 2

    assert len(entity.valuesForProperty(testProperty)) > 1


def test_entity_series_representation(entity: Entity) -> None:

    rep = entity.seriesRepresentation()
    assert rep.get("identifier") is not None
    assert rep.get("generatorid") is not None
    for prop in entity.properties:
        if entity.valuesForProperty(prop):
            assert rep.get(prop.identifier) is not None


def test_virtual_property_request(
    entity: Entity, abstract_properties: list[AbstractPropertyDescriptor]
) -> None:
    for obs in entity.observedProperties:
        for e in PropertyAggregationMethodEnum:
            vps = entity.virtualObservedPropertiesFromIdentifier(
                f"{obs.identifier}-{e.value}"
            )
            assert len(vps) == 1
            assert vps[0].identifier == f"{obs.identifier}-{e.value}"

    for ap in abstract_properties:
        for e in PropertyAggregationMethodEnum:
            vps = entity.virtualObservedPropertiesFromIdentifier(
                f"{ap.identifier}-{e.value}"
            )
            assert vps[0].identifier != f"{ap.identifier}-{e.value}"
            assert vps[0].baseObservedProperty.targetProperty == ap

            # we know all the properties have just one value
            value = entity.valueForProperty(vps[0].baseObservedProperty)
            if e in [
                PropertyAggregationMethodEnum.mean,
                PropertyAggregationMethodEnum.median,
                PropertyAggregationMethodEnum.min,
                PropertyAggregationMethodEnum.max,
            ]:
                assert vps[0].aggregate([value.value]).value == value.value

    # Check the actual aggregation works
    # Add multiple values to one of the properties
    x = [2, 4, 6, 7]
    prop = entity.observedProperties[0]
    for a in x:
        v = ObservedPropertyValue(value=a, property=prop)
        entity.add_measurement_result(
            ValidMeasurementResult(entityIdentifier=entity.identifier, measurements=[v])
        )

    values = [v.value for v in entity.valuesForProperty(prop)]

    # Create the virtual property - the mean
    aggregation_method = PropertyAggregationMethod(
        identifier=PropertyAggregationMethodEnum("mean")
    )
    virtualProperty = VirtualObservedProperty(
        baseObservedProperty=prop, aggregationMethod=aggregation_method
    )

    print(entity.valueForProperty(virtualProperty))
    # Check the virtual property value is correct
    assert entity.valueForProperty(virtualProperty).value == np.asarray(values).mean()


def test_virtual_property_request_no_values(
    entity: Entity,
    abstract_properties: list[AbstractPropertyDescriptor],
    experiment: Experiment,
) -> None:
    # Create the virtual property - the mean
    aggregation_method = PropertyAggregationMethod(
        identifier=PropertyAggregationMethodEnum("mean")
    )
    virtualProperty = VirtualObservedProperty(
        baseObservedProperty=ObservedProperty(
            targetProperty=ConcretePropertyDescriptor(identifier="nonexistent"),
            experimentReference=experiment.reference,
        ),
        aggregationMethod=aggregation_method,
    )

    assert entity.valueForProperty(virtualProperty) is None


def test_virtual_property_request_invalid_identifier(entity: Entity) -> None:

    # Check that if the identifier is not a virtual property id a ValueError is raises
    with pytest.raises(
        ValueError, match="random_string is not a valid virtual property identifier"
    ):
        entity.virtualObservedPropertiesFromIdentifier("random_string")

    # Check that if the identifier is valid but not of an observed property an empty list is returned
    assert (
        entity.virtualObservedPropertiesFromIdentifier("random_property_name-mean")
        is None
    )


def test_entity_rich_print(entity: Entity) -> None:
    from rich.console import Console

    Console().print(entity)


def test_entity_to_dict(
    csv_sample_store: CSVSampleStore,
) -> None:

    e = csv_sample_store.entities[0]
    # Ensure the entity has some properties and values
    assert len(e.properties) != 0
    assert len(e.propertyValues) != 0
    csv_sample_store.entities[0].model_dump(exclude_defaults=True, exclude_unset=True)


def test_entity_to_json(
    csv_sample_store: CSVSampleStore,
) -> None:

    e = csv_sample_store.entities[0]
    assert len(e.properties) != 0
    assert len(e.propertyValues) != 0
    csv_sample_store.entities[0].model_dump_json(
        exclude_defaults=True, exclude_unset=True
    )


def test_identifier_from_property_values(
    entity_for_parameterized_experiment: tuple[Entity, Experiment],
) -> None:

    from rich.console import Console

    test_entity, _test_experiment = entity_for_parameterized_experiment

    constitutive_property_values = test_entity.constitutive_property_values

    Console().print(test_entity)

    assert (
        ident := Entity.identifier_from_property_values(constitutive_property_values)
    ), "Expected identifier_from_property_values to return an identifier given a set of constitutive property values"
    assert ident == "-".join(
        [f"{pv.property.identifier}.{pv.value}" for pv in constitutive_property_values]
    ), "Expected the ident to have a certain format: $PROP1_ID.$PROP1_VALUE-$PROP2_ID.$PROP2_VALUE ..."

    constitutive_property_values = [
        *list(constitutive_property_values),
        ObservedPropertyValue(
            value=3,
            property=ObservedProperty(
                targetProperty=AbstractPropertyDescriptor(identifier="test"),
                experimentReference=ExperimentReference(
                    experimentIdentifier="test", actuatorIdentifier="test"
                ),
            ),
        ),
    ]
    with pytest.raises(
        ValueError, match="All values must be for ConstitutiveProperties"
    ) as expected_exception:
        Entity.identifier_from_property_values(constitutive_property_values)

    assert (
        expected_exception is not None
    ), "Expected property with identifier to fail if a non constitutive property was passed"


def test_value_error_duplicate_constitutive_properties(
    entity_for_parameterized_experiment: tuple[Entity, Experiment],
) -> None:
    test_entity, _test_experiment = entity_for_parameterized_experiment
    constitutive_property_values = test_entity.constitutive_property_values

    assert (
        ident := Entity.identifier_from_property_values(constitutive_property_values)
    ), "Expected identifier_from_property_values to return an identifier given a set of constitutive property values"
    assert ident == "-".join(
        [f"{pv.property.identifier}.{pv.value}" for pv in constitutive_property_values]
    ), "Expected the ident to have a certain format: $PROP1_ID.$PROP1_VALUE-$PROP2_ID.$PROP2_VALUE ..."

    with pytest.raises(ValueError, match="Constitutive properties must be unique"):
        Entity(
            constitutive_property_values=tuple(
                list(constitutive_property_values)
                + list(constitutive_property_values[:1])
            )
        )


def test_value_error_duplicate_measurement_results(
    valid_measurement_result_and_entity: tuple[Entity, ValidMeasurementResult],
) -> None:

    test_entity, result = valid_measurement_result_and_entity
    # On init and via add_measurement_result
    test_entity.add_measurement_result(result)

    with pytest.raises(
        ValueError,
        match=f"Entity {re.escape(str(test_entity.identifier))} already contained "
        f"a MeasurementResult with id {re.escape(str(result.uid))}",
    ) as expected_exception:
        test_entity.add_measurement_result(result)

    assert (
        expected_exception is not None
    ), "Expected adding duplicate result would raise ValueError"

    d = test_entity.model_dump()
    d.pop("measurement_results")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "There were 1 duplicate MeasurementResults (they had the same UID)"
        ),
    ):
        Entity(measurement_results=[result, result], **d)


def test_observed_properties_from_experiment_reference(
    valid_measurement_result_and_entity: tuple[Entity, ValidMeasurementResult],
    global_registry: ActuatorRegistry,
) -> None:

    test_entity, result = valid_measurement_result_and_entity

    ref = result.experimentReference
    assert not test_entity.observedPropertiesFromExperimentReference(
        ref
    ), "Expected the entity fixture to have no results added"
    test_entity.add_measurement_result(result)
    exp = global_registry.experimentForReference(ref)

    assert (
        ops := test_entity.observedPropertiesFromExperimentReference(ref)
    ), "Expected the entity to return properties once results added"
    assert set(ops) == set(
        ParameterizedExperiment(
            parameterization=ref.parameterization, **exp.model_dump()
        ).observedProperties
    ), "Expected the observed properties to have the names of the properties of parameterized experiment the measurements came from"

    assert (
        results := test_entity.measurement_results_for_experiment_reference(ref)
    ), "Expected a result to be returned for the experiment added"
    assert len(results) == 1, "Expected there to be only 1 result as only 1 was added"
    assert (
        results[0] == result
    ), "Expected the returned result to be identical to the added result"


def test_series_representation_with_observed_property_values(
    valid_measurement_result_and_entity: tuple[Entity, ValidMeasurementResult],
    global_registry: ActuatorRegistry,
) -> None:

    test_entity, result = valid_measurement_result_and_entity
    ref = result.experimentReference
    assert not test_entity.observedPropertiesFromExperimentReference(
        ref
    ), "Expected the entity fixture to have no results added"

    test_entity.add_measurement_result(result)

    # Test plain series_representation
    assert (
        ser := test_entity.seriesRepresentation()
    ) is not None, "Expected a series representation to be returned"
    for cp in test_entity.constitutive_property_values:
        assert (
            ser[cp.property.identifier] == cp.value
        ), f"Expected the series representation to have a key:value for {cp}"

    for ov in test_entity.observedPropertyValues:
        assert (
            ser[ov.property.identifier] == ov.value
        ), f"Expected the series representation to have a key:value for {ov}"

    # test constitutiveOnly
    assert (
        ser := test_entity.seriesRepresentation(constitutiveOnly=True)
    ) is not None, "Expected a series representation to be returned"
    for cp in test_entity.constitutive_property_values:
        assert (
            ser[cp.property.identifier] == cp.value
        ), f"Expected the series representation to have a key:value for {cp}"

    for ov in test_entity.observedPropertyValues:
        assert not ser.get(
            ov.property.identifier
        ), f"Expected the series representation to have a key:value for {ov}"

    # Test series_representation with specific references
    assert (
        ser := test_entity.seriesRepresentation(experimentReferences=[ref])
    ) is not None, "Expected a series representation to be returned"
    for cp in test_entity.constitutive_property_values:
        assert (
            ser[cp.property.identifier] == cp.value
        ), f"Expected the series representation to have a key:value for {cp}"

    ovs = test_entity.propertyValuesFromExperimentReference(ref)
    assert len(ovs) > 0
    for ov in ovs:
        assert (
            ser[ov.property.identifier] == ov.value
        ), f"Expected the series representation to have a key:value for {ov}"

    # Test series_representation with virtual_properties
    vp = VirtualObservedProperty(
        baseObservedProperty=test_entity.observedProperties[0],
        aggregationMethod=PropertyAggregationMethod(
            identifier=PropertyAggregationMethodEnum.max
        ),
    )
    assert (
        ser := test_entity.seriesRepresentation(
            virtualTargetPropertyIdentifiers=[vp.identifier],
        )
    ) is not None, "Expected a series representation to be returned"
    for cp in test_entity.constitutive_property_values:
        assert (
            ser[cp.property.identifier] == cp.value
        ), f"Expected the series representation to have a key:value for {cp}"

    for ov in test_entity.observedPropertyValues:
        assert (
            ser[ov.property.identifier] == ov.value
        ), f"Expected the series representation to have a key:value for {ov}"

    assert ser[
        vp.identifier
    ], f"Expected the series representation to have a key:value for {vp}"


def test_series_representation_multiple_observed(
    valid_measurement_result_and_entity: tuple[Entity, ValidMeasurementResult],
    global_registry: ActuatorRegistry,
) -> None:

    test_entity, result = valid_measurement_result_and_entity
    test_entity.add_measurement_result(result)

    # Test multiple results for a property
    # Add another result for same experiment
    values = [
        ObservedPropertyValue(value=np.random.default_rng().random(), property=op)
        for op in test_entity.observedPropertiesFromExperimentReference(
            result.experimentReference
        )
    ]

    second_result = ValidMeasurementResult(
        entityIdentifier=test_entity.identifier,
        measurements=values,
    )

    test_entity.add_measurement_result(second_result)

    # Test if we don't aggregate we get a list for each property
    assert (
        ser := test_entity.seriesRepresentation(
            experimentReferences=[result.experimentReference]
        )
    ) is not None, "Expected a series representation to be returned"
    for cp in test_entity.constitutive_property_values:
        assert (
            ser[cp.property.identifier] == cp.value
        ), f"Expected the series representation to have a key:value for {cp}"

    ops = test_entity.observedPropertiesFromExperimentReference(
        result.experimentReference
    )
    assert len(ops) > 0
    for op in ops:
        assert (
            len(ser[op.identifier]) == 2
        ), f"Expected the series representation to have two values for {op}"

    # Test if we do aggregate we get a single virtual property
    assert (
        ser := test_entity.seriesRepresentation(
            experimentReferences=[result.experimentReference],
            aggregationMethod=PropertyAggregationMethodEnum.mean,
        )
    ) is not None, "Expected a series representation to be returned"
    for cp in test_entity.constitutive_property_values:
        assert (
            ser[cp.property.identifier] == cp.value
        ), f"Expected the series representation to have a key:value for {cp}"

    ops = test_entity.observedPropertiesFromExperimentReference(
        result.experimentReference
    )
    assert len(ops) > 0
    for op in ops:
        assert not (
            ser.get(op.identifier)
        ), f"Expected the series representation with aggregation to have no values for {op}"
        vp = VirtualObservedProperty(
            baseObservedProperty=op,
            aggregationMethod=PropertyAggregationMethod(
                identifier=PropertyAggregationMethodEnum.mean
            ),
        )
        assert (
            ser.get(vp.identifier) is not None
        ), f"Expected the series representation with aggregation to have 1 value for {vp}"


def test_experiment_series(
    valid_measurement_result_and_entity: tuple[Entity, ValidMeasurementResult],
    global_registry: ActuatorRegistry,
) -> None:
    test_entity, result = valid_measurement_result_and_entity
    ref = result.experimentReference
    assert not test_entity.observedPropertiesFromExperimentReference(
        ref
    ), f"Expected the entity fixture to have no results added for {ref}"

    test_entity.add_measurement_result(result)

    # Test plain experiment_series
    ser = test_entity.experimentSeries()
    assert ser, "Expected a list of experiment series' to be returned"

    assert len(ser) == len(
        test_entity.experimentReferences
    ), "Expected 1 result for each experiment appleid to entity"
    for s in ser:
        for cp in test_entity.constitutive_property_values:
            assert (
                s[cp.property.identifier] == cp.value
            ), f"Expected the experiment series to contain a key:value for {cp}"

        for ov in test_entity.propertyValuesFromExperimentReference(s["experiment_id"]):
            assert (
                s[ov.property.targetProperty.identifier] == ov.value
            ), f"Expected the experiment series for {s['experiment_id']} to contain a key:value for {ov}"

    # Test  experiment_series if a reference does not exist
    ser = test_entity.experimentSeries(
        experimentReferences=[
            ExperimentReference(actuatorIdentifier="test", experimentIdentifier="mock")
        ]
    )
    assert (
        not ser
    ), "Expected that if a series for an experiment which did not have values in the Entity was requested no series would be returned"

    # Test  experiment_series with specific references
    ser = test_entity.experimentSeries(experimentReferences=[ref])
    assert ser, "Expected a list of experiment series' to be returned"
    assert len(ser) == 1, f"Expected there to be only 1 measurement for {ref}"
    ser = ser[0]
    for cp in test_entity.constitutive_property_values:
        assert (
            ser[cp.property.identifier] == cp.value
        ), f"Expected the experiment series to contain a key:value for {cp}"

    for ov in test_entity.propertyValuesFromExperimentReference(ref):
        assert (
            ser[ov.property.targetProperty.identifier] == ov.value
        ), f"Expected the experiment series for {ref} to contain a key:value for {ov}"

    assert (
        ser.get("experiment_id") == ref
    ), f"Expected the value of the 'experiment_id' key to be to {ref}"

    # Test  experiment_series with virtual_properties
    vp = VirtualObservedProperty(
        baseObservedProperty=test_entity.observedProperties[0],
        aggregationMethod=PropertyAggregationMethod(
            identifier=PropertyAggregationMethodEnum.max
        ),
    )

    ser = test_entity.experimentSeries(
        virtualTargetPropertyIdentifiers=[vp.identifier],
    )
    assert ser, "Expected a list of experiment series' to be returned"

    for s in ser:
        for cp in test_entity.constitutive_property_values:
            assert (
                s[cp.property.identifier] == cp.value
            ), f"Expected the experiment series to contain a key:value for {cp}"

        for ov in test_entity.propertyValuesFromExperimentReference(s["experiment_id"]):
            assert (
                s[ov.property.targetProperty.identifier] == ov.value
            ), f"Expected the experiment series for {s['experiment_id']} to contain a key:value for {ov}"

        if s["experiment_id"] == test_entity.observedProperties[0].experimentReference:
            assert s[
                vp.virtualTargetPropertyIdentifier
            ], f"Expected the experiment series for {test_entity.observedProperties[0].experimentReference} to contain a key:value for {vp}"


def test_experiment_series_multiple_observed(
    valid_measurement_result_and_entity: tuple[Entity, ValidMeasurementResult],
    global_registry: ActuatorRegistry,
) -> None:

    test_entity, result = valid_measurement_result_and_entity
    test_entity.add_measurement_result(result)

    # Test multiple results for a property
    # Add another result for same experiment
    values = [
        ObservedPropertyValue(value=np.random.default_rng().random(), property=op)
        for op in test_entity.observedPropertiesFromExperimentReference(
            result.experimentReference
        )
    ]

    second_result = ValidMeasurementResult(
        entityIdentifier=test_entity.identifier,
        measurements=values,
    )

    test_entity.add_measurement_result(second_result)

    # Test if we don't aggregate we get a list for each property
    ser = test_entity.experimentSeries(
        experimentReferences=[result.experimentReference]
    )
    assert ser is not None, "Expected a experiment series to be returned"
    assert (
        len(ser) == 1
    ), f"Expected there to be only 1 entry for {result.experimentReference}"
    ser = ser[0]
    for cp in test_entity.constitutive_property_values:
        assert (
            ser[cp.property.identifier] == cp.value
        ), f"Expected the series representation to have a key:value for {cp}"

    ops = test_entity.observedPropertiesFromExperimentReference(
        result.experimentReference
    )
    assert len(ops) > 0
    for op in ops:
        assert (
            len(ser[op.targetProperty.identifier]) == 2
        ), f"Expected the series representation to have two values for {op}"

    # Test if we do aggregate we get a single virtual property
    ser = test_entity.experimentSeries(
        experimentReferences=[result.experimentReference],
        aggregationMethod=PropertyAggregationMethodEnum.mean,
    )
    assert ser is not None, "Expected a series representation to be returned"
    assert (
        len(ser) == 1
    ), f"Expected there to be only 1 entry for {result.experimentReference}"
    ser = ser[0]
    for cp in test_entity.constitutive_property_values:
        assert (
            ser[cp.property.identifier] == cp.value
        ), f"Expected the series representation to have a key:value for {cp}"

    ops = test_entity.observedPropertiesFromExperimentReference(
        result.experimentReference
    )
    assert len(ops) > 0
    for op in ops:
        assert not (
            ser.get(op.targetProperty.identifier)
        ), f"Expected the series representation with aggregation to have no values for {op}"
        vp = VirtualObservedProperty(
            baseObservedProperty=op,
            aggregationMethod=PropertyAggregationMethod(
                identifier=PropertyAggregationMethodEnum.mean
            ),
        )
        assert (
            ser.get(vp.virtualTargetPropertyIdentifier) is not None
        ), f"Expected the series representation with aggregation to have 1 value for key {vp.virtualTargetPropertyIdentifier}"


def test_required_constitutive_properties_present(
    entity_for_parameterized_experiment: tuple[Entity, Experiment],
) -> None:

    test_entity, test_experiment = entity_for_parameterized_experiment
    if not test_experiment.requiredProperties:
        pytest.skip("There are no required properties to test")

    assert CheckRequiredConstitutivePropertyValuesPresent(
        entity=test_entity, experiment=test_experiment
    ), "Expected the test entity fixture to have all properties required by the test experiment"

    # Remove a constitutive property
    test_entity_broken = test_entity.model_copy(
        update={
            "constitutive_property_values": test_entity.constitutive_property_values[1:]
        }
    )
    assert not CheckRequiredConstitutivePropertyValuesPresent(
        entity=test_entity_broken, experiment=test_experiment
    ), (
        "Expected that after removing a constitutive property the entity would not have all properties required "
        "by the test experiment"
    )
