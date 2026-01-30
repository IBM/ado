# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
from typing import Any

import pytest

from orchestrator.core.discoveryspace.samplers import sample_random_entity_from_space
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.entity import Entity
from orchestrator.schema.entityspace import EntitySpaceRepresentation
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.measurementspace import (
    MeasurementSpace,
    MeasurementSpaceConfiguration,
)
from orchestrator.schema.property import (
    AbstractPropertyDescriptor,
    ConstitutiveProperty,
)
from orchestrator.schema.property_value import ConstitutivePropertyValue


def test_entity_space_from_measurement_space(
    measurement_space: MeasurementSpace,
) -> None:

    # The input measurement space has two experiment both requiring a single
    # constitutive property called "smiles" with no domain
    # The entity space should have one constitutive property called smiles

    entitySpace = (
        measurement_space.compatibleEntitySpace()
    )  # type: EntitySpaceRepresentation

    assert len(entitySpace.constitutiveProperties) == 1
    assert entitySpace.constitutiveProperties[0].identifier == "smiles"


def test_entity_space_compatibility_with_measurement_space() -> None:

    cp1 = ConstitutiveProperty(
        identifier="gpu_model",
        propertyDomain=PropertyDomain(values=["A100", "L40S"]),
    )

    cp2 = ConstitutiveProperty(
        identifier="batch_size",
        propertyDomain=PropertyDomain(domainRange=[1, 129], interval=1),
    )

    cp3 = ConstitutiveProperty(
        identifier="memory",
        propertyDomain=PropertyDomain(
            variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE, domainRange=[0, 256]
        ),
    )

    experiment_one = Experiment(
        identifier="experiment_one",
        actuatorIdentifier="test",
        targetProperties=[AbstractPropertyDescriptor(identifier="throughput")],
        requiredProperties=(cp1, cp2),
    )

    experiment_two = Experiment(
        identifier="experiment_two",
        actuatorIdentifier="test",
        targetProperties=[AbstractPropertyDescriptor(identifier="oom")],
        requiredProperties=(cp3, cp1),
    )

    # This test should involve properties that have defined domains

    measurement_space_conf = MeasurementSpaceConfiguration(
        experiments=[experiment_one, experiment_two]
    )
    measurement_space = MeasurementSpace(measurement_space_conf)

    # Test that a compatible entity space is correctly identified as compatible
    # First an entity space that is identical in domain to the measurement space
    entity_space = EntitySpaceRepresentation([cp1, cp2, cp3])
    assert measurement_space.checkEntitySpaceCompatible(entity_space)

    # An entity space with properties whose domain are a subdomain of the measurement space properties
    ConstitutiveProperty(
        identifier="gpu_model",
        propertyDomain=PropertyDomain(values=["A100"]),
    )
    cp2 = ConstitutiveProperty(
        identifier="batch_size",
        propertyDomain=PropertyDomain(values=[2, 4, 8]),
    )
    entity_space = EntitySpaceRepresentation([cp1, cp2, cp3])

    assert measurement_space.checkEntitySpaceCompatible(entity_space)

    # Test that an entity space with a missing property is correct identified as incompatible

    entity_space = EntitySpaceRepresentation([cp1, cp2])

    with pytest.raises(
        ValueError,
        match="Identified a measurement space constitutive property not in entity space",
    ):
        measurement_space.checkEntitySpaceCompatible(entity_space)

    # Test that an entity space with a mis-spelt property is correct identified as incompatible
    cp3_mis_spelt = ConstitutiveProperty(
        identifier="memor",  # codespell:ignore
        propertyDomain=PropertyDomain(
            variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE, domainRange=[0, 256]
        ),
    )

    entity_space = EntitySpaceRepresentation([cp1, cp2, cp3_mis_spelt])
    with pytest.raises(
        ValueError,
        match="Identified a measurement space constitutive property not in entity space",
    ):
        measurement_space.checkEntitySpaceCompatible(entity_space)

    # Test that an entity space which has a property whose domain does not match the experiment
    # is correct identified as incompatible
    cp1_wrong_values = ConstitutiveProperty(
        identifier="gpu_model",
        propertyDomain=PropertyDomain(values=["A100", "V100"]),
    )

    entity_space = EntitySpaceRepresentation([cp1_wrong_values, cp2, cp3])
    with pytest.raises(
        ValueError,
        match="Identified an entity space dimension not compatible with the measurement space requirements",
    ):
        measurement_space.checkEntitySpaceCompatible(entity_space)

    # Test that an entity space which has a property that is not an input for any experiment
    # in the measurement space is incompatible by default (strict is True)

    cp_extra = ConstitutiveProperty(
        identifier="redundant_extra_property",
        propertyDomain=PropertyDomain(
            variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE, domainRange=[0, 256]
        ),
    )

    entity_space = EntitySpaceRepresentation([cp_extra, cp1, cp2, cp3])
    with pytest.raises(
        ValueError,
        match=f"Identified an entity space dimension, {cp_extra}, that is not required "
        "for any experiment in the measurement space and is hence redundant",
    ):
        measurement_space.checkEntitySpaceCompatible(entity_space)

    # That setting strict to False allows above
    assert measurement_space.checkEntitySpaceCompatible(entity_space, strict=False)


def test_entity_space_representation(
    constitutive_property_configuration_general: list[ConstitutiveProperty],
    constitutive_property_configuration_general_yaml: dict[str, Any],  # noqa: ANN401
) -> None:
    rep = EntitySpaceRepresentation.representationFromConfiguration(
        constitutive_property_configuration_general
    )

    assert len(rep.constitutiveProperties) == len(
        constitutive_property_configuration_general
    )
    assert len(rep.constitutiveProperties) > 1

    if rep.isDiscreteSpace:
        assert rep.size == 1497
    else:
        # Test that you can't calculate size of continuous space
        with pytest.raises(AttributeError):
            _ = rep.size

    assert rep.config == constitutive_property_configuration_general

    assert [
        e.model_dump(exclude_defaults=True, exclude_unset=True) for e in rep.config
    ] == constitutive_property_configuration_general_yaml


def test_entity_in_space(
    constitutive_property_configuration_general: list[ConstitutiveProperty],
) -> None:

    es = EntitySpaceRepresentation(
        constitutiveProperties=constitutive_property_configuration_general
    )

    if not es.isDiscreteSpace:
        pytest.xfail("We currently do not have a way to sample from continuous space")

    entity = sample_random_entity_from_space(es)
    for v in entity.constitutive_property_values:
        assert es.propertyWithIdentifier(
            v.property.identifier
        ).propertyDomain.valueInDomain(v.value)

    assert es.isEntityInSpace(entity)
    assert es.isEntityCompatibleWithSpace(entity)

    # Test if the entity has more properties it is not in space but is compatible with it
    extraConstitutiveProperty = ConstitutiveProperty(
        propertyDomain=PropertyDomain(values=[1, 2, 3, 4]), identifier="extra_prop"
    )
    extraConstitutivePropertyValue = ConstitutivePropertyValue(
        value=2, property=extraConstitutiveProperty.descriptor()
    )
    newEntity = entity.model_copy(
        update={
            "constitutive_property_values": [
                *list(entity.constitutive_property_values),
                extraConstitutivePropertyValue,
            ]
        }
    )
    assert not es.isEntityInSpace(newEntity)
    assert es.isEntityCompatibleWithSpace(newEntity)

    # Test an entity with less constitutive propertiesit is not in space AND is not compatible with it
    newEntity = entity.model_copy(
        update={
            "constitutive_property_values": entity.constitutive_property_values[:-1]
        }
    )

    assert not es.isEntityInSpace(newEntity)
    assert not es.isEntityCompatibleWithSpace(newEntity)

    # Test if the value of an entity is outside the bounds of one of the dimensions it fails
    acceleratorPropValue = entity.valueForConstitutivePropertyIdentifier(
        "AcceleratorType"
    )
    assert acceleratorPropValue
    acceleratorPropValue.value = "H100"
    assert (
        acceleratorPropValue
        not in es.propertyWithIdentifier("AcceleratorType").propertyDomain.values
    )
    values = [acceleratorPropValue, *list(entity.constitutive_property_values[1:])]

    newEntity = Entity(constitutive_property_values=tuple(values))
    assert not es.isEntityInSpace(newEntity)
    assert not es.isEntityCompatibleWithSpace(newEntity)


def test_entity_space_rich_print(
    constitutive_property_configuration_general: list[ConstitutiveProperty],
) -> None:
    from rich.console import Console

    ## Add an Unknown property

    constitutive_property_configuration_general.append(
        ConstitutiveProperty(
            identifier="unknown_prop",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.UNKNOWN_VARIABLE_TYPE
            ),
        )
    )

    ## Add a binary property
    constitutive_property_configuration_general.append(
        ConstitutiveProperty(
            identifier="binary_prop",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE
            ),
        )
    )

    es = EntitySpaceRepresentation(
        constitutiveProperties=constitutive_property_configuration_general
    )

    Console().print(es)


def test_entity_space_dimension_values(
    measurement_space_from_single_parameterized_experiment: MeasurementSpace,
) -> None:

    es = measurement_space_from_single_parameterized_experiment.compatibleEntitySpace()
    es.dimension_values()

    # add a continuous property
    es_continuous = EntitySpaceRepresentation(
        constitutiveProperties=[
            *es.constitutiveProperties,
            ConstitutiveProperty(
                identifier="continuous_prop",
                propertyDomain=PropertyDomain(
                    variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
                ),
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Cannot return dimension_values for a space with continuous or unknown dimensions",
    ):
        es_continuous.dimension_values()

    binary_entity_space = EntitySpaceRepresentation(
        constitutiveProperties=[
            ConstitutiveProperty(
                identifier="binary_constitutive_property",
                propertyDomain=PropertyDomain(
                    variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE
                ),
            )
        ]
    )
    assert binary_entity_space.dimension_values() == {
        "binary_constitutive_property": [False, True]
    }


def test_entity_space_iterators(
    measurement_space_from_single_parameterized_experiment: MeasurementSpace,
) -> None:

    exp = measurement_space_from_single_parameterized_experiment.experiments[0]
    if not exp.requiredConstitutiveProperties:
        pytest.skip(
            "No required properties in the measurement space - cannot create entity space"
        )

    es = measurement_space_from_single_parameterized_experiment.compatibleEntitySpace()
    assert es.size == 300, "Expected there to be 300 points in the test spaces"

    i = 0
    sequential = []
    ids = [cp.identifier for cp in es.constitutiveProperties]
    point = None
    for i, point in enumerate(es.sequential_point_iterator()):  # noqa: B007
        sequential.append(point)
        assert es.isPointInSpace(
            dict(zip(ids, point, strict=True))
        ), "Expected all points iterated over to be in space"

    assert i != 0, "Expected points to be returned"
    assert (
        i == es.size - 1
    ), "Expected the number of points iterated over to be equal to the space size"

    assert len(set(sequential)) == len(
        sequential
    ), "Expected no points to be duplicated"

    sequential_repeat = list(es.sequential_point_iterator())
    assert (
        sequential == sequential_repeat
    ), "Expected sequential iterator to be deterministic"

    i = 0
    random = []
    for i, point in enumerate(es.random_point_iterator()):  # noqa: B007
        random.append(point)
        assert es.isPointInSpace(
            dict(zip(ids, point, strict=True))
        ), "Expected all points iterated over to be in space"

    assert i != 0, "Expected points to be returned"
    assert (
        i == es.size - 1
    ), "Expected the number of points iterated over to be equal to the space size"

    assert (
        random[:5] != sequential[:5]
    ), "Expected the first five random points to not be identical to first five sequential points"

    assert len(set(random)) == len(random), "Expected no points to be duplicated"


def test_entity_space_updates_open_categorical_property() -> None:
    es = EntitySpaceRepresentation(
        constitutiveProperties=[
            ConstitutiveProperty(
                identifier="open_categorical_prop",
                propertyDomain=PropertyDomain(
                    variableType=VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE,
                    values=["a", "b", "c"],
                ),
            )
        ]
    )
    assert (
        es.constitutiveProperties[0].propertyDomain.variableType
        == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE
    ), "Expected open categorical property to be updated to categorical"
    assert es.constitutiveProperties[0].propertyDomain.values == [
        "a",
        "b",
        "c",
    ], "Expected values to be retained"
