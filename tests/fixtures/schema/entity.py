# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import typing
from collections.abc import Callable

import pytest

from orchestrator.core.samplestore.csv import CSVSampleStore
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.entity import Entity
from orchestrator.schema.entityspace import EntitySpaceRepresentation
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.measurementspace import (
    MeasurementSpace,
    MeasurementSpaceConfiguration,
)
from orchestrator.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from orchestrator.schema.property import (
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
)
from orchestrator.schema.property_value import (
    ConstitutivePropertyValue,
    ValueTypeEnum,
)
from orchestrator.schema.result import ValidMeasurementResult


@pytest.fixture
def value_for_value_type() -> (
    typing.Callable[[ValueTypeEnum], int | float | str | bytes | None]
):
    def _value_for_value_type(
        value_type: ValueTypeEnum = ValueTypeEnum.NUMERIC_VALUE_TYPE,
    ) -> int | float | str | bytes | None:
        import os
        import random
        import string

        import numpy as np

        vector_length = 3
        vector_width = 2
        string_length = 8
        blob_length_bytes = 16
        rng = np.random.default_rng()

        if value_type == ValueTypeEnum.NUMERIC_VALUE_TYPE:
            value = rng.random()
        elif value_type == ValueTypeEnum.VECTOR_VALUE_TYPE:
            value = list(rng.random(vector_length, vector_width))
        elif value_type == ValueTypeEnum.STRING_VALUE_TYPE:
            value = "".join(
                random.choices(string.ascii_letters + string.digits, k=string_length)
            )
        elif value_type == ValueTypeEnum.BLOB_VALUE_TYPE:
            value = os.urandom(blob_length_bytes)
        else:
            value = None

        return value

    return _value_for_value_type


@pytest.fixture
def values_for_properties(
    value_for_value_type: typing.Callable[
        [ValueTypeEnum], int | float | str | bytes | None
    ],
) -> typing.Callable[
    [
        ConstitutiveProperty
        | ObservedProperty
        | ConstitutivePropertyDescriptor
        | list[ConstitutiveProperty]
        | list[ObservedProperty]
        | list[ConstitutivePropertyDescriptor, ValueTypeEnum]
    ],
    list[ConstitutivePropertyValue | ObservedPropertyValue],
]:
    def value_class_for_property(
        p: ConstitutiveProperty | ObservedProperty | ConstitutivePropertyDescriptor,
    ) -> type[ObservedPropertyValue, ConstitutivePropertyValue]:
        return (
            ObservedPropertyValue
            if isinstance(p, ObservedProperty)
            else ConstitutivePropertyValue
        )

    def _values_for_properties(
        properties: (
            ConstitutiveProperty
            | ConstitutivePropertyDescriptor
            | ObservedProperty
            | list[ConstitutiveProperty]
            | list[ObservedProperty]
            | list[ConstitutivePropertyDescriptor]
        ),
        value_type: ValueTypeEnum = ValueTypeEnum.NUMERIC_VALUE_TYPE,
    ) -> list[ConstitutivePropertyValue | ObservedPropertyValue]:
        if not isinstance(properties, list):
            properties = [properties]

        return [
            value_class_for_property(p)(
                value=value_for_value_type(value_type),
                valueType=value_type,
                property=p,
            )
            for p in properties
        ]

    return _values_for_properties


@pytest.fixture
def property_values(
    values_for_properties: typing.Callable[
        [
            ConstitutiveProperty
            | ObservedProperty
            | ConstitutivePropertyDescriptor
            | list[ConstitutiveProperty]
            | list[ObservedProperty]
            | list[ConstitutivePropertyDescriptor],
            ValueTypeEnum,
        ],
        list[ConstitutivePropertyValue | ObservedPropertyValue],
    ],
    experiment: Experiment,
) -> list[ObservedPropertyValue | ConstitutivePropertyValue]:

    return values_for_properties(
        experiment.observedProperties,
        ValueTypeEnum.NUMERIC_VALUE_TYPE,
    )


@pytest.fixture
def entity(
    experiment: Experiment,
    constitutive_properties: list[ConstitutiveProperty],
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
    values_for_properties: list[ObservedPropertyValue | ConstitutivePropertyValue],
) -> Entity:

    entity_identifier = "COOH"
    generator_id = "testgen"

    constitutive_property_values: list[ConstitutivePropertyValue] = (
        values_for_properties(constitutive_properties)
    )

    measurement_results = [
        ValidMeasurementResult(
            entityIdentifier=entity_identifier, measurements=property_values
        )
    ]

    return Entity(
        identifier=entity_identifier,
        generatorid=generator_id,
        constitutive_property_values=tuple(constitutive_property_values),
        measurement_results=measurement_results,
    )


@pytest.fixture(params=["required_only_in_es", "optional_in_es"])
def entity_for_parameterized_experiment(
    parameterized_experiment: ParameterizedExperiment,
    global_registry: ActuatorRegistry,
    request: pytest.FixtureRequest,
) -> tuple[Entity, ParameterizedExperiment]:
    """Returns various entities that are compatible with parameterized experiments

    For required-only, the entity constitutive properties are == experiments
    For with-optional, an optional property is moved into the entity space
    """

    # Create a measurement space with the parameterized experiment reference
    ms = MeasurementSpace(
        configuration=MeasurementSpaceConfiguration(
            experiments=[parameterized_experiment]
        )
    )

    # Create an entity space compatible with the measurement space
    es = ms.compatibleEntitySpace()
    if request.param == "optional_in_es":
        # add an optional parameter to the entity space
        es = EntitySpaceRepresentation(
            constitutiveProperties=es.constitutiveProperties
            + list(parameterized_experiment.optionalProperties[:1])
        )

    # We might get required_only_in_es with an experiment which has no_required
    # This is not possible
    if (
        request.param == "required_only_in_es"
        and len(parameterized_experiment.requiredProperties) == 0
    ):
        pytest.skip(
            "Can't create an entity from a space that only contains required properties given an experiment that has no required properties"
        )

    from orchestrator.core.discoveryspace.samplers import (
        ExplicitEntitySpaceGridSampleGenerator,
        WalkModeEnum,
    )

    # Sample an entity from the entity space
    s = ExplicitEntitySpaceGridSampleGenerator(mode=WalkModeEnum.RANDOM)
    entity = None
    for e in s.entitySpaceIterator(es):
        entity = e[0]
        break

    # If the experiment has a required observed property, add a value for it to the entity
    props = parameterized_experiment.requiredObservedProperties
    if len(props) == 1:
        entity.add_measurement_result(
            ValidMeasurementResult(
                entityIdentifier=entity.identifier,
                measurements=[ObservedPropertyValue(property=props[0], value=3)],
            )
        )

    return entity, parameterized_experiment


@pytest.fixture
def random_entities(csv_sample_store: CSVSampleStore) -> Callable[[int], list[Entity]]:
    def _random_entities(quantity: int) -> list[Entity]:
        return csv_sample_store.entities[:quantity]

    return _random_entities
