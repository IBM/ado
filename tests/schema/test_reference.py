# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import re

import numpy.random
import pytest

from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.measurementspace import MeasurementSpace
from orchestrator.schema.observed_property import ObservedPropertyValue
from orchestrator.schema.property import (
    ConstitutivePropertyDescriptor,
)
from orchestrator.schema.property_value import ConstitutivePropertyValue
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.result import ValidMeasurementResult


def test_parameterized_reference_equality(
    parameterized_references: list[ExperimentReference],
) -> None:

    for r in parameterized_references:
        assert r == r
        assert r != r.model_copy(update={"actuatorIdentifier": "changed"})


def test_experiment_reference_matching(
    experiment: Experiment, experiment_reference: ExperimentReference
) -> None:
    assert experiment.reference == experiment_reference


def test_reference_of_parameterizable_experiment_has_no_parameters(
    mock_parameterizable_experiment: Experiment,
) -> None:

    assert mock_parameterizable_experiment.reference.parameterization is None


def test_create_parameterizable_experiment_reference_with_parameters(
    global_registry: ActuatorRegistry,
    mock_parameterizable_experiment: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:

    import copy

    ### Test we can create a valid experiment reference with parameters
    ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=customParameterization,
    )

    ### Test creation fails if parameters are not valid parameters

    # Test parameterization with duplicate property fails on validation
    with pytest.raises(
        ValueError,
        match="The parameterization contains multiple values for same property",
    ):
        ExperimentReference(
            actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
            experimentIdentifier=mock_parameterizable_experiment.identifier,
            parameterization=customParameterization + customParameterization[:1],
        ).validate_parameterization()

    # Test parameterization with incorrectly named property fails on validation
    incorrectParameterization = copy.deepcopy(customParameterization)
    pv = incorrectParameterization[0]
    incorrectParameterization[0] = ConstitutivePropertyValue(
        value=pv.value, property=ConstitutivePropertyDescriptor(identifier="tes_opt1")
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "parameterized properties not in optionalProperties list. Missing: ['tes_opt1']"
        ),
    ):
        ExperimentReference(
            actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
            experimentIdentifier=mock_parameterizable_experiment.identifier,
            parameterization=incorrectParameterization,
        ).validate_parameterization()

    # Test parameterization with same values as default fails on validation
    incorrectParameterization = copy.deepcopy(customParameterization)
    incorrectParameterization[0] = (
        mock_parameterizable_experiment.defaultParameterization[0]
    )
    with pytest.raises(
        ValueError,
        match=f"Custom parameterization for property {incorrectParameterization[0].property.identifier} "
        f"with value {incorrectParameterization[0].value} has same value as default parameterization: "
        f"{mock_parameterizable_experiment.defaultParameterization[0]}",
    ):
        ExperimentReference(
            actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
            experimentIdentifier=mock_parameterizable_experiment.identifier,
            parameterization=incorrectParameterization,
        ).validate_parameterization()


def test_parameterized_experiment_reference_equality(
    global_registry: ActuatorRegistry,
    mock_parameterizable_experiment: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:

    ref = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=customParameterization,
    )

    ref2 = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=customParameterization,
    )

    # Test that two references with same parameterization are equal
    assert ref == ref2

    # Test that two references to same experiment one with and one without parameterization are different
    assert ref != mock_parameterizable_experiment.reference

    # Test that two references to same experiment with different parameterization options are different
    ref2 = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=customParameterization[:1],
    )

    assert ref != ref2


def test_parameterized_experiment_retrieval_from_registry(
    global_registry: ActuatorRegistry,
    mock_parameterizable_experiment: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:

    # Test we can retrieve the experiment from the registry using its own reference
    assert global_registry.experimentForReference(
        mock_parameterizable_experiment.reference
    )

    # Test we can retrieve the experiment from the registry using a parameterized reference to it

    ref = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=customParameterization,
    )

    assert global_registry.experimentForReference(ref)


def test_entity_property_values_from_experiment_reference_parameterized(
    global_registry: ActuatorRegistry,
    mock_parameterizable_experiment: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:
    """Tests that Entity.propertyValuesFromExperimentReference works for parameterized experiments"""

    # 1. Create the parameterized experiment
    # 2. Create an entity with the required constitutive properties
    # 3. Add property values for a parameterized experiment

    #
    # 1. Create an experiment references
    #

    # Parameterized Experiment Reference - values for this experiment will be added to entity
    parameterized_ref = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=customParameterization,
    )

    # Reference to base (non-parameterized) experiment - for testing
    base_ref = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
    )

    # A different parameterized version of the base experiment - for testing
    different_parameterized_ref = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=customParameterization[:-1],
    )

    #
    # Part 2: Creating an entity with required constitutive properties
    #

    # Create a measurement space with the parameterized experiment reference
    ms = MeasurementSpace.measurementSpaceFromExperimentReferences([parameterized_ref])

    # Create an entity space compatible with the measurement space
    es = ms.compatibleEntitySpace()
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

    #
    # 3. Add property values for a parameterized experiment
    #

    #  Get the actual parameterized experiment from the parameterized reference
    exp = ms.experimentForReference(parameterized_ref)

    # Add a measurement
    values = []
    for op in exp.observedProperties:
        pv = ObservedPropertyValue(
            property=op, value=numpy.random.default_rng().random()
        )
        values.append(pv)

    entity.add_measurement_result(
        result=ValidMeasurementResult(
            entityIdentifier=entity.identifier, measurements=values
        )
    )

    #
    # Tests
    #

    print(f"Added values for {parameterized_ref}")
    print(
        f"Testing we do not retrieve values for {base_ref} or {different_parameterized_ref}"
    )

    assert (
        len(entity.propertyValuesFromExperimentReference(parameterized_ref)) > 0
    ), f"Expected entity to have values for observed properties of parameterized experiment {parameterized_ref} but it has None"

    assert len(entity.propertyValuesFromExperimentReference(parameterized_ref)) == len(
        exp.observedProperties
    ), (
        f"Expected entity to have {len(exp.observedProperties)} values for observed properties of parameterized "
        f"experiment {parameterized_ref} but it has {len(entity.propertyValuesFromExperimentReference(parameterized_ref))}"
    )

    assert (
        len(entity.propertyValuesFromExperimentReference(base_ref)) == 0
    ), f"Entity had values added for {parameterized_ref}. It should not return any values when given base experiment reference {base_ref}"

    assert (
        len(entity.propertyValuesFromExperimentReference(different_parameterized_ref))
        == 0
    ), f"Entity had values added for {parameterized_ref}. It should not return any values when given base experiment reference {different_parameterized_ref}"


def test_experiment_reference_from_invalid_string() -> None:

    with pytest.raises(
        ValueError,
        match=re.escape(
            "String, string_with_no_separator is not a valid representation of an ExperimentReference. "
            "At least one '.' is required to separate actuator id from experiment id. "
            "If actuator id contains a period this method will not be able to parse the id from the reference string representation"
        ),
    ):
        ExperimentReference.referenceFromString("string_with_no_separator")

    # This should work as there only needs to be 1 .
    ExperimentReference.referenceFromString("string.with.more_separators")


def test_experiment_reference_equality_non_reference() -> None:

    ref = ExperimentReference(experimentIdentifier="test", actuatorIdentifier="test")
    assert ref != "test.test"
