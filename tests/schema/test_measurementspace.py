# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import re

import pytest

from orchestrator.core.discoveryspace.samplers import sample_random_entity_from_space
from orchestrator.modules.actuators.registry import (
    ActuatorRegistry,
    UnknownActuatorError,
    UnknownExperimentError,
)
from orchestrator.schema.entityspace import EntitySpaceRepresentation
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.measurementspace import (
    MeasurementSpace,
    MeasurementSpaceConfiguration,
)
from orchestrator.schema.observed_property import ObservedPropertyValue
from orchestrator.schema.property import ConstitutiveProperty
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest
from orchestrator.schema.result import ValidMeasurementResult
from orchestrator.schema.virtual_property import (
    PropertyAggregationMethod,
    PropertyAggregationMethodEnum,
    VirtualObservedProperty,
)


def test_measurement_space_config(
    measurement_space: MeasurementSpace,
    measurement_space_configuration: list[ExperimentReference],
) -> None:
    assert measurement_space.experimentReferences == measurement_space_configuration


def test_measurement_space_from_parameterized_selectors(
    parameterizable_experiments: list[Experiment],
    parameterized_selectors: list[ExperimentReference],
    global_registry: ActuatorRegistry,
) -> None:

    # Test does not allow duplicate parameterized experiments
    # Test does allow different parameterization of same base experiment

    for e in parameterizable_experiments:
        assert global_registry.experimentForReference(e.reference)

    for s in parameterized_selectors:
        assert global_registry.experimentForReference(s)

    ms = MeasurementSpace.measurementSpaceFromSelection(
        selectedExperiments=parameterized_selectors
    )

    # Currently all references must match exactly
    # So passing non-parameterized version of an experiment, or reference to it, to experimentForReference
    # will not return parameterized versions of that experiment
    for r in parameterizable_experiments:
        assert not ms.experimentForReference(r.reference)

    # Test that experimentForReference returns the correct parameterized experiment
    for r in parameterized_selectors:
        assert ms.experimentForReference(r)

    # Test that the target properties are as expected
    props = []
    for e in parameterizable_experiments:
        props.extend(e.targetProperties)

    targets = ms.targetProperties
    for target in props:
        assert target in targets

    # Test that the observed properties are as expected
    base_observed_properties = []
    expected_observed_properties_identifiers = []
    for e in parameterizable_experiments:
        base_observed_properties.extend(e.observedProperties)
        expected_observed_properties_identifiers.extend(
            [
                f"{op.experimentReference.experimentIdentifier}-{op.targetProperty.identifier}"
                for op in e.observedProperties
            ]
        )

    parameterised_observed_properties = ms.observedProperties
    for observed_property in base_observed_properties:
        assert observed_property not in parameterised_observed_properties
        assert observed_property.identifier in expected_observed_properties_identifiers

    # How does experiment equality work?
    # With parameterized experiment a measurement space could have two parameterized
    # versions of same base experiment ...
    for e in parameterizable_experiments:
        assert len(ms.observedPropertiesForExperiment(e)) == 0

    for r in parameterized_selectors:
        assert len(ms.observedPropertiesForExperimentReference(r)) != 0


def test_measurement_space_with_parameterized_experiments_ser_dser(
    parameterized_selectors: list[ExperimentReference],
    global_registry: ActuatorRegistry,
) -> None:

    ms = MeasurementSpace.measurementSpaceFromSelection(
        selectedExperiments=parameterized_selectors
    )

    assert isinstance(ms.experiments[0], ParameterizedExperiment), (
        f"Expected experiments in measurement space to be of type ParameterizedExperiment but they are not. "
        f"Parameterized selector: {parameterized_selectors[0]}. "
        f"Experiment in measurement space: {ms.experiments[0]}"
    )

    config = ms.selfContainedConfig
    assert config.experiments[0].parameterization

    ser = config.model_dump()

    dser = MeasurementSpaceConfiguration.model_validate(ser)
    ms = MeasurementSpace(configuration=dser)

    assert ms.experiments[0].parameterization

    assert isinstance(ms.experiments[0], ParameterizedExperiment), (
        f"Expected experiments in measurement space on dser to be of type ParameterizedExperiment but they are not. "
        f"Parameterized selector: {parameterized_selectors[0]}. "
        f"Experiment in measurement space: {ms.experiments[0]}"
    )


def test_measurement_space_from_parameterized_references(
    parameterizable_experiments: list[Experiment],
    parameterized_references: list[ExperimentReference],
    global_registry: ActuatorRegistry,
) -> None:
    # Test does not allow duplicate parameterized experiments
    # Test does allow different parameterization of same base experiment

    for e in parameterizable_experiments:
        assert global_registry.experimentForReference(e.reference)

    for r in parameterized_references:
        assert global_registry.experimentForReference(r)

    ms = MeasurementSpace.measurementSpaceFromExperimentReferences(
        experimentReferences=parameterized_references
    )

    # Try rich print
    from rich.console import Console

    Console().print(ms)

    # Currently all references must match exactly
    # So passing non-parameterized version of an experiment, or reference to it, to experimentForReference
    # will not return parameterized versions of that experiment
    for r in parameterizable_experiments:
        assert not ms.experimentForReference(r.reference)

    # Test that experimentForReference returns the correct parameterized experiment
    for r in parameterized_references:
        assert ms.experimentForReference(r)

    # Test that the target properties are as expected
    props = []
    for e in parameterizable_experiments:
        props.extend(e.targetProperties)

    targets = ms.targetProperties
    for target in props:
        assert target in targets

    # Test that the observed properties are as expected
    base_observed_properties = []
    expected_observed_properties_identifiers = []
    for e in parameterizable_experiments:
        base_observed_properties.extend(e.observedProperties)
        expected_observed_properties_identifiers.extend(
            [
                f"{op.experimentReference.experimentIdentifier}-{op.targetProperty.identifier}"
                for op in e.observedProperties
            ]
        )

    parameterised_observed_properties = ms.observedProperties
    for observed_property in base_observed_properties:
        assert observed_property not in parameterised_observed_properties
        assert observed_property.identifier in expected_observed_properties_identifiers

    # How does experiment equality work?
    # With parameterized experiment a measurement space could have two parameterized
    # versions of same base experiment ...
    for e in parameterizable_experiments:
        assert len(ms.observedPropertiesForExperiment(e)) == 0

    for r in parameterized_references:
        assert len(ms.observedPropertiesForExperimentReference(r)) != 0


def test_measurement_space(measurement_space: MeasurementSpace) -> None:
    """Test a measurement space"""

    space = measurement_space

    assert len(space.experiments) == 1
    assert len(space.experimentReferences) == 1

    for ref in space.experimentReferences:
        if ref.experimentIdentifier == "transformer-toxicity-inference-experiment":
            assert len(space.observedPropertiesForExperimentReference(ref)) == 8

    assert len(space.targetProperties) == 8
    assert len(space.observedProperties) == 8

    # From the test_generations.csv
    expectedTargets = [
        "logws",
        "ld50",
        "biodegradation halflife",
        "pka",
        "mw",
        "logd",
        "loghl",
        "bcf",
        "scscore",
    ]

    for p in space.observedProperties:
        assert p.targetProperty.identifier in expectedTargets


def test_experiment_selector(
    measurement_space_configuration: list[ExperimentReference],
) -> None:
    selectedExperiments = measurement_space_configuration

    assert len(selectedExperiments) == 1

    assert (
        selectedExperiments[0].experimentIdentifier
        == "transformer-toxicity-inference-experiment"
    )

    assert selectedExperiments[0].actuatorIdentifier == "replay"


def test_space_with_unknown_experiment(
    parameterized_references: list[ExperimentReference],
) -> None:
    """Test correct error is raised if an experiment cannot be found"""

    with pytest.raises(UnknownActuatorError):
        MeasurementSpace.measurementSpaceFromSelection(
            selectedExperiments=[
                *parameterized_references,
                ExperimentReference(
                    actuatorIdentifier="unknown_actuator",
                    experimentIdentifier="unknown_experiment",
                ),
            ]
        )

    with pytest.raises(UnknownExperimentError):
        MeasurementSpace.measurementSpaceFromSelection(
            selectedExperiments=[
                *parameterized_references,
                ExperimentReference(
                    actuatorIdentifier="mock", experimentIdentifier="unknown_experiment"
                ),
            ]
        )


def test_supported_experiments(
    parameterized_references: list[ExperimentReference],
) -> None:
    """Test  MeasurementSpace returns supported/deprecated experiments correctly"""

    ms = MeasurementSpace.measurementSpaceFromSelection(
        selectedExperiments=parameterized_references
    )
    assert len(ms.supported_experiments) == len(parameterized_references)

    assert not ms.deprecated_experiments
    assert not ms.has_deprecated_experiments


def test_independent_and_dependent_experiments_single(
    measurement_space_from_single_parameterized_experiment: MeasurementSpace,
) -> None:
    """Test MeasurementSpace returns independent and dependent experiments"""

    ms = measurement_space_from_single_parameterized_experiment
    if [r for r in ms.experiments if r.requiredObservedProperties]:
        assert ms.dependentExperiments
        assert len(ms.dependentExperiments) == 1

    assert len(ms.independentExperiments) == len(ms.experiments) - len(
        ms.dependentExperiments
    )


def test_independent_and_dependent_experiments_multiple(
    measurement_space_from_multiple_parameterized_experiments: MeasurementSpace,
) -> None:
    """Test MeasurementSpace returns independent and dependent experiments"""

    ms = measurement_space_from_multiple_parameterized_experiments
    if [r for r in ms.experiments if r.requiredObservedProperties]:
        assert ms.dependentExperiments
        assert len(ms.dependentExperiments) == 1

    assert len(ms.independentExperiments) == len(ms.experiments) - len(
        ms.dependentExperiments
    )


def test_is_consistent_single(
    measurement_space_from_single_parameterized_experiment: MeasurementSpace,
) -> None:

    if measurement_space_from_single_parameterized_experiment.dependentExperiments:
        ### If there is only a single experiment, but it requires others, the MeasurementSpace is not consistent
        assert not measurement_space_from_single_parameterized_experiment.isConsistent
    else:
        assert measurement_space_from_single_parameterized_experiment.isConsistent


def test_is_consistent_multiple(
    measurement_space_from_multiple_parameterized_experiments: MeasurementSpace,
) -> None:

    ### Need an experiment with dependencies
    assert measurement_space_from_multiple_parameterized_experiments.isConsistent


def _shared_property_with_identifier_tests(
    parameterized_measurement_space: MeasurementSpace,
) -> None:
    # Test all property types including virtual
    for exp in parameterized_measurement_space.experiments:
        for op in exp.observedProperties:
            assert parameterized_measurement_space.propertyWithIdentifierInSpace(
                op.identifier
            )
            vp = VirtualObservedProperty(
                baseObservedProperty=op,
                aggregationMethod=PropertyAggregationMethod(
                    identifier=PropertyAggregationMethodEnum.mean
                ),
            )
            assert parameterized_measurement_space.propertyWithIdentifierInSpace(
                vp.identifier
            )

            # Test an invalid vp identifier
            assert not parameterized_measurement_space.propertyWithIdentifierInSpace(
                "invalid_ident"
            )

            # Test an incorrect vp identifier
            assert not parameterized_measurement_space.propertyWithIdentifierInSpace(
                "some_experiment.some_property-mean"
            )


def test_property_with_identifier_in_space_single(
    measurement_space_from_single_parameterized_experiment: MeasurementSpace,
) -> None:

    _shared_property_with_identifier_tests(
        measurement_space_from_single_parameterized_experiment
    )


def test_property_with_identifier_in_space_multiple(
    measurement_space_from_multiple_parameterized_experiments: MeasurementSpace,
) -> None:

    _shared_property_with_identifier_tests(
        measurement_space_from_multiple_parameterized_experiments
    )


def test_check_entity_space_compatibility_multiple(
    measurement_space_from_multiple_parameterized_experiments: MeasurementSpace,
) -> None:

    # In particular test with an experiment that has optional properties

    es = (
        measurement_space_from_multiple_parameterized_experiments.compatibleEntitySpace()
    )
    assert measurement_space_from_multiple_parameterized_experiments.checkEntitySpaceCompatible(
        es
    ), "Expect the EntitySpace return by compatibleEntitySpace to be compatible"

    # Remove a required constitutive property and it should not be compatible

    es_test = EntitySpaceRepresentation(
        constitutiveProperties=es.constitutiveProperties[1:]
    )

    with pytest.raises(
        ValueError,
        match="Identified a measurement space constitutive property not in entity space",
    ) as expected_exception:
        measurement_space_from_multiple_parameterized_experiments.checkEntitySpaceCompatible(
            es_test
        )

    assert expected_exception is not None, (
        "Expect an entity space without all required constitutive properties for experiments in measurement space "
        "to not be compatible"
    )


def test_check_entity_space_compatibility_single(
    measurement_space_from_single_parameterized_experiment: MeasurementSpace,
) -> None:

    # In particular test with an experiment that has optional properties

    es = measurement_space_from_single_parameterized_experiment.compatibleEntitySpace()
    assert measurement_space_from_single_parameterized_experiment.checkEntitySpaceCompatible(
        es
    ), "Expect the EntitySpace return by compatibleEntitySpace to be compatible"

    # Remove a required constitutive property and it should not be compatible

    es_test = EntitySpaceRepresentation(
        constitutiveProperties=es.constitutiveProperties[1:]
    )
    if (
        len(
            measurement_space_from_single_parameterized_experiment.experiments[
                0
            ].requiredConstitutiveProperties
        )
        != 0
    ):
        with pytest.raises(
            ValueError,
            match="Identified a measurement space constitutive property not in entity space",
        ) as expected_exception:
            measurement_space_from_single_parameterized_experiment.checkEntitySpaceCompatible(
                es_test
            )

        assert expected_exception is not None, (
            "Expect an entity space without all required constitutive properties for experiments in measurement space "
            "to not be compatible"
        )
    else:
        # This space has no required properties so we can remove the constitutive property and should still work
        assert measurement_space_from_single_parameterized_experiment.checkEntitySpaceCompatible(
            es_test
        ), (
            "It is expected that if an optional property of an experiment in the "
            "measurement space is removed from the entity space it remains compatible"
        )


def test_check_entity_space_compatibility_optional_in_entity_space(
    measurement_space_from_multiple_parameterized_experiments: MeasurementSpace,
    optionalProperties: list[ConstitutiveProperty],
) -> None:
    """Test checkEntitySpaceCompatible works when some optional parameters have been moved to entityspace"""

    es = (
        measurement_space_from_multiple_parameterized_experiments.compatibleEntitySpace()
    )

    # Add an optional property of one of the experiments to the entity-space
    # None of the experiment in MS provide a value for the last optional property
    es_opt1 = EntitySpaceRepresentation(
        constitutiveProperties=es.constitutiveProperties + optionalProperties[-1:]
    )
    assert measurement_space_from_multiple_parameterized_experiments.checkEntitySpaceCompatible(
        es_opt1
    ), "Expect an EntitySpace containing an optional experimental property is compatible"

    # Add an optional property of one of the experiments to the entity-space that is already parameterized
    # by one or more experiments - this should fail
    es_opt2 = EntitySpaceRepresentation(
        constitutiveProperties=es.constitutiveProperties + optionalProperties[:1]
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Identified an entity space dimension, cp-test_opt1, "
            "that also has a custom parameterization in the measurement space. "
            "It is inconsistent for a property to have a custom parameterization in the measurement space "
            "and also be a dimension of the entityspace."
        ),
    ):
        # "Expect an EntitySpace containing an optional experimental property that also has a custom parameterization
        # to not be compatible"
        measurement_space_from_multiple_parameterized_experiments.checkEntitySpaceCompatible(
            es_opt2
        )

    # Modify the property in the EntitySpace so it is not a subdomain
    # This should fail
    optional_property = optionalProperties[-1]
    modified_domain = optional_property.propertyDomain.model_copy(
        update={"domainRange": [0, 110]}
    )
    modified_optional_property = optional_property.model_copy(
        update={"propertyDomain": modified_domain}
    )
    es_opt3 = EntitySpaceRepresentation(
        constitutiveProperties=[*es.constitutiveProperties, modified_optional_property]
    )
    with pytest.raises(
        ValueError,
        match="Identified an entity space dimension not compatible with the measurement space requirements",
    ) as expected_exception:
        measurement_space_from_multiple_parameterized_experiments.checkEntitySpaceCompatible(
            es_opt3
        ),

    assert expected_exception is not None, (
        "Expected an entity space containing an optional constitutive property of an experiment,"
        "and where the domain of this property was not a sub-domain of the experiments, to fail"
    )


def test_experiments_applied_to_entity_single(
    measurement_space_from_single_parameterized_experiment: MeasurementSpace,
) -> None:
    es = measurement_space_from_single_parameterized_experiment.compatibleEntitySpace()
    entity = sample_random_entity_from_space(es)
    assert (
        measurement_space_from_single_parameterized_experiment.numberExperimentsApplied(
            entity
        )
        == 0
    ), "Expected the test entity to have no experiments applied from the space"

    import numpy.random

    rng = numpy.random.default_rng()
    exp: Experiment = (
        measurement_space_from_single_parameterized_experiment.experiments[0]
    )

    #
    # Create a measurement result
    #
    values = [
        ObservedPropertyValue(value=rng.random(), property=op)
        for op in exp.observedProperties
    ]

    result = ValidMeasurementResult(
        entityIdentifier=entity.identifier, measurements=values
    )
    entity.add_measurement_result(result)

    assert (
        measurement_space_from_single_parameterized_experiment.numberExperimentsApplied(
            entity
        )
        == 1
    ), "Expected the test entity to have one experiments applied from the space"

    #
    # Create another measurement result for same experiment
    #
    values = []
    for op in exp.observedProperties:
        values.append(ObservedPropertyValue(value=rng.random(), property=op))

    result = ValidMeasurementResult(
        entityIdentifier=entity.identifier, measurements=values
    )
    entity.add_measurement_result(result)
    assert (
        measurement_space_from_single_parameterized_experiment.numberExperimentsApplied(
            entity
        )
        == 1
    ), (
        "Expected the test entity to report one experiment applied from the space, "
        "even when that experiment is applied multiple times"
    )


def test_experiments_applied_to_entity_multiple(
    measurement_space_from_multiple_parameterized_experiments: MeasurementSpace,
) -> None:

    es = (
        measurement_space_from_multiple_parameterized_experiments.compatibleEntitySpace()
    )
    entity = sample_random_entity_from_space(es)
    assert (
        measurement_space_from_multiple_parameterized_experiments.numberExperimentsApplied(
            entity
        )
        == 0
    ), "Expected the test entity to have no experiments applied"

    #
    # Create a measurement result for each experiment
    #
    import numpy as np

    rng = np.random.default_rng()
    for exp in measurement_space_from_multiple_parameterized_experiments.experiments:
        values = [
            ObservedPropertyValue(value=rng.random(), property=op)
            for op in exp.observedProperties
        ]

        result = ValidMeasurementResult(
            entityIdentifier=entity.identifier, measurements=values
        )
        entity.add_measurement_result(result)

    assert measurement_space_from_multiple_parameterized_experiments.numberExperimentsApplied(
        entity
    ) == len(
        measurement_space_from_multiple_parameterized_experiments.experiments
    ), "Expected the test entity to have each of the experiments in the space applied to it"

    #
    # Create another measurement result for each experiment
    #
    import numpy as np

    for exp in measurement_space_from_multiple_parameterized_experiments.experiments:
        values = []
        for op in exp.observedProperties:
            values.append(ObservedPropertyValue(value=rng.random(), property=op))

        result = ValidMeasurementResult(
            entityIdentifier=entity.identifier, measurements=values
        )
        entity.add_measurement_result(result)

    assert measurement_space_from_multiple_parameterized_experiments.numberExperimentsApplied(
        entity
    ) == len(
        measurement_space_from_multiple_parameterized_experiments.experiments
    ), "Expected numberExperimentsApplied to count each experiment once, regardless how many times it was applied"


def test_dependent_experiments_single(
    measurement_space_from_single_parameterized_experiment: MeasurementSpace,
) -> None:
    """Test
    - dependentExperimentsThatCanBeAppliedToEntity
    - dependentExperimentsThatCanBeAppliedAfterMeasurementRequest

    for a space containing one experiment which may/may not require others

    A particular aspect tested here is behaviour when the MeasurementSpace has
    a single experiment that requires another experiment"""

    es = measurement_space_from_single_parameterized_experiment.compatibleEntitySpace()
    entity = sample_random_entity_from_space(es)

    # We know one of the MeasurementSpaces passed contains an experiment
    # that requires entities to have the output of an experiment not in this space
    # For this entity no experiment in the space can be applied
    # For the other space, which have no dependent experiments, the answer is also zero
    # We have two tests to support to different assert error messages
    if measurement_space_from_single_parameterized_experiment.dependentExperiments:
        assert not measurement_space_from_single_parameterized_experiment.isConsistent
        assert (
            len(
                measurement_space_from_single_parameterized_experiment.dependentExperimentsThatCanBeAppliedToEntity(
                    entity=entity
                )
            )
            == 0
        ), "Expected the test entity to have no values the properties required by the dependent experiment in the space, and hence that the dependent experiment in the space could not be applied to the Entity"
    else:
        assert measurement_space_from_single_parameterized_experiment.isConsistent
        assert (
            len(
                measurement_space_from_single_parameterized_experiment.dependentExperimentsThatCanBeAppliedToEntity(
                    entity=entity
                )
            )
            == 0
        ), "There are no dependent experiments in the space so expect that none can be applied to entity"

    import numpy.random

    exp = measurement_space_from_single_parameterized_experiment.experiments[0]
    values = [
        ObservedPropertyValue(value=numpy.random.default_rng().random(), property=op)
        for op in exp.observedProperties
    ]

    #
    # Create a measurement result & request for testing dependentExperimentsThatCanBeAppliedAfterMeasurementRequest
    # and dependentExperimentsThatCanBeAppliedToEntity when it has the results
    #
    result = ValidMeasurementResult(
        entityIdentifier=entity.identifier, measurements=values
    )

    # The request automatically adds the results to the entity
    request = MeasurementRequest(
        operation_id="test",
        requestIndex=0,
        experimentReference=exp.reference,
        entities=[entity],
        measurements=(result,),
    )

    #
    # Test for dependentExperimentsThatCanBeAppliedAfterMeasurementRequest
    #
    if measurement_space_from_single_parameterized_experiment.dependentExperiments:
        assert (
            measurement_space_from_single_parameterized_experiment.dependentExperimentsThatCanBeAppliedAfterMeasurementRequest(
                request
            )
            == {}
        ), (
            "Expected that after adding the values required by MeasurementSpace, the dependent experiments that can be applied is still 0 "
            "as the values for the required properties were not added"
        )
    else:
        assert (
            measurement_space_from_single_parameterized_experiment.dependentExperimentsThatCanBeAppliedAfterMeasurementRequest(
                request
            )
            == {}
        ), "Expected that after adding the values required by MeasurementSpace the dependent experiment that can be applied is still 0 as there are none"

    if measurement_space_from_single_parameterized_experiment.dependentExperiments:
        #
        # Test: dependentExperimentsThatCanBeAppliedToEntity, when results of dependent experiment are present but not the required
        #
        # There should be no dependent experiments that can be applied, since we added the results for it
        assert not measurement_space_from_single_parameterized_experiment.dependentExperimentsThatCanBeAppliedToEntity(
            entity=entity
        ), "Expected dependentExperimentsThatCanBeAppliedToEntity to return no experiment since the required property is still missing"

        assert (
            len(
                measurement_space_from_single_parameterized_experiment.dependentExperimentsThatCanBeAppliedToEntity(
                    entity=entity, excludeApplied=False
                )
            )
            == 0
        ), (
            "Expect dependentExperimentsThatCanBeAppliedToEntity to return no experiment, even with excludeApplied=False,"
            " since the required property is still missing"
        )

        #
        # Test: dependentExperimentsThatCanBeAppliedToEntity, when results of dependent experiment are present AND required are present
        #
        for (
            exp
        ) in (
            measurement_space_from_single_parameterized_experiment.dependentExperiments
        ):
            for prop in exp.requiredObservedProperties:
                result = ValidMeasurementResult(
                    entityIdentifier=entity.identifier,
                    measurements=[ObservedPropertyValue(value=3, property=prop)],
                )

            request = MeasurementRequest(
                operation_id="test-with-experiment-applied",
                requestIndex=0,
                experimentReference=exp.reference,
                entities=[entity],
                measurements=(result,),
            )

            assert not measurement_space_from_single_parameterized_experiment.dependentExperimentsThatCanBeAppliedAfterMeasurementRequest(
                measurementRequest=request
            ), (
                "Expected dependentExperimentsThatCanBeAppliedAfterMeasurementRequest to return no experiment since "
                "results have already been added for the dependent experiment"
            )

        assert not measurement_space_from_single_parameterized_experiment.dependentExperimentsThatCanBeAppliedToEntity(
            entity=entity
        ), "Expected dependentExperimentsThatCanBeAppliedToEntity to return no experiment since excludeApplied=True"

        assert (
            len(
                measurement_space_from_single_parameterized_experiment.dependentExperimentsThatCanBeAppliedToEntity(
                    entity=entity, excludeApplied=False
                )
            )
            == 1
        ), "Expect dependentExperimentsThatCanBeAppliedToEntity to return an experiment since excludeApplied=False"


def test_dependent_experiments_multiple(
    measurement_space_from_multiple_parameterized_experiments: MeasurementSpace,
) -> None:
    """Test
    - dependentExperimentsThatCanBeAppliedToEntity
    - dependentExperimentsThatCanBeAppliedAfterMeasurementRequest"""

    assert measurement_space_from_multiple_parameterized_experiments.isConsistent

    es = (
        measurement_space_from_multiple_parameterized_experiments.compatibleEntitySpace()
    )
    entity = sample_random_entity_from_space(es)

    # Unlike the single case, this MeasurementSpace contains both
    # an independent experiment and an experiment that depends on it
    #
    assert (
        len(
            measurement_space_from_multiple_parameterized_experiments.dependentExperimentsThatCanBeAppliedToEntity(
                entity=entity
            )
        )
        == 0
    ), "Expected the test entity to have no values the properties required by the dependent experiment in the space, and hence that the dependent experiment in the space could not be applied to the Entity"

    # Check that there is actually more than one experiment
    assert (
        len(measurement_space_from_multiple_parameterized_experiments.experiments) > 1
    ), "Expected more than one experiment in this space"

    assert (
        len(
            measurement_space_from_multiple_parameterized_experiments.dependentExperiments
        )
        == 1
    ), "Expected exactly one dependent experiment in this space"

    #
    # Add a measurement for the experiment required by the dependent experiment
    #

    # Get a ref of the experiment required by the dependent experiment
    #
    # Note: The required experiment in the MS is parameterized, whereas the
    # dependent experiment references the base experiment
    # - this is handled by dependentExperimentsThatCanBeAppliedAfterMeasurementRequest
    dependent_exp: Experiment = (
        measurement_space_from_multiple_parameterized_experiments.dependentExperiments[
            0
        ]
    )
    required_ref = next(iter(dependent_exp.references_of_required_input_experiments))

    import numpy.random

    # Here we find the experiment in the MS that matches the requirements of the dependent experiment
    rng = numpy.random.default_rng()
    required_experiment = None
    for (
        r
    ) in measurement_space_from_multiple_parameterized_experiments.experimentReferences:
        if r.experimentIdentifier == required_ref.experimentIdentifier:
            required_experiment = measurement_space_from_multiple_parameterized_experiments.experimentForReference(
                r
            )
    assert required_experiment

    #
    # Generate random values for the observed properties
    #
    values = [
        ObservedPropertyValue(value=rng.random(), property=op)
        for op in required_experiment.observedProperties
    ]

    # Create a measurement result
    result = ValidMeasurementResult(
        entityIdentifier=entity.identifier, measurements=values
    )

    # NOTE: this adds the MeasurementResult to the Entity
    request = MeasurementRequest(
        operation_id="test",
        requestIndex=0,
        experimentReference=required_experiment.reference,
        entities=[entity],
        measurements=(result,),
    )

    #
    # Test for dependentExperimentsThatCanBeAppliedAfterMeasurementRequest
    #
    dependent_exps = measurement_space_from_multiple_parameterized_experiments.dependentExperimentsThatCanBeAppliedAfterMeasurementRequest(
        request
    )

    assert (
        len(dependent_exps) == 1
    ), "Expected that after measuring values required by the dependent Experiment, the dependent experiment becomes available to be applied"
    assert isinstance(dependent_exps, dict)

    #
    # Test: dependentExperimentsThatCanBeAppliedToEntity, when results of independent experiment are present
    #

    # There should be no available dependent experiments by default since it is measured
    assert (
        len(
            measurement_space_from_multiple_parameterized_experiments.dependentExperimentsThatCanBeAppliedToEntity(
                entity=entity
            )
        )
        == 1
    ), "Expected that after adding values required by the dependent experiment it should be available to apply"

    #
    # Test: dependentExperimentsThatCanBeAppliedToEntity, when the results of the dependent experiment are present
    #

    #
    # Add results for the dependent experiment
    #

    values = []
    for op in dependent_exp.observedProperties:
        values.append(ObservedPropertyValue(value=rng.random(), property=op))

    # Create a measurement result
    result = ValidMeasurementResult(
        entityIdentifier=entity.identifier, measurements=values
    )

    # NOTE: this adds the MeasurementResult to the Entity
    request = MeasurementRequest(
        operation_id="test-2",
        requestIndex=0,
        experimentReference=required_experiment.reference,
        entities=[entity],
        measurements=(result,),
    )

    assert (
        len(
            measurement_space_from_multiple_parameterized_experiments.dependentExperimentsThatCanBeAppliedToEntity(
                entity=entity, excludeApplied=True
            )
        )
        == 0
    ), "Expected that after applying dependent experiments to an Entity they are no longer returned by dependentExperimentsThatCanBeAppliedToEntity by default"

    assert (
        len(
            measurement_space_from_multiple_parameterized_experiments.dependentExperimentsThatCanBeAppliedToEntity(
                entity=entity, excludeApplied=False
            )
        )
        == 1
    ), (
        "Expected that after applying dependent experiments to an Entity, "
        "they will be still returned by dependentExperimentsThatCanBeAppliedToEntity if excludeApplied=False"
    )

    assert (
        len(
            measurement_space_from_multiple_parameterized_experiments.dependentExperimentsThatCanBeAppliedAfterMeasurementRequest(
                request
            )
        )
        == 1
    ), (
        "Expect dependentExperimentsThatCanBeAppliedAfterMeasurementRequest "
        "to return dependent experiment even if it has already been applied "
    )

    assert (
        measurement_space_from_multiple_parameterized_experiments.dependentExperimentsThatCanBeAppliedToEntity(
            entity=entity, excludeApplied=False
        )[
            0
        ]
        == dependent_exp
    ), (
        "Expected that after applying dependent experiments to an Entity, the correct dependent experiment "
        "is returned by dependentExperimentsThatCanBeAppliedToEntity if excludeApplied=False"
    )


def test_measurementspace_rich_print(
    measurement_space_from_single_parameterized_experiment: MeasurementSpace,
) -> None:

    from rich.console import Console

    Console().print(measurement_space_from_single_parameterized_experiment)
    print(measurement_space_from_single_parameterized_experiment)


def test_measurementspace_rich_print_multiple(
    measurement_space_from_multiple_parameterized_experiments: MeasurementSpace,
) -> None:

    from rich.console import Console

    Console().print(measurement_space_from_multiple_parameterized_experiments)
    print(measurement_space_from_multiple_parameterized_experiments)


def test_property_with_identifier_format_target(
    measurement_space_from_multiple_parameterized_experiments: MeasurementSpace,
) -> None:
    """Test propertyWithIdentifierInSpace with format='target'"""

    ms = measurement_space_from_multiple_parameterized_experiments

    # Get a valid target property identifier
    target_property = ms.targetProperties[0]
    target_id = target_property.identifier

    # Test with valid target identifier
    assert ms.propertyWithIdentifierInSpace(
        target_id, format="target"
    ), f"Expected target identifier '{target_id}' to be found with format='target'"

    # Test with observed identifier in target format (should fail)
    observed_property = ms.observedProperties[0]
    observed_id = observed_property.identifier
    assert not ms.propertyWithIdentifierInSpace(
        observed_id, format="target"
    ), f"Expected observed identifier '{observed_id}' to fail with format='target'"

    # Test with invalid identifier
    assert not ms.propertyWithIdentifierInSpace(
        "invalid_metric", format="target"
    ), "Expected invalid identifier to fail with format='target'"


def test_property_with_identifier_format_observed(
    measurement_space_from_multiple_parameterized_experiments: MeasurementSpace,
) -> None:
    """Test propertyWithIdentifierInSpace with format='observed'"""

    ms = measurement_space_from_multiple_parameterized_experiments

    # Get a valid observed property identifier
    observed_property = ms.observedProperties[0]
    observed_id = observed_property.identifier

    # Test with valid observed identifier
    assert ms.propertyWithIdentifierInSpace(
        observed_id, format="observed"
    ), f"Expected observed identifier '{observed_id}' to be found with format='observed'"

    # Test with target identifier in observed format (should fail unless it matches an observed)
    target_property = ms.targetProperties[0]
    target_id = target_property.identifier
    # This should fail because target_id is not the full observed property identifier
    assert not ms.propertyWithIdentifierInSpace(
        target_id, format="observed"
    ), f"Expected target identifier '{target_id}' to fail with format='observed'"

    # Test with invalid identifier
    assert not ms.propertyWithIdentifierInSpace(
        "invalid_metric", format="observed"
    ), "Expected invalid identifier to fail with format='observed'"

    # Test virtual property based on observed
    vp = VirtualObservedProperty(
        baseObservedProperty=observed_property,
        aggregationMethod=PropertyAggregationMethod(
            identifier=PropertyAggregationMethodEnum.mean
        ),
    )
    assert ms.propertyWithIdentifierInSpace(
        vp.identifier, format="observed"
    ), f"Expected virtual observed property '{vp.identifier}' to be found with format='observed'"
