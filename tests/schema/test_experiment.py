# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import re
from typing import Any

import pydantic
import pytest

from orchestrator.modules.actuators.registry import (
    ActuatorRegistry,
)
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import (
    Experiment,
    ParameterizedExperiment,
)
from orchestrator.schema.observed_property import (
    ObservedPropertyValue,
)
from orchestrator.schema.property import (
    AbstractProperty,
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
    MeasuredPropertyTypeEnum,
)
from orchestrator.schema.property_value import (
    ConstitutivePropertyValue,
    CustomBytes,
)
from orchestrator.schema.reference import (
    ExperimentReference,
    check_parameterization_validity,
)
from orchestrator.schema.result import ValidMeasurementResult
from orchestrator.utilities.support import get_experiment_input_values

# test experiment from concrete property identifiers
# Property Retrieval
# Validation
# test alternate routes in set_parameterized_identifier validator
#  - Test an invalid parameteriation
#


def test_property_values_from_entity_multiple_required() -> None:

    pass


def test_property_values_from_entity_missing_required_constitutive() -> None:

    pass


def experiment_equality_non_experiment(experiment: Experiment) -> None:
    "Utility function for use in tests"

    assert experiment != "string"
    assert experiment != experiment.model_dump()


def experiment_is_hashable(experiment: Experiment) -> None:
    "Utility function for use in tests"

    # Check the instance can be used as dict key
    # Hash-ability is based on actuator identifier
    d = {experiment: "value"}

    ## Copy the experiment and use it
    experimentCopy = experiment.model_copy(deep=True)
    assert d[experimentCopy] == "value"
    d[experimentCopy] = "3"
    assert len(d) == 1

    ## Copy the experiment changing the actuator id and use it - check the length of the dict is two
    test = experiment.model_copy(update={"actuatorIdentifier": "random"}, deep=True)
    assert test.actuatorIdentifier == "random"
    assert not d.get(test)
    d[test] = 4
    assert len(d) == 2


def test_parameterizable_experiment_equality_experiment_same_id_different_actuator(
    parameterizable_experiment: Experiment, global_registry: ActuatorRegistry
) -> None:

    cp = parameterizable_experiment.model_copy(
        update={"actuatorIdentifier": "randomstring"}
    )
    assert parameterizable_experiment.reference != cp.reference
    assert parameterizable_experiment != cp


def test_parameterized_experiment_equality_experiment_same_id_different_actuator(
    parameterized_experiment: ParameterizedExperiment, global_registry: ActuatorRegistry
) -> None:

    assert parameterized_experiment != parameterized_experiment.model_copy(
        update={"actuatorIdentifier": "randomstring"}
    )


def test_parameterizable_experiment_equality_non_experiment(
    parameterizable_experiment: Experiment, global_registry: ActuatorRegistry
) -> None:

    experiment_equality_non_experiment(parameterizable_experiment)


def test_parameterized_experiment_equality_non_experiment(
    parameterized_experiment: ParameterizedExperiment, global_registry: ActuatorRegistry
) -> None:

    experiment_equality_non_experiment(parameterized_experiment)


def test_parameterizable_experiment_is_hashable(
    parameterizable_experiment: Experiment,
) -> None:

    experiment_is_hashable(parameterizable_experiment)


def test_parameterized_experiment_is_hashable(
    parameterized_experiment: ParameterizedExperiment,
) -> None:

    experiment_is_hashable(parameterized_experiment)


def test_parameterized_experiment_base_equality_methods(
    parameterized_experiment: ParameterizedExperiment, global_registry: ActuatorRegistry
) -> None:

    base_exp = global_registry.experimentForReference(
        parameterized_experiment.reference
    )

    assert parameterized_experiment.has_same_base_as_experiment_reference(
        base_exp.reference
    )
    assert base_exp.has_same_base_as_experiment_reference(
        parameterized_experiment.reference
    )

    assert parameterized_experiment.has_same_base_as_experiment(base_exp)
    assert base_exp.has_same_base_as_experiment(parameterized_experiment)


def test_parameterizable_experiment_rich_print(
    parameterized_experiment: ParameterizedExperiment, global_registry: ActuatorRegistry
) -> None:

    from rich.console import Console

    # Requesting global_registry fixture to ensure the experiments are added to the registry for testing
    Console().print(parameterized_experiment)


def test_parameterized_experiment_rich_print(
    parameterized_experiment: ParameterizedExperiment, global_registry: ActuatorRegistry
) -> None:

    from rich.console import Console

    # Requesting global_registry fixture to ensure the experiments are added to the registry for testing
    Console().print(parameterized_experiment)


def test_experiment_observed_properties(
    experiment: Experiment,
    expected_observed_property_identifiers: list[str],
    target_property_list: list[str],
) -> None:
    """Test that the observed properties created by an experiment are as expected"""

    for op in experiment.observedProperties:
        assert op.identifier in expected_observed_property_identifiers

    for ident in expected_observed_property_identifiers:
        assert experiment.hasObservedPropertyWithIdentifier(ident)

    # Test Property retrieval

    #
    # via Observed Property (or observed property id)
    #

    assert experiment.hasObservedPropertyWithIdentifier(
        expected_observed_property_identifiers[0]
    )
    assert not experiment.hasObservedPropertyWithIdentifier("made_up_identifier")
    assert experiment.hasObservedProperty(experiment.observedProperties[0])

    #
    # via Target Property (or target property id)
    #

    op = experiment.observedPropertyForTargetIdentifier(target_property_list[0])
    assert op
    assert op.targetProperty.identifier == target_property_list[0]

    assert experiment.hasTargetPropertyWithIdentifier(target_property_list[0])
    assert experiment.hasTargetProperty(
        AbstractProperty(identifier=target_property_list[-1])
    )
    assert not experiment.hasTargetProperty(
        AbstractProperty(identifier="random_string")
    )

    #
    # Via virtual observed property id
    #

    assert experiment.virtualObservedPropertyFromIdentifier(
        f"{expected_observed_property_identifiers[0]}-mean"
    )
    assert experiment.virtualObservedPropertyFromIdentifier(
        f"{target_property_list[0]}-mean"
    )
    assert not experiment.virtualObservedPropertyFromIdentifier("randomstring-mean")

    with pytest.raises(
        ValueError,
        match=re.escape(
            "randomstring-not-a-virtual-property-id is not a valid virtual property identifier"
        ),
    ):
        experiment.virtualObservedPropertyFromIdentifier(
            "randomstring-not-a-virtual-property-id"
        )


@pytest.fixture
def experimentWithOptions(
    experimentRawNoOptional: dict[str, Any],
    optionalProperties: list[ConstitutivePropertyValue],
    defaultParameterization: list[ConstitutivePropertyValue],
) -> Experiment:

    return Experiment(
        optionalProperties=tuple(optionalProperties),
        defaultParameterization=tuple(defaultParameterization),
        **experimentRawNoOptional,
    )


def test_create_experiment_with_optional_params(
    experimentRawNoOptional: dict,
    optionalProperties: list[ConstitutiveProperty],
    defaultParameterization: list[ConstitutivePropertyValue],
) -> None:
    """Test we can create an experiment with optional parameters"""

    Experiment(
        optionalProperties=tuple(optionalProperties),
        defaultParameterization=tuple(defaultParameterization),
        **experimentRawNoOptional,
    )

    # Test creating optional properties without default parameterization fails

    with pytest.raises(pydantic.ValidationError):
        Experiment(
            optionalProperties=tuple(optionalProperties),
            **experimentRawNoOptional,
        )

    # Test specifying default parameterization without options fails
    with pytest.raises(pydantic.ValidationError):

        Experiment(
            defaultParameterization=tuple(defaultParameterization),
            **experimentRawNoOptional,
        )

    # Test default parameterization with missing/incorrect properties fails
    with pytest.raises(pydantic.ValidationError):

        Experiment(
            optionalProperties=tuple(optionalProperties),
            defaultParameterization=tuple(defaultParameterization[:-2]),
            **experimentRawNoOptional,
        )

    # Test default parameterization with duplicate values for a property fails
    with pytest.raises(pydantic.ValidationError):
        Experiment(
            optionalProperties=tuple(optionalProperties),
            defaultParameterization=tuple(
                defaultParameterization + defaultParameterization[:1]
            ),
            **experimentRawNoOptional,
        )

    # Test adding optional properties with same id fails
    with pytest.raises(pydantic.ValidationError):
        Experiment(
            optionalProperties=tuple(optionalProperties + optionalProperties[:1]),
            defaultParameterization=tuple(defaultParameterization),
            **experimentRawNoOptional,
        )

    # Test default parameterization with value not in default options domain fails
    import copy

    incorrectParameterization = copy.deepcopy(defaultParameterization)
    incorrectParameterization[0].value = "D"
    with pytest.raises(pydantic.ValidationError):
        Experiment(
            optionalProperties=tuple(optionalProperties),
            defaultParameterization=tuple(incorrectParameterization),
            **experimentRawNoOptional,
        )

    # Test adding optional properties with same name as required property fails
    baseExp = Experiment.model_validate(experimentRawNoOptional)
    print(baseExp.requiredProperties)
    with pytest.raises(pydantic.ValidationError):
        Experiment(
            optionalProperties=(*optionalProperties, baseExp.requiredProperties[0]),
            defaultParameterization=(
                *defaultParameterization,
                ConstitutivePropertyValue(
                    value="X", property=baseExp.requiredProperties[0].descriptor()
                ),
            ),
            **experimentRawNoOptional,
        )


def test_get_experiment_optional_parameter_value(
    experimentWithOptions: Experiment,
    defaultParameterization: list[ConstitutivePropertyValue],
    customParameterization: list[ConstitutivePropertyValue],
) -> None:

    #  Test valueForProperty returns the default parameterized values
    for pv in defaultParameterization:
        assert (
            experimentWithOptions.valueForOptionalProperty(pv.property.identifier).value
            == pv.value
        )

    # Test asking for non-optional property raises ValueError
    with pytest.raises(
        ValueError,
        match=f"Experiment {experimentWithOptions.identifier} has no optional property RandomString",
    ):
        experimentWithOptions.valueForOptionalProperty("RandomString")

    #  Test valueForProperty returns the correct parameterized values we set
    p = ParameterizedExperiment(
        **experimentWithOptions.model_dump(), parameterization=customParameterization
    )
    for pv in customParameterization:
        assert p.valueForOptionalProperty(pv.property.identifier).value == pv.value

    # Test we can provide only a subset of values in custom parameterization
    # ParameterizedExperiment should return the default value for those not specified
    p = ParameterizedExperiment(
        **experimentWithOptions.model_dump(),
        parameterization=customParameterization[1:],
    )

    assert (
        p.valueForOptionalProperty(defaultParameterization[0].property.identifier).value
        == defaultParameterization[0].value
    )
    for pv in customParameterization[1:]:
        assert p.valueForOptionalProperty(pv.property.identifier).value == pv.value

    # Test asking for non-optional property raises ValueError
    with pytest.raises(ValueError, match="No optional property called RandomString"):
        p.valueForOptionalProperty("RandomString")


def test_is_valid_experiment_parameterization(
    experimentWithOptions: Experiment,
    defaultParameterization: list[ConstitutivePropertyValue],
) -> None:

    assert experimentWithOptions.isValidParameterization(defaultParameterization)

    values = [
        ConstitutivePropertyValue(
            value="D", property=ConstitutivePropertyDescriptor(identifier="test_opt1")
        ),
        ConstitutivePropertyValue(
            value=-3, property=ConstitutivePropertyDescriptor(identifier="test_opt2")
        ),
        ConstitutivePropertyValue(
            value=5.78, property=ConstitutivePropertyDescriptor(identifier="test_opt3")
        ),
    ]

    # THis value is not in the domain of test_opt1 optional property so it should cause failure

    assert not experimentWithOptions.isValidParameterization(values)

    # Remove a parameterization
    assert not experimentWithOptions.isValidParameterization(values[:-1])

    # Rename a parameter e.g. due to misspelling
    values[0] = ConstitutivePropertyValue(
        value="C", property=ConstitutivePropertyDescriptor(identifier="test_op1")
    )

    assert not experimentWithOptions.isValidParameterization(values)


def test_custom_experiment_parameterization_is_valid(
    experimentWithOptions: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:

    assert experimentWithOptions.isValidParameterization(customParameterization)


def test_experiment_rich_print(experiment: Experiment) -> None:

    from rich.console import Console

    Console().print(experiment)


def test_measurement_types(experiment: Experiment) -> None:
    """Test that created experiments have the correct measurements types"""

    assert (
        experiment.observedProperties[0].propertyType
        == MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE
    )


def test_retrieve_parameterizable_experiment(
    global_registry: ActuatorRegistry, mock_parameterizable_experiment: Experiment
) -> None:

    assert global_registry.experimentForReference(
        mock_parameterizable_experiment.reference
    )


def test_parameterized_experiment_serialize_deserialize(
    global_registry: ActuatorRegistry,
    mock_parameterizable_experiment: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:
    ref = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=customParameterization,
    )

    ser = ref.model_dump()
    print(ser)
    ExperimentReference.model_validate(ser)


def test_create_parameterized_experiment(
    experimentWithOptions: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
    global_registry: ActuatorRegistry,
) -> None:

    import copy

    p = ParameterizedExperiment(
        **experimentWithOptions.model_dump(), parameterization=customParameterization
    )

    param_copy = ParameterizedExperiment(
        **experimentWithOptions.model_dump(), parameterization=customParameterization
    )

    assert experimentWithOptions == experimentWithOptions
    assert p != experimentWithOptions
    assert p == p
    assert p == param_copy
    assert p.parameterizedIdentifier == param_copy.parameterizedIdentifier
    assert p.identifier == param_copy.identifier

    # To test the reference we have to add the experiment to the registry
    global_registry.catalogForActuatorIdentifier(
        experimentWithOptions.actuatorIdentifier
    ).addExperiment(experimentWithOptions)

    assert (
        p.reference.parameterizedExperimentIdentifier
        == param_copy.reference.parameterizedExperimentIdentifier
    )

    # Check the parameterized experiments id is as expected.
    # We construct it here as it's expected to be done
    pstr = "-".join(
        [f"{v.property.identifier}.{v.value}" for v in customParameterization]
    )
    assert p.parameterizedIdentifier == f"{experimentWithOptions.identifier}-{pstr}"

    # Test parameterization with duplicate property fails
    with pytest.raises(
        ValueError,
        match="The parameterization contains multiple values for same property",
    ):
        ParameterizedExperiment(
            **experimentWithOptions.model_dump(),
            parameterization=customParameterization + customParameterization[:1],
        )

    # Test parameterization with incorrectly named property fails
    incorrectParameterization = copy.deepcopy(customParameterization)
    pv = incorrectParameterization[0]
    incorrectParameterization[0] = ConstitutivePropertyValue(
        value=pv.value, property=ConstitutivePropertyDescriptor(identifier="tes_opt1")
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"parameterized properties not in optionalProperties list. "
            f"Missing: {[incorrectParameterization[0].property.identifier]}"
        ),
    ):
        ParameterizedExperiment(
            **experimentWithOptions.model_dump(),
            parameterization=incorrectParameterization,
        )

    # Test parameterization with same values as default fails
    incorrectParameterization = copy.deepcopy(customParameterization)
    incorrectParameterization[0] = experimentWithOptions.defaultParameterization[0]
    with pytest.raises(
        ValueError,
        match=f"Default value {experimentWithOptions.defaultParameterization[0].value} "
        f"for property {incorrectParameterization[0].property.identifier} is the same as the custom value, "
        f"{incorrectParameterization[0].value}",
    ):
        ParameterizedExperiment(
            **experimentWithOptions.model_dump(),
            parameterization=incorrectParameterization,
        )


# Missing:
# - Test a ValueError is raised if the experiment has a required observed property and
#   the Entity doesn't have a value for it
# - Testing when there are multiple observed property values for the required observed property
# - Testing a Entity that doesn't have the correct ConstitutiveProperties
def test_experiment_property_values_from_entity(
    entity_for_parameterized_experiment: tuple[Entity, ParameterizedExperiment],
) -> None:
    """Test Experiment.propertyValuesFromEntity works"""

    entity: Entity = entity_for_parameterized_experiment[0]
    exp: ParameterizedExperiment = entity_for_parameterized_experiment[1]

    # Get the input property values for the experiment from the entity - including required and optional
    params = exp.propertyValuesFromEntity(entity)

    # Construct "params" from source data for comparison to propertyValuesFromEntity
    expectedValues = {
        c.property.identifier: c.value for c in exp.defaultParameterization
    }
    expectedValues.update(
        {c.property.identifier: c.value for c in exp.parameterization}
    )
    expectedValues.update(
        {c.property.identifier: c.value for c in entity.constitutive_property_values}
    )

    def reduction(
        values: list[ObservedPropertyValue],
    ) -> (
        int
        | float
        | list
        | str
        | CustomBytes
        | None
        | list[int | float | list | str | CustomBytes | None]
    ):

        return values[0].value if len(values) == 1 else [v.value for v in values]

    expectedValues.update(
        {
            c.identifier: reduction(
                entity.valuesForObservedPropertyIdentifier(c.identifier)
            )
            for c in exp.requiredObservedProperties
        }
    )

    assert set(expectedValues.keys()) == set(params.keys())
    for k in expectedValues:
        assert (
            expectedValues[k] == params[k]
        ), f"Expected value for {k}: {expectedValues[k]} does not match returned value {params[k]}"

    other_values = get_experiment_input_values(entity=entity, experiment=exp)
    assert expectedValues == other_values

    if len(exp.requiredObservedProperties) > 0:
        # Test behaviour if there are multiple measurements for the observed property
        # Check assumption that there is only 1 measurement result
        assert entity.measurement_results
        assert len(entity.measurement_results) == 1
        # We expect 3 was set in the fixture
        assert entity.measurement_results[0].measurements[0].value == 3

        # Add a new value for same property
        pv = (
            entity.measurement_results[0]
            .measurements[0]
            .model_copy(update={"value": 4})
        )
        entity.add_measurement_result(
            ValidMeasurementResult(
                measurements=[pv], entityIdentifier=entity.identifier
            )
        )
        params = exp.propertyValuesFromEntity(entity)
        assert len(params[pv.property.identifier]) == 2

    ## Test some error cases
    if len(exp.requiredObservedProperties) > 0:
        # Test that if the entity doesn't have the required observed property an exception is raised
        entity_copy = Entity(
            identifier=entity.identifier,
            constitutive_property_values=entity.constitutive_property_values,
        )
        with pytest.raises(
            ValueError,
            match=(
                rf"Entity {re.escape(str(entity.identifier))} \(.*\) has no value for required observed property "
                "test_parameterizable_experiment-measurable_one"
            ),
        ):
            exp.propertyValuesFromEntity(entity_copy)

    if len(exp.requiredConstitutiveProperties) > 0:
        # Test that if the entity doesn't have a required constitutive property an exception is raised
        entity_copy = Entity(
            identifier=entity.identifier,
            constitutive_property_values=entity.constitutive_property_values[:1],
            measurement_results=entity.measurement_results,
        )
        with pytest.raises(
            ValueError,
            match=rf"Entity {re.escape(str(entity.identifier))} \(.*\) has no value for "
            "required constitutive property",
        ):
            exp.propertyValuesFromEntity(entity_copy)


def test_parameterized_experiment_reference_validation_detects_invalid_cases() -> None:

    # Test creating a parameterized reference for an experiment that doesn't exist
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Failed validating parameterization. Cannot find experiment test from actuator mock in catalog"
        ),
    ):
        ExperimentReference(
            experimentIdentifier="test",
            actuatorIdentifier="mock",
            parameterization=[
                ConstitutivePropertyValue(
                    value=3, property=ConstitutivePropertyDescriptor(identifier="test")
                )
            ],
        ).validate_parameterization()

    # Test trying to parameterize a non-parameterizable experiment
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Experiment reference mock.test-experiment-test.3 specifies custom parameterization "
            "but the referenced experiment has no parameterizable properties."
        ),
    ):
        ExperimentReference(
            experimentIdentifier="test-experiment",
            actuatorIdentifier="mock",
            parameterization=[
                ConstitutivePropertyValue(
                    value=3, property=ConstitutivePropertyDescriptor(identifier="test")
                )
            ],
        ).validate_parameterization()


def test_validate_parameterization_function_with_none_values(
    optionalProperties: list[ConstitutiveProperty],
    defaultParameterization: list[ConstitutivePropertyValue],
    customParameterization: list[ConstitutivePropertyValue],
) -> None:

    # Initial check it works as intended
    check_parameterization_validity(
        parameterizableProperties=optionalProperties,
        defaultParameterization=defaultParameterization,
        customParameterization=customParameterization,
    )

    with pytest.raises(
        ValueError,
        match="Passed None for parameterizableProperties to check_parameterization_validity",
    ):
        # None parameterization properties should raise ValueError
        check_parameterization_validity(
            parameterizableProperties=None,  # type: ignore[assignment]
            defaultParameterization=defaultParameterization,
            customParameterization=customParameterization,
        )

    with pytest.raises(
        ValueError,
        match="Passed None for customParameterization to check_parameterization_validity",
    ):
        # None parameterization properties should raise ValueError
        check_parameterization_validity(
            parameterizableProperties=optionalProperties,
            defaultParameterization=defaultParameterization,
            customParameterization=None,  # type: ignore[assignment]
        )

    # None default parametrization is allowed
    check_parameterization_validity(
        parameterizableProperties=optionalProperties,
        defaultParameterization=None,
        customParameterization=customParameterization,
    )


# Required so we can test fields are readonly without causing pycharm to flag the test assignments as errors
def assert_field_is_readonly(
    instance: Any, field: str, value: Any  # noqa: ANN401
) -> None:
    with pytest.raises(pydantic.ValidationError):
        setattr(instance, field, value)


def test_parameterized_experiment_fields_immutable(
    mock_parameterizable_experiment: Experiment,
) -> None:
    """Test we cannot change the values of requiredProperties, optionalProperties or defaultParameterization

    Either by reassigning or modifying the container via a reference to the field
    """

    test_property_value = ConstitutivePropertyValue(
        value="C", property=ConstitutivePropertyDescriptor(identifier="test_opt1")
    )

    # default parameterization

    assert_field_is_readonly(
        mock_parameterizable_experiment, field="defaultParameterization", value=[]
    )

    assert (
        test_property_value
        != mock_parameterizable_experiment.defaultParameterization[0]
    )

    defaults = mock_parameterizable_experiment.defaultParameterization
    with pytest.raises(TypeError):
        # Suppress pycharm error as, if there is no bug, it can detect this is an error
        defaults[0] = test_property_value  # type: ignore[assignment]
    assert (
        mock_parameterizable_experiment.defaultParameterization[0]
        != test_property_value
    )

    # optional properties

    assert_field_is_readonly(
        mock_parameterizable_experiment, field="optionalProperties", value=[]
    )

    test_property = ConstitutiveProperty(identifier="test_opt_fake")

    assert test_property != mock_parameterizable_experiment.optionalProperties[0]

    optional_properties = mock_parameterizable_experiment.optionalProperties
    with pytest.raises(TypeError):
        # Suppress pycharm error as, if there is no bug, it can detect this is an error
        optional_properties[0] = test_property  # type: ignore[assignment]
    assert mock_parameterizable_experiment.optionalProperties[0] != test_property

    # required properties
    # the test experiment has no required properties

    assert_field_is_readonly(
        mock_parameterizable_experiment, field="requiredProperties", value=[]
    )

    assert len(mock_parameterizable_experiment.requiredProperties) == 3

    required_properties = mock_parameterizable_experiment.requiredProperties
    with pytest.raises(AttributeError):
        # Suppress pycharm error as, if there is no bug, it can detect this is an error
        required_properties.append(test_property)  # type: ignore[assignment]
    assert len(mock_parameterizable_experiment.requiredProperties) == 3


def test_cannot_set_parameterized_experiment_identifier_for_experiment(
    parameterizable_experiment: Experiment,
) -> None:
    # Create a new value for one of the properties
    test_property_value = ConstitutivePropertyValue(
        value="C", property=ConstitutivePropertyDescriptor(identifier="test_opt1")
    )
    with pytest.raises(pydantic.ValidationError):
        ParameterizedExperiment(
            identifier=parameterizable_experiment.identifier,
            actuatorIdentifier=parameterizable_experiment.actuatorIdentifier,
            parameterization=[test_property_value],
            parameterizedIdentifier="some-identifier",
        )


def test_experiment_provides_requirements(
    mock_parameterizable_experiment_with_required_observed: Experiment,
    mock_parameterizable_experiment: Experiment,
    mock_parameterizable_experiment_no_required: Experiment,
) -> None:
    """Test the method experimentProvidesRequirements"""

    # The "with_required_observed" experiment is constructed to require the mock_parameterizable_experiment
    assert mock_parameterizable_experiment_with_required_observed.experimentProvidesRequirements(
        mock_parameterizable_experiment
    )
    # it doesn't require the "no_required" experiment
    assert not mock_parameterizable_experiment_with_required_observed.experimentProvidesRequirements(
        mock_parameterizable_experiment_no_required
    )

    # try with an experiment that doesn't have requirements - always returns False
    assert (
        not mock_parameterizable_experiment_no_required.experimentProvidesRequirements(
            mock_parameterizable_experiment
        )
    )


@pytest.fixture(scope="module")
def nevergrad_opt_3d_test_func_experiment() -> Experiment:
    # Define required constitutive properties (x0, x1, x2, all continuous)
    required_props = [
        ConstitutiveProperty(
            identifier="x0",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
            ),
        ),
        ConstitutiveProperty(
            identifier="x1",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
            ),
        ),
        ConstitutiveProperty(
            identifier="x2",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
            ),
        ),
    ]
    # Optional property: name (categorical)
    optional_props = (
        ConstitutiveProperty(
            identifier="name",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
                values=["rosenbrock", "griewank", "sphere"],
            ),
        ),
    )
    default_param = (
        ConstitutivePropertyValue(
            value="rosenbrock",
            property=ConstitutivePropertyDescriptor(identifier="name"),
        ),
    )
    return Experiment(
        actuatorIdentifier="custom_experiments",
        identifier="nevergrad_opt_3d_test_func",
        targetProperties=[],
        requiredProperties=tuple(required_props),
        optionalProperties=optional_props,
        defaultParameterization=default_param,
    )


def entity_with_props(props: list[ConstitutivePropertyValue]) -> Entity:
    return Entity(constitutive_property_values=tuple(props))


def test_validate_entity_required_only(
    nevergrad_opt_3d_test_func_experiment: Experiment,
) -> None:
    props = [
        ConstitutivePropertyValue(
            value=0.5, property=ConstitutivePropertyDescriptor(identifier="x0")
        ),
        ConstitutivePropertyValue(
            value=1.5, property=ConstitutivePropertyDescriptor(identifier="x1")
        ),
        ConstitutivePropertyValue(
            value=2.5, property=ConstitutivePropertyDescriptor(identifier="x2")
        ),
    ]
    entity = entity_with_props(props)
    assert nevergrad_opt_3d_test_func_experiment.validate_entity(entity) is True


def test_validate_entity_with_optional_valid(
    nevergrad_opt_3d_test_func_experiment: Experiment,
) -> None:
    props = [
        ConstitutivePropertyValue(
            value=0.5, property=ConstitutivePropertyDescriptor(identifier="x0")
        ),
        ConstitutivePropertyValue(
            value=1.5, property=ConstitutivePropertyDescriptor(identifier="x1")
        ),
        ConstitutivePropertyValue(
            value=2.5, property=ConstitutivePropertyDescriptor(identifier="x2")
        ),
        ConstitutivePropertyValue(
            value="sphere", property=ConstitutivePropertyDescriptor(identifier="name")
        ),
    ]
    entity = entity_with_props(props)
    assert nevergrad_opt_3d_test_func_experiment.validate_entity(entity) is True


def test_validate_entity_with_optional_invalid(
    nevergrad_opt_3d_test_func_experiment: Experiment,
) -> None:
    props = [
        ConstitutivePropertyValue(
            value=0.5, property=ConstitutivePropertyDescriptor(identifier="x0")
        ),
        ConstitutivePropertyValue(
            value=1.5, property=ConstitutivePropertyDescriptor(identifier="x1")
        ),
        ConstitutivePropertyValue(
            value=2.5, property=ConstitutivePropertyDescriptor(identifier="x2")
        ),
        ConstitutivePropertyValue(
            value="foobar", property=ConstitutivePropertyDescriptor(identifier="name")
        ),
    ]
    entity = entity_with_props(props)
    assert nevergrad_opt_3d_test_func_experiment.validate_entity(entity) is False


def test_validate_entity_missing_required(
    nevergrad_opt_3d_test_func_experiment: Experiment,
) -> None:
    # missing x2
    props = [
        ConstitutivePropertyValue(
            value=0.5, property=ConstitutivePropertyDescriptor(identifier="x0")
        ),
        ConstitutivePropertyValue(
            value=1.5, property=ConstitutivePropertyDescriptor(identifier="x1")
        ),
    ]
    entity = entity_with_props(props)
    assert nevergrad_opt_3d_test_func_experiment.validate_entity(entity) is False


def test_validate_entity_missing_required_with_optional_valid(
    nevergrad_opt_3d_test_func_experiment: Experiment,
) -> None:
    # missing x2 but valid name
    props = [
        ConstitutivePropertyValue(
            value=0.5, property=ConstitutivePropertyDescriptor(identifier="x0")
        ),
        ConstitutivePropertyValue(
            value=1.5, property=ConstitutivePropertyDescriptor(identifier="x1")
        ),
        ConstitutivePropertyValue(
            value="griewank", property=ConstitutivePropertyDescriptor(identifier="name")
        ),
    ]
    entity = entity_with_props(props)
    assert nevergrad_opt_3d_test_func_experiment.validate_entity(entity) is False


def test_validate_entity_additional_property_strict_optional_false(
    nevergrad_opt_3d_test_func_experiment: Experiment,
) -> None:
    props = [
        ConstitutivePropertyValue(
            value=0.5, property=ConstitutivePropertyDescriptor(identifier="x0")
        ),
        ConstitutivePropertyValue(
            value=1.5, property=ConstitutivePropertyDescriptor(identifier="x1")
        ),
        ConstitutivePropertyValue(
            value=2.5, property=ConstitutivePropertyDescriptor(identifier="x2")
        ),
        ConstitutivePropertyValue(
            value=10, property=ConstitutivePropertyDescriptor(identifier="test")
        ),
    ]
    entity = entity_with_props(props)
    # Default: strict_optional=False, extra property is fine
    assert nevergrad_opt_3d_test_func_experiment.validate_entity(entity) is True


def test_validate_entity_additional_property_strict_optional_true(
    nevergrad_opt_3d_test_func_experiment: Experiment,
) -> None:
    props = [
        ConstitutivePropertyValue(
            value=0.5, property=ConstitutivePropertyDescriptor(identifier="x0")
        ),
        ConstitutivePropertyValue(
            value=1.5, property=ConstitutivePropertyDescriptor(identifier="x1")
        ),
        ConstitutivePropertyValue(
            value=2.5, property=ConstitutivePropertyDescriptor(identifier="x2")
        ),
        ConstitutivePropertyValue(
            value=10, property=ConstitutivePropertyDescriptor(identifier="test")
        ),
    ]
    entity = entity_with_props(props)
    assert (
        nevergrad_opt_3d_test_func_experiment.validate_entity(
            entity, disallow_extra_properties=True
        )
        is False
    )
