# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import typing
from typing import Any

import pytest

from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.property import (
    AbstractPropertyDescriptor,
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
)
from orchestrator.schema.property_value import ConstitutivePropertyValue
from orchestrator.schema.reference import ExperimentReference


@pytest.fixture
def experiment_identifier() -> str:
    return "pytest-mock-experiment"


@pytest.fixture
def actuator_identifier() -> str:
    return "mock"


@pytest.fixture
def experiment_reference(
    experiment_identifier: typing.AnyStr, actuator_identifier: typing.AnyStr
) -> ExperimentReference:
    return ExperimentReference(
        experimentIdentifier=experiment_identifier,
        actuatorIdentifier=actuator_identifier,
    )


@pytest.fixture
def experiment(
    experiment_identifier: str,
    actuator_identifier: str,
    abstract_properties: list[AbstractPropertyDescriptor],
    requiredProperties: list[ConstitutiveProperty],
) -> Experiment:
    return Experiment(
        identifier=experiment_identifier,
        metadata={"maintainer": "unknown@somewhere.com"},
        requiredProperties=tuple(requiredProperties),
        targetProperties=abstract_properties,
        actuatorIdentifier=actuator_identifier,
    )


@pytest.fixture
def expected_observed_property_identifiers(
    target_property_list: list[str],
    experiment_identifier: str,
) -> list[str]:
    return [f"{experiment_identifier}-{t}" for t in target_property_list]


@pytest.fixture(scope="module")
def optionalProperties() -> list[ConstitutiveProperty]:

    return [
        ConstitutiveProperty(
            identifier="test_opt1",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
                values=["A", "B", "C"],
            ),
        ),
        ConstitutiveProperty(
            identifier="test_opt2",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
                domainRange=[-5, 10],
                interval=2,
            ),
        ),
        ConstitutiveProperty(
            identifier="test_opt3",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
                domainRange=[0, 100],
            ),
        ),
    ]


@pytest.fixture(scope="module")
def defaultParameterization(
    optionalProperties: list[ConstitutivePropertyValue],
) -> list[ConstitutivePropertyValue]:
    return [
        ConstitutivePropertyValue(
            value="B", property=ConstitutivePropertyDescriptor(identifier="test_opt1")
        ),
        ConstitutivePropertyValue(
            value=-3, property=ConstitutivePropertyDescriptor(identifier="test_opt2")
        ),
        ConstitutivePropertyValue(
            value=5.78, property=ConstitutivePropertyDescriptor(identifier="test_opt3")
        ),
    ]


@pytest.fixture
def experimentRawNoOptional(experiment: Experiment) -> dict[str, Any]:  # noqa: ANN401
    """Returns an experiment model as dict without optionalProperties or defaultParameterization"""

    d = experiment.model_dump()
    d.pop("optionalProperties")
    d.pop("defaultParameterization")

    return d


@pytest.fixture(scope="module")
def customParameterization(
    optionalProperties: list[ConstitutiveProperty],
) -> list[ConstitutivePropertyValue]:
    return [
        ConstitutivePropertyValue(
            value="C", property=ConstitutivePropertyDescriptor(identifier="test_opt1")
        ),
        ConstitutivePropertyValue(
            value=-1, property=ConstitutivePropertyDescriptor(identifier="test_opt2")
        ),
        ConstitutivePropertyValue(
            value=6.78, property=ConstitutivePropertyDescriptor(identifier="test_opt3")
        ),
    ]


@pytest.fixture(scope="module")
def mock_parameterizable_experiment(
    requiredProperties: list[ConstitutiveProperty],
    optionalProperties: list[ConstitutiveProperty],
    defaultParameterization: list[ConstitutivePropertyValue],
) -> Experiment:
    """A parameterizable experiment that has required and optional properties"""

    return Experiment(
        actuatorIdentifier="mock",
        identifier="test_parameterizable_experiment",
        requiredProperties=tuple(requiredProperties),
        optionalProperties=tuple(optionalProperties),
        defaultParameterization=tuple(defaultParameterization),
        targetProperties=[AbstractPropertyDescriptor(identifier="measurable_one")],
        metadata={"description": "A mock experiment for testing"},
    )


@pytest.fixture(scope="module")
def mock_parameterizable_experiment_no_required(
    optionalProperties: list[ConstitutiveProperty],
    defaultParameterization: list[ConstitutivePropertyValue],
) -> Experiment:
    """A parameterizable experiment that has no required properties"""

    return Experiment(
        actuatorIdentifier="mock",
        identifier="test_parameterizable_experiment_two",
        optionalProperties=tuple(optionalProperties),
        defaultParameterization=tuple(defaultParameterization),
        targetProperties=[AbstractPropertyDescriptor(identifier="measurable_two")],
        metadata={"description": "A mock experiment for testing"},
    )


@pytest.fixture(scope="module")
def mock_parameterizable_experiment_with_required_observed(
    requiredProperties: list[ConstitutivePropertyValue],
    optionalProperties: list[ConstitutivePropertyValue],
    defaultParameterization: list[ConstitutivePropertyValue],
    mock_parameterizable_experiment: Experiment,
) -> Experiment:
    """A parameterizable experiment that has a required observed property"""

    # We add the observed property of the standard mock experiment
    op = mock_parameterizable_experiment.observedProperties[0]

    # leave out description so we have one test experiment without it
    return Experiment(
        actuatorIdentifier="mock",
        identifier="test_parameterizable_experiment_three",
        requiredProperties=(*requiredProperties, op),
        optionalProperties=tuple(optionalProperties),
        defaultParameterization=tuple(defaultParameterization),
        targetProperties=[AbstractPropertyDescriptor(identifier="measurable_three")],
    )


@pytest.fixture(scope="module")
def parameterizable_experiments(
    mock_parameterizable_experiment_no_required: Experiment,
    mock_parameterizable_experiment: Experiment,
    mock_parameterizable_experiment_with_required_observed: Experiment,
) -> list[Experiment]:
    """Returns a set of parameterizable experiments"""

    return [
        mock_parameterizable_experiment,
        mock_parameterizable_experiment_no_required,
        mock_parameterizable_experiment_with_required_observed,
    ]


@pytest.fixture(scope="module")
def parameterized_experiments(
    parameterizable_experiments: list[Experiment],
    customParameterization: list[ConstitutivePropertyValue],
) -> list[ParameterizedExperiment]:
    """Returns a set of parameterized experiments"""

    # Note: We deliberately leave out the last custom parameterization
    # to enable tests where that parameter is made part of an entity space
    return [
        ParameterizedExperiment(
            parameterization=customParameterization[:-1],
            **parameterizable_experiments[0].model_dump(),
        ),
        ParameterizedExperiment(
            parameterization=customParameterization[:-1],
            **parameterizable_experiments[1].model_dump(),
        ),
        # A different parameterization of the same experiment
        ParameterizedExperiment(
            parameterization=customParameterization[:-1],
            **parameterizable_experiments[2].model_dump(),
        ),
    ]


@pytest.fixture(scope="module")
def parameterized_references(
    parameterized_experiments: list[ParameterizedExperiment],
    global_registry: ActuatorRegistry,
) -> list[ExperimentReference]:
    return [e.reference for e in parameterized_experiments]


@pytest.fixture(params=["standard", "no_required", "with_required_observed"])
def parameterizable_experiment(
    mock_parameterizable_experiment: Experiment,
    mock_parameterizable_experiment_no_required: Experiment,
    mock_parameterizable_experiment_with_required_observed: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
    request: pytest.FixtureRequest,
) -> Experiment:
    """Use to obtain a single Experiment instance with optional properties"""

    exp = None
    if request.param == "standard":
        exp = mock_parameterizable_experiment
    elif request.param == "no_required":
        exp = mock_parameterizable_experiment_no_required
    elif request.param == "with_required_observed":
        exp = mock_parameterizable_experiment_with_required_observed

    return exp


@pytest.fixture
def parameterized_experiment(
    parameterizable_experiment: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
) -> ParameterizedExperiment:
    """Use to obtain a single ParameterizedExperiment instance"""

    return ParameterizedExperiment(
        parameterization=customParameterization[:-1],
        **parameterizable_experiment.model_dump(),
    )
