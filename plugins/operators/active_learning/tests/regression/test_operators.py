# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Tests for the PKH operator registration."""

from collections.abc import Callable

import pydantic
import pytest
from active_learning.regression._shared import (
    _outer_operation_output,
    _random_walk_parameters,
)
from active_learning.regression.pkh.operator import pkh
from active_learning.regression.pkh.parameters import (
    PKHOperatorParameters,
    PKHParameters,
)
from active_learning.regression.pkh.sampler import PKHSampleSelector

from ado.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
)
from ado.core.operation.operation import OperationOutput
from ado.core.operation.resource import OperationResource
from ado.modules.operators.collections import characterize
from ado.modules.operators.randomwalk import (
    CustomSamplerConfiguration,
    FilterModeEnum,
    SamplerModuleConf,
)


@pytest.mark.parametrize(
    ("name", "function", "configuration_model"),
    [
        ("pkh", pkh, PKHOperatorParameters),
    ],
)
def test_import_registers_operator_metadata(
    name: str,
    function: Callable[..., OperationOutput],
    configuration_model: type[PKHOperatorParameters],
) -> None:
    """Importing each strategy's operator module registers its own metadata."""

    metadata = characterize.operators[name]

    assert metadata.name == name
    assert metadata.version == "0.1.0"
    assert metadata.function is function
    assert metadata.configuration_model is configuration_model
    assert metadata.example_configuration == configuration_model.example_configuration()


@pytest.mark.parametrize(
    ("parameters_class", "sampler_parameters_class"),
    [
        (PKHOperatorParameters, PKHParameters),
    ],
    ids=["pkh"],
)
def test_operator_parameter_models_round_trip(
    parameters_class: type[PKHOperatorParameters],
    sampler_parameters_class: type[PKHParameters],
) -> None:
    """Operator configuration adds a positive budget to sampler settings."""

    parameters = parameters_class(
        targetOutput="benchmark.latency",
        numberEntities=7,
        nEstimators=11,
        minSamplesLeaf=1,
        nJobs=1,
    )
    restored = parameters_class.model_validate(parameters.model_dump())
    sampler_parameters = sampler_parameters_class.model_validate(
        parameters.model_dump(exclude={"numberEntities"})
    )

    assert restored == parameters
    assert restored.numberEntities == 7
    assert sampler_parameters.targetOutput == "benchmark.latency"
    assert "numberEntities" not in sampler_parameters.model_dump()

    with pytest.raises(pydantic.ValidationError):
        parameters_class(targetOutput="latency", numberEntities=0)


@pytest.mark.parametrize(
    ("operator_parameters", "selector_class"),
    [
        (
            PKHOperatorParameters(
                targetOutput="latency",
                numberEntities=3,
                epochLength=2,
                nJobs=1,
            ),
            PKHSampleSelector,
        ),
    ],
    ids=["pkh"],
)
def test_operator_settings_construct_the_matching_sampler(
    operator_parameters: PKHOperatorParameters,
    selector_class: type[PKHSampleSelector],
) -> None:
    """Each operator's settings load only its own sampler implementation."""

    configuration = CustomSamplerConfiguration(
        module=SamplerModuleConf(
            moduleName=selector_class.__module__,
            moduleClass=selector_class.__name__,
        ),
        parameters=operator_parameters.model_dump(exclude={"numberEntities"}),
    )

    selector = configuration.sampler()

    assert isinstance(selector, selector_class)
    assert selector.params.targetOutput == operator_parameters.targetOutput


@pytest.mark.parametrize(
    ("operator_parameters", "selector_class"),
    [
        (
            PKHOperatorParameters(
                targetOutput="latency",
                numberEntities=4,
                nEstimators=5,
                minSamplesLeaf=1,
                nJobs=1,
            ),
            PKHSampleSelector,
        ),
    ],
    ids=["pkh"],
)
def test_operator_delegates_to_sequential_unfiltered_random_walk(
    operator_parameters: PKHOperatorParameters,
    selector_class: type[PKHSampleSelector],
) -> None:
    """Each operator builds a safe adaptive RandomWalk configuration."""

    random_walk_parameters = _random_walk_parameters(
        operator_parameters,
        number_entities=operator_parameters.numberEntities,
        sampler_class=selector_class,
    )

    assert random_walk_parameters.numberEntities == operator_parameters.numberEntities
    assert random_walk_parameters.batchSize == 1
    assert random_walk_parameters.singleMeasurement is True
    assert random_walk_parameters.filter.filterMode is FilterModeEnum.noFilter
    assert isinstance(
        random_walk_parameters.samplerConfig,
        CustomSamplerConfiguration,
    )
    assert random_walk_parameters.samplerConfig.module == SamplerModuleConf(
        moduleName=selector_class.__module__,
        moduleClass=selector_class.__name__,
    )
    selector = random_walk_parameters.samplerConfig.sampler()
    assert isinstance(selector, selector_class)
    assert selector.params.targetOutput == operator_parameters.targetOutput


def test_nested_random_walk_operation_becomes_an_outer_resource() -> None:
    """Retain nested operation provenance without replacing the outer operation."""

    nested_operation = OperationResource(
        operationType=DiscoveryOperationEnum.EXPLORE,
        operatorIdentifier="random_walk-0.1.0",
        config=DiscoveryOperationResourceConfiguration(
            spaces=["space-test"],
            operation=DiscoveryOperationConfiguration(),
        ),
    )

    output = _outer_operation_output(
        OperationOutput(operation=nested_operation, metadata={"nested": True})
    )
    empty_output = _outer_operation_output(OperationOutput())

    assert output.operation is None
    assert output.resources == [nested_operation]
    assert output.metadata == {}
    assert output.other == []
    assert empty_output.operation is None
    assert empty_output.resources == []
