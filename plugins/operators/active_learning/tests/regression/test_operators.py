# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Tests for the distinct PKH and FLORA operator registrations."""

from collections.abc import Callable

import pydantic
import pytest
from active_learning.regression._shared import (
    _outer_operation_output,
    _random_walk_parameters,
)
from active_learning.regression.flora.operator import flora
from active_learning.regression.flora.parameters import (
    FLORAOperatorParameters,
    FLORAParameters,
)
from active_learning.regression.flora.sampler import FLORASampleSelector
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
        ("flora", flora, FLORAOperatorParameters),
    ],
)
def test_import_registers_two_distinct_operators(
    name: str,
    function: Callable[..., OperationOutput],
    configuration_model: type[PKHOperatorParameters] | type[FLORAOperatorParameters],
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
        (FLORAOperatorParameters, FLORAParameters),
    ],
    ids=["pkh", "flora"],
)
@pytest.mark.parametrize("pilot_strategy", ["random", "clhs", "sobol"])
def test_operator_parameter_models_round_trip(
    parameters_class: type[PKHOperatorParameters] | type[FLORAOperatorParameters],
    sampler_parameters_class: type[PKHParameters] | type[FLORAParameters],
    pilot_strategy: str,
) -> None:
    """Operator configuration adds a positive budget to sampler settings."""

    parameters = parameters_class(
        targetOutput="benchmark.latency",
        numberEntities=7,
        nEstimators=11,
        minSamplesLeaf=1,
        nJobs=1,
        pilotSize=3,
        pilotStrategy=pilot_strategy,
        maxFeatures=0.5,
        outputDirectory="saved-model",
    )
    restored = parameters_class.model_validate(parameters.model_dump())
    sampler_parameters = sampler_parameters_class.model_validate(
        parameters.model_dump(include=set(sampler_parameters_class.model_fields))
    )

    assert restored == parameters
    assert restored.numberEntities == 7
    assert restored.outputDirectory == "saved-model"
    assert sampler_parameters.targetOutput == "benchmark.latency"
    assert restored.pilotSize == 3
    assert sampler_parameters.pilotSize == 3
    assert restored.pilotStrategy == pilot_strategy
    assert sampler_parameters.pilotStrategy == pilot_strategy
    assert restored.maxFeatures == 0.5
    assert sampler_parameters.maxFeatures == 0.5
    assert "numberEntities" not in sampler_parameters.model_dump()
    assert "outputDirectory" not in sampler_parameters.model_dump()

    with pytest.raises(pydantic.ValidationError):
        parameters_class(targetOutput="latency", numberEntities=0)


@pytest.mark.parametrize(
    ("parameters_class", "directory"),
    [(PKHOperatorParameters, "pkh_models"), (FLORAOperatorParameters, "flora_models")],
)
def test_operator_model_directory_defaults_and_validation(
    parameters_class: type[PKHOperatorParameters] | type[FLORAOperatorParameters],
    directory: str,
) -> None:
    """Each operator has a distinct output directory and rejects blank paths."""

    parameters = parameters_class(targetOutput="latency", numberEntities=3)
    assert parameters.outputDirectory == directory
    for invalid in ("", "   "):
        with pytest.raises(pydantic.ValidationError):
            parameters_class(
                targetOutput="latency", numberEntities=3, outputDirectory=invalid
            )


@pytest.mark.parametrize(
    ("operator_parameters", "selector_class"),
    [
        (
            PKHOperatorParameters(
                targetOutput="latency",
                numberEntities=3,
                epochLength=2,
                nJobs=1,
                maxFeatures=1.0,
            ),
            PKHSampleSelector,
        ),
        (
            FLORAOperatorParameters(
                targetOutput="latency",
                numberEntities=3,
                nJobs=1,
                maxFeatures=1.0,
            ),
            FLORASampleSelector,
        ),
    ],
    ids=["pkh", "flora"],
)
def test_operator_settings_construct_the_matching_sampler(
    operator_parameters: PKHOperatorParameters | FLORAOperatorParameters,
    selector_class: type[PKHSampleSelector] | type[FLORASampleSelector],
) -> None:
    """Each operator's settings load only its own sampler implementation."""

    configuration = CustomSamplerConfiguration(
        module=SamplerModuleConf(
            moduleName=selector_class.__module__,
            moduleClass=selector_class.__name__,
        ),
        parameters=operator_parameters.model_dump(
            exclude={"numberEntities", "outputDirectory", "finalPredictor", "autoGluon"}
        ),
    )

    selector = configuration.sampler()

    assert isinstance(selector, selector_class)
    assert selector.params.targetOutput == operator_parameters.targetOutput
    assert selector.params.maxFeatures == 1.0
    assert isinstance(selector.params.maxFeatures, float)


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
                maxFeatures=1.0,
            ),
            PKHSampleSelector,
        ),
        (
            FLORAOperatorParameters(
                targetOutput="latency",
                numberEntities=4,
                nEstimators=5,
                minSamplesLeaf=1,
                nJobs=1,
                maxFeatures=1.0,
            ),
            FLORASampleSelector,
        ),
    ],
    ids=["pkh", "flora"],
)
def test_operator_delegates_to_sequential_unfiltered_random_walk(
    operator_parameters: PKHOperatorParameters | FLORAOperatorParameters,
    selector_class: type[PKHSampleSelector] | type[FLORASampleSelector],
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
    assert selector.params.maxFeatures == 1.0
    assert isinstance(selector.params.maxFeatures, float)


@pytest.mark.parametrize(
    ("parameters_class", "selector_class"),
    [
        (PKHOperatorParameters, PKHSampleSelector),
        (FLORAOperatorParameters, FLORASampleSelector),
    ],
)
def test_final_predictor_choice_preserves_acquisition_configuration(
    parameters_class: type[PKHOperatorParameters] | type[FLORAOperatorParameters],
    selector_class: type[PKHSampleSelector] | type[FLORASampleSelector],
) -> None:
    """Final model selection round-trips without changing the sampling settings."""

    default = parameters_class(targetOutput="latency", numberEntities=20)
    selected = parameters_class(
        targetOutput="latency",
        numberEntities=20,
        finalPredictor="autogluon",
        autoGluon={
            "presets": "medium_quality",
            "timeLimit": 30,
            "numCpus": 1,
            "hyperparameters": {"RF": {"n_estimators": 8}},
        },
    )
    assert default.finalPredictor == "random_forest"
    assert parameters_class.model_validate(selected.model_dump()) == selected
    default_walk = _random_walk_parameters(
        default, number_entities=20, sampler_class=selector_class
    )
    selected_walk = _random_walk_parameters(
        selected, number_entities=20, sampler_class=selector_class
    )
    assert default_walk == selected_walk
    assert "finalPredictor" not in selected_walk.samplerConfig.parameters
    assert "autoGluon" not in selected_walk.samplerConfig.parameters


@pytest.mark.parametrize(
    "parameters_class", [PKHOperatorParameters, FLORAOperatorParameters]
)
@pytest.mark.parametrize(
    "options",
    [
        {"finalPredictor": "unknown"},
        {"autoGluon": {"timeLimit": 0}},
        {"autoGluon": {"timeLimit": float("inf")}},
        {"autoGluon": {"numCpus": 0}},
        {"autoGluon": {"numCpus": True}},
        {"autoGluon": {"presets": "unknown"}},
        {"autoGluon": {"train_data": "unmeasured.csv"}},
        {"autoGluon": {"tuning_data": "unmeasured.csv"}},
        {"autoGluon": {"path": "another-model"}},
    ],
)
def test_invalid_final_predictor_configuration_is_rejected(
    parameters_class: type[PKHOperatorParameters] | type[FLORAOperatorParameters],
    options: dict,
) -> None:
    """Final predictor configuration owns its data and output paths."""

    with pytest.raises(pydantic.ValidationError):
        parameters_class(targetOutput="latency", numberEntities=20, **options)


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
