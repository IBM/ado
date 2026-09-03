# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import enum
import logging
from typing import Annotated, Literal

import pydantic
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


class MissingTargetMeasurementMode(str, enum.Enum):
    Error = "Error"
    InjectDefaultValue = "InjectDefaultValue"
    Skip = "Skip"


class MissingTargetMeasurements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget: Annotated[
        int | None,
        pydantic.Field(
            description="Maximum number of measurements missing targetOutput to tolerate before raising an error. "
            "None means unlimited. Does not apply when @mode=Error."
        ),
    ] = None

    mode: Annotated[
        MissingTargetMeasurementMode,
        pydantic.Field(
            description="Action to take when a measurement has no targetOutput value. "
            "Error: abort the operation after a single such measurement. "
            "Skip: exclude the measurement. "
            "InjectDefaultValue: substitute defaultValue as the targetOutput."
        ),
    ] = MissingTargetMeasurementMode.Error

    defaultValue: Annotated[
        float,
        pydantic.Field(
            description="Value substituted for a missing targetOutput. Only used when mode=InjectDefaultValue."
        ),
    ] = 0.0


class NoPriorsParameters(BaseModel):
    """
    Parameters for sampling high-dimensional spaces without prior model structure.

    The `sampling_strategy` must be one of the Literals supported.
    Source of truth for supported strategies is the comment block right here:

        strategy (str): sampling subroutine:
        - 'random': selects random points from the beginning
        - 'clhs': refer to concatenated_latin_hypercube_sampling
        - 'sobol': sobol sampling
    """

    samples: Annotated[
        int,
        Field(
            ge=1,
            description="Number of unique points to sample (must be >= 1).",
        ),
    ] = 20

    batchSize: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Batch size parameter used by certain samplers (e.g., randomWalk) via continuous batching; "
                "by default set equal to iterationSize in those contexts. Must be >= 1."
            ),
        ),
    ] = 1

    sampling_strategy: Annotated[
        Literal["random", "clhs", "sobol"],
        BeforeValidator(lambda s: s.lower()),
        Field(
            description=(
                "Sampling subroutine. Supported values:\n"
                " - 'random': selects random points from the beginning\n"
                " - 'clhs': dimension-wise random without replacement until each dim cycles\n"
                " - 'sobol': sobol sampling via scipy\n"
                "Validation is case-insensitive; value is normalized to lowercase."
            ),
        ),
    ] = "clhs"


class NoPriorsParametersInternal(NoPriorsParameters):
    """Runtime extension of NoPriorsParameters used only by NoPriorsSampler.

    These fields are NOT on NoPriorsParameters because that model is serialised
    to YAML as the operation configuration.
    """

    targetOutput: Annotated[
        str,
        Field(
            description="The measured property you will treat as a target variable.",
        ),
    ]

    missingTargetMeasurements: Annotated[
        MissingTargetMeasurements,
        Field(
            description="Controls how the no-priors sampler handles measurements that have no targetOutput value.",
        ),
    ] = MissingTargetMeasurements()


class SamplingBudget(pydantic.BaseModel):
    minPoints: Annotated[
        int,
        pydantic.Field(
            description="Minimum number of points to sample, "
            "a suggestion is setting this equal to twice the number of features",
        ),
    ] = 18
    maxPoints: Annotated[
        int,
        pydantic.Field(
            description="Maximum number of points to sample, "
            "a suggestion is setting this equal to 80 per cent of the target space",
        ),
    ] = 40


class StoppingCriterion(pydantic.BaseModel):
    enabled: Annotated[
        bool,
        pydantic.Field(description="Whether to enable stopping criterion"),
    ] = True
    meanThreshold: Annotated[
        float,
        pydantic.Field(description="Mean threshold for stopping"),
    ] = 0.9
    stdThreshold: Annotated[
        float,
        pydantic.Field(description="Standard deviation threshold for stopping"),
    ] = 0.75


class AutoGluonArgs(BaseModel):
    tabularPredictorArgs: Annotated[
        dict,
        Field(
            default_factory=lambda: {"verbosity": 1},
            description="A dictionary containing key-value pairs of "
            "AutoGluon optional parameters in Tabular Predictor",
        ),
    ]

    fitArgs: Annotated[
        dict,
        Field(
            default_factory=lambda: {
                "time_limit": 60,
                "presets": "medium",
                "excluded_model_types": ["GBM"],
            },
            description="A dictionary containing key-value pairs of "
            "AutoGluon optional parameters in Tabular Predictor fit",
        ),
    ]


class TrimParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    autoGluonArgs: Annotated[
        AutoGluonArgs,
        Field(
            description="Contains pydantic models for both autogluon TabularPredictor and for its fit function. "
            "Both models are dictionaries whose key-value pairs are AutoGluon optional parameters.",
        ),
    ] = AutoGluonArgs()

    finalModelAutoGluonArgs: Annotated[
        AutoGluonArgs,
        Field(
            description="Contains pydantic models for both autogluon TabularPredictor and for its fit function."
            "Both models are dictionaries whose key-value pairs are AutoGluon optional parameters."
            "These parameters are used when finalizing the model."
            "That is, all sampled points go in the training set",
        ),
    ] = AutoGluonArgs()

    targetOutput: Annotated[
        str,
        pydantic.Field(
            description="The measured property you will treat as a target variable",
        ),
    ]

    outputDirectory: Annotated[
        str,
        pydantic.Field(
            description="The relative path of the model directory from the root folder.",
        ),
    ] = "trim_models"

    debugDirectory: Annotated[
        str,
        pydantic.Field(
            description="The relative path of the directory where debug files will be stored.",
        ),
    ] = "debug_output"

    iterationSize: Annotated[
        int,
        pydantic.Field(
            description="TRIM iteration size, sets the number of models that"
            "the stopping criterion considers when determining whether to stop"
        ),
    ] = 5

    holdoutSize: Annotated[
        int | None,
        pydantic.Field(
            description="Sample Size of the holdout set, default is setting this equal to iterationSize",
        ),
    ] = None

    samplingBudget: Annotated[
        SamplingBudget,
        pydantic.Field(
            description="Sampling budget configuration",
        ),
    ] = SamplingBudget()

    stoppingCriterion: Annotated[
        StoppingCriterion,
        pydantic.Field(
            description="Stopping criterion configuration",
        ),
    ] = StoppingCriterion()

    noPriorParameters: Annotated[
        NoPriorsParameters,
        pydantic.Field(
            description="Parameters of the no_priors_characterization operation.",
        ),
    ] = NoPriorsParameters()

    missingTargetMeasurements: Annotated[
        MissingTargetMeasurements,
        pydantic.Field(
            description="Controls how TRIM handles measurements that have no targetOutput value."
        ),
    ] = MissingTargetMeasurements()

    # disablePredictiveModeling: Annotated[
    #     bool,
    #     pydantic.Field(
    #         description="Routes trim to a progressive sampler",
    #     ),
    # ] = False

    @classmethod
    def example_configuration(cls) -> "TrimParameters":
        return cls(targetOutput="TO_BE_SET")

    @model_validator(mode="after")
    def set_final_model_args(self) -> "TrimParameters":
        if self.finalModelAutoGluonArgs == AutoGluonArgs():
            self.finalModelAutoGluonArgs = self.autoGluonArgs.model_copy(deep=True)
        return self

    @model_validator(mode="after")
    def set_holdout_size(self) -> "TrimParameters":
        if not self.holdoutSize:
            self.holdoutSize = self.iterationSize
        if self.holdoutSize != self.iterationSize:
            logging.warning(
                "Currently the holdout size must be equal to the iterationSize."
                f"Setting it equals to it. Batch size = {self.iterationSize}"
            )
            self.holdoutSize = self.iterationSize
        return self

    @model_validator(mode="after")
    def set_no_priors_sample(self) -> "TrimParameters":
        if self.samplingBudget.minPoints != self.noPriorParameters.samples:
            logging.info(
                "Overwriting the 'samples' field of the no-priors characterization.\n"
                f"  samplingBudget.minPoints = {self.samplingBudget.minPoints}\n"
                f"  noPriorParameters.samples = {self.noPriorParameters.samples}\n"
                f"  Setting noPriorParameters.samples = {self.samplingBudget.minPoints}"
            )
        self.noPriorParameters.samples = self.samplingBudget.minPoints
        return self


class TrimSamplerParameters(TrimParameters):
    """Runtime extension of TrimParameters used only by TrimSampleSelector.

    These fields are NOT on TrimParameters because
    that model is serialised to YAML as the operation configuration.
    """

    numberEntitiesIterativeModeling: Annotated[
        int,
        Field(
            description="Number of entities RandomWalk will draw during the "
            "iterative modeling phase. Used by the sampler to detect the last "
            "yield and call finalize_model() before RandomWalk stops consuming "
            "the generator.",
        ),
    ]

    missingTargetMeasurements: Annotated[
        MissingTargetMeasurements,
        pydantic.Field(
            description="Controls how TRIM handles measurements that have no targetOutput value."
        ),
    ] = MissingTargetMeasurements()


class TrimSamplerParametersInternal(TrimSamplerParameters):
    """Runtime extension of TrimSamplerParameters used only by TrimSampleSelector.

    These fields are NOT on TrimSamplerParameters because that model is
    serialised into the RandomWalk operation configuration.
    """

    noPriorsOperationId: Annotated[
        str | None,
        Field(
            description="Operation identifier of the no-priors RandomWalk phase. "
            "Used by TrimSampleSelector to query invalid measurement results from "
            "that phase and pre-populate injected defaults before iterative modeling.",
        ),
    ] = None


if __name__ == "__main__":
    # Test with required targetOutput parameter
    params = TrimParameters.model_validate(
        TrimParameters(
            targetOutput="test",
            samplingBudget=SamplingBudget(minPoints=10),
            noPriorParameters=NoPriorsParameters(samples=2),
        )
    )
    print(f"Parameters set are:\n{params}")
