# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging
from typing import Annotated

import pydantic
from no_priors_characterization.no_priors_pydantic import NoPriorsParameters
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    model_config = ConfigDict(extra="allow")  # Allows optional extra params

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
        str | None,
        pydantic.Field(
            description="The relative path of the model directory from the root folder.",
        ),
    ] = None

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
            description="Parameters of the no_priors_characterization operation. "
            "The targetOutput will be automatically set from TrimParameters.targetOutput.",
        ),
    ] = NoPriorsParameters(targetOutput="")

    # disablePredictiveModeling: Annotated[
    #     bool,
    #     pydantic.Field(
    #         description="Routes trim to a progressive sampler",
    #     ),
    # ] = False

    @classmethod
    def defaultOperationParameters(cls) -> "TrimParameters":
        return cls(targetOutput="TO_BE_SET")

    @model_validator(mode="after")
    def set_final_model_args(self) -> "TrimParameters":
        if self.finalModelAutoGluonArgs == AutoGluonArgs():
            self.finalModelAutoGluonArgs = self.autoGluonArgs
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

    @model_validator(mode="after")
    def set_model_folder(self) -> "TrimParameters":
        if self.autoGluonArgs.tabularPredictorArgs.get("path", None):
            if self.outputDirectory:
                if (
                    self.autoGluonArgs.tabularPredictorArgs["path"]
                    != self.outputDirectory
                ):
                    logging.error(
                        f"Mismatch in model save path configuration: "
                        f"AutoGluonArgs specifies '{self.autoGluonArgs.tabularPredictorArgs['path']}', "
                        f"but expected '{self.outputDirectory}'. Changing to {self.outputDirectory}"
                    )
                    self.autoGluonArgs.tabularPredictorArgs["path"] = (
                        self.outputDirectory
                    )
            else:
                logging.info(
                    f"Model folder is: {self.autoGluonArgs.tabularPredictorArgs['path']}"
                )
                self.outputDirectory = self.autoGluonArgs.tabularPredictorArgs["path"]
        else:
            self.autoGluonArgs.tabularPredictorArgs["path"] = self.outputDirectory

        return self

    @model_validator(mode="after")
    def set_no_priors_target_output(self) -> "TrimParameters":
        if self.noPriorParameters.targetOutput != self.targetOutput:
            logging.debug(
                "set_no_priors_target_output: Synchronizing target output between TRIM and no-priors characterization.\n"
                f"  noPriorParameters.targetOutput = '{self.noPriorParameters.targetOutput}'\n"
                f"  TrimParameters.targetOutput = '{self.targetOutput}'\n"
                f"  Setting noPriorParameters.targetOutput = '{self.targetOutput}'"
            )
            self.noPriorParameters.targetOutput = self.targetOutput
        return self


if __name__ == "__main__":
    # Test with required targetOutput parameter
    params = TrimParameters.model_validate(
        TrimParameters(
            targetOutput="test",
            samplingBudget=SamplingBudget(minPoints=10),
            noPriorParameters=NoPriorsParameters(targetOutput="test", samples=2),
        )
    )
    print(f"Parameters set are:\n{params}")
