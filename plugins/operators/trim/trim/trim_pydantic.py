# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging

import pydantic
from pydantic import BaseModel, ConfigDict, Field, model_validator

from orchestrator.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    OperatorFunctionConf,
)
from trim.no_priors_pydantic import NoPriorsParameters


class SamplingBudget(pydantic.BaseModel):
    minPoints: int = pydantic.Field(
        default=18,
        description="Minimum number of points to sample, default is setting this equal to the number of features",
    )
    maxPoints: int = pydantic.Field(
        default=22,
        description="Maximum number of points to sample, default is setting this equal to 80 per cent of the target space",
    )


class StoppingCriterion(pydantic.BaseModel):
    enabled: bool = pydantic.Field(
        default=True, description="Whether to enable stopping criterion"
    )
    meanThreshold: float = pydantic.Field(
        default=0.9, description="Mean threshold for stopping"
    )
    stdThreshold: float = pydantic.Field(
        default=0.75, description="Standard deviation threshold for stopping"
    )


class AutoGluonArgs(BaseModel):
    tabularPredictorArgs: dict = Field(
        default={},
        description="A dictionary containing key-value pairs of AutoGluon optional parameters in Tabular Predictor",
    )

    fitArgs: dict = Field(
        default={"time_limit": 60, "presets": "medium"},
        description="A dictionary containing key-value pairs of AutoGluon optional parameters in Tabular Predictor fit",
    )


class TrimParameters(BaseModel):
    model_config = ConfigDict(extra="allow")  # Allows optional extra params

    autoGluonArgs: AutoGluonArgs = Field(
        default_factory=AutoGluonArgs,
        description="""Contains pydantic models for both autogluon TabularPredictor and for its fit function.
        Both models are dictionaries whose key-value pairs are AutoGluon optional parameters.""",
    )

    finalModelAutoGluonArgs: AutoGluonArgs = Field(
        default_factory=AutoGluonArgs,
        description="""Contains pydantic models for both autogluon TabularPredictor and for its fit function.
        Both models are dictionaries whose key-value pairs are AutoGluon optional parameters.
        These parameters are used when finalizing the model.
        That is, all sampled points go in the training set""",
    )

    targetOutput: str = pydantic.Field(
        default="hi",
        description="The measured property you will treat as a target variable",
    )

    outputDirectory: str | None = pydantic.Field(
        default=None,
        description="""
            The relative path of the model directory from the root folder.
        """,
    )

    debugDirectory: str = pydantic.Field(
        default="debug_output",
        description="""
            The relative path of the directory where debug files will be stored.
        """,
    )

    iterationSize: int = pydantic.Field(
        default=5, description="TRIM iteration size, sets"
    )

    batchSize: int | None = pydantic.Field(
        default=None,
        description="Batch size parameter of randomWalk, default is setting this equal to iterationSize",
    )

    holdoutSize: int | None = pydantic.Field(
        default=None,
        description="Sample Size of the holdout set, default is setting this equal to iterationSize",
    )

    samplingBudget: SamplingBudget = pydantic.Field(
        default=SamplingBudget(), description="Sampling budget configuration"
    )

    independentVariablesToKeep: int = pydantic.Field(
        default=7,
        description="Number of independent variables to retain based on importance",
    )

    minMeasuredEntities: int = pydantic.Field(
        default=16,
        description="Minimum number of entities already Measured in the given Discovery Space, otherwise I will have an error",
    )

    stoppingCriterion: StoppingCriterion = pydantic.Field(
        default=StoppingCriterion(), description="Stopping criterion configuration"
    )

    noPriorParameters: NoPriorsParameters = pydantic.Field(
        default=NoPriorsParameters(),
        description="Parameters of the no_priors_characterization operation",
    )

    @classmethod
    def defaultOperation(cls):
        return DiscoveryOperationConfiguration(
            module=OperatorFunctionConf(
                operatorName="trim",
                operationType=DiscoveryOperationEnum.CHARACTERIZE,
            ),
            parameters=cls(),  # cls.model_validate({'batchSize':2}),
        )

    @model_validator(mode="after")
    def set_final_model_args(self):
        if self.finalModelAutoGluonArgs == AutoGluonArgs():
            self.finalModelAutoGluonArgs = self.autoGluonArgs
        return self

    @model_validator(mode="after")
    def set_final_batch_size(self):
        if not self.batchSize:
            self.batchSize = self.iterationSize
        return self

    @model_validator(mode="after")
    def set_final_holdout_size(self):
        if not self.holdoutSize:
            self.holdoutSize = self.iterationSize
        return self

    @model_validator(mode="after")
    def set_model_folder(self):
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
    def set_no_priors_target_output(self):
        if self.noPriorParameters.targetOutput != self.targetOutput:
            logging.error(
                f"No priors characterization target output = {self.noPriorParameters.targetOutput }"
                f"Trim target output = {self.targetOutput }"
                f"Setting them equal to '{self.targetOutput }'."
            )
            self.noPriorParameters.targetOutput = self.targetOutput
        return self


if __name__ == "__main__":
    params = TrimParameters.model_validate(TrimParameters())
    print(
        f"type of model_validate output on TRIM default is {type(params)}, printing the full object gives {params}"
    )
    print(f"Default operation is {TrimParameters.defaultOperation()}")
