# Copyright (c) IBM Corporation

# SPDX-License-Identifier: MIT

import functools
import importlib.resources
import logging
import math
import traceback
from typing import NamedTuple

from autogluon.tabular import TabularPredictor

from autoconf.utils.pydantic_models import JobConfig
from autoconf.utils.recommender import (
    NoRecommendationError,
    recommend_min_gpu,
)
from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import ConstitutiveProperty

moduleLog = logging.getLogger()


class GPUsAndWorkers(NamedTuple):
    gpus: int
    workers: int


@functools.cache
def load_model(model_version: str) -> TabularPredictor:
    """Loads the model

    Args:
        model_version:
            The version of the Autogluon model to use e.g. 1.0.0

    Returns:
        The predictor
    """
    if model_version == "1.1.0":
        path_weights = str(
            importlib.resources.files("autoconf")
            / "AutoGluonModels"
            / "v1-1-0_ag-20251112_155927-refit-clone-opt"
        )
    elif model_version == "2.0.0":
        path_weights = str(
            importlib.resources.files("autoconf")
            / "AutoGluonModels"
            / "v2-0-0_ag-20251113_154241-refit-clone-opt"
        )
    else:
        raise ValueError("Unknown model_version", model_version)

    return TabularPredictor.load(path_weights, require_py_version_match=False)


ModelVersion = ConstitutiveProperty(
    identifier="model_version",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["1.1.0", "2.0.0"],
    ),
)

ModelName = ConstitutiveProperty(
    identifier="model_name",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=[
            "allam-1-13b",
            "granite-13b-v2",
            "granite-20b-v2",
            "granite-3-8b",
            "granite-3.1-2b",
            "granite-3.1-3b-a800m-instruct",
            "granite-3.1-8b-instruct",
            "granite-34b-code-base",
            "granite-3b-code-base-128k",
            "granite-4.0-1b",
            "granite-4.0-350m",
            "granite-4.0-h-1b",
            "granite-4.0-h-micro",
            "granite-4.0-h-small",
            "granite-4.0-h-tiny",
            "granite-4.0-micro",
            "granite-7b-base",
            "granite-8b-code-base",
            "granite-8b-japanese",
            "llama-13b",
            "llama-7b",
            "llama2-70b",
            "llama3-70b",
            "llama3-8b",
            "llama3.1-405b",
            "llama3.1-70b",
            "llama3.1-8b",
            "mistral-123b-v2",
            "mistral-7b-v0.1",
            "mixtral-8x7b-instruct-v0.1",
        ],
    ),
)

TuningMethod = ConstitutiveProperty(
    identifier="method",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE, values=["full", "lora"]
    ),
)

GPUModel = ConstitutiveProperty(
    identifier="gpu_model",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=[
            "L40S",
            "NVIDIA-A100-80GB-PCIe",
            "NVIDIA-A100-SXM4-80GB",
            "NVIDIA-H100-PCIe",
        ],
    ),
)

TokensPerSample = ConstitutiveProperty(
    identifier="tokens_per_sample",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 10000001],  # VV: Seen values go up to 8192
    ),
)

BatchSize = ConstitutiveProperty(
    identifier="batch_size",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 10000001],  # VV: Seen values go up to 128
    ),
)
GPUsPerWorker = ConstitutiveProperty(
    identifier="gpus_per_worker",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 10000001],  # VV: Seen values is just [8]
    ),
)

MaxGPUs = ConstitutiveProperty(
    identifier="max_gpus",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 10000001],  # VV: Seen values go up to 128
    ),
)


@custom_experiment(
    required_properties=[
        ModelName,
        TuningMethod,
        GPUModel,
        TokensPerSample,
        BatchSize,
        ModelVersion,
    ],
    optional_properties=[GPUsPerWorker, MaxGPUs],
    output_property_identifiers=["can_recommend", "gpus", "workers"],
    metadata={
        "description": "An AutoConf plugin that suggests the minimum number of "
        "gpus per worker and number of workers necessary to execute a Tuning job"
    },
    parameterization={},
)
def min_gpu_recommender(
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    model_version: str,
    gpus_per_worker: int = 8,
    max_gpus: int = 8,
) -> dict[str, int | bool]:
    try:
        parameters = {
            "model_name": model_name,
            "method": method,
            "gpu_model": gpu_model,
            "tokens_per_sample": tokens_per_sample,
            "batch_size": batch_size,
            "model_version": model_version,
            "gpus_per_worker": gpus_per_worker,
            "max_gpus": max_gpus,
        }
        try:
            predictor = load_model(model_version=model_version)

            config = JobConfig.model_validate(
                {
                    "model_name": parameters["model_name"],
                    "gpu_model": parameters["gpu_model"],
                    "method": parameters["method"],
                    "tokens_per_sample": parameters["tokens_per_sample"],
                    "batch_size": parameters["batch_size"],
                }
            )
            moduleLog.debug(f"Configuration supplied is {config}")
            valid_n_gpus = []
            i = 1
            while i <= max_gpus:
                valid_n_gpus.append(i)
                i *= 2

            min_gpus, metadata = recommend_min_gpu(
                config, predictor=predictor, valid_n_gpu_list=valid_n_gpus
            )

            if min_gpus < 1:
                raise NoRecommendationError(str(metadata))

            workers = math.ceil(min_gpus / gpus_per_worker)
            gpus = math.ceil(min_gpus / workers)

            ret = GPUsAndWorkers(gpus=gpus, workers=workers)
        except NoRecommendationError as e:
            moduleLog.warning(
                f"recommend_min_gpus_and_workers() for {parameters} cannot produce a recommendation: {e}"
            )
            return {"can_recommend": False}
        except ValueError as e:
            # Handling the case when the validation of pydantic model fails
            moduleLog.warning(
                f"recommend_min_gpus_and_workers() for {parameters} failed with error {e}"
            )
            moduleLog.debug(f"Traceback {traceback.format_exc()}")
            return {"can_recommend": False}

        return {
            "can_recommend": True,
            "gpus": ret.gpus,
            "workers": ret.workers,
        }
    except Exception as e:
        # General failure due to recommender model not loading.. autogluon environment issues
        # should result in InvalidMeasurements
        moduleLog.warning(e)
        raise e
