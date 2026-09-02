# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

import functools
import logging
import math
import traceback
from pathlib import Path
from typing import NamedTuple

from autogluon.tabular import TabularPredictor

from ado.modules.actuators.custom_experiments import custom_experiment
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.property import ConstitutiveProperty
from autoconf.model_paths import (
    MODEL_VERSION,
    model_path,
)
from autoconf.utils.pydantic_models import JobConfig
from autoconf.utils.recommender import (
    NoRecommendationError,
    get_model_prediction_and_metadata,
    recommend_min_gpu,
)

moduleLog = logging.getLogger()


class GPUsAndWorkers(NamedTuple):
    gpus: int
    workers: int


@functools.cache
def load_model(model_version: str, model_root: Path | None = None) -> TabularPredictor:
    """Load a locally generated AutoConf model.

    Args:
        model_version: Version of the AutoGluon model to use.
        model_root: Optional local model root override.

    Returns:
        The loaded predictor.

    Raises:
        FileNotFoundError: If the model has not been generated locally.
        ValueError: If the model version is unsupported.
    """

    if model_version != MODEL_VERSION:
        raise ValueError(f"Unknown model_version: {model_version}")

    path_weights = model_path(model_root)
    if not path_weights.is_dir():
        raise FileNotFoundError(
            f"AutoConf model {model_version} was not found at {path_weights}. "
            "Generate it in this environment with: autoconf_build_model"
        )

    return TabularPredictor.load(str(path_weights), require_py_version_match=False)


ModelVersion = ConstitutiveProperty(
    identifier="model_version",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=[MODEL_VERSION],
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

# Need to separate this from the
PerDeviceBatchSize = ConstitutiveProperty(
    identifier="per_device_train_batch_size",
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

NumberGPUs = ConstitutiveProperty(
    identifier="number_gpus",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 1024 + 1],  # VV: Arbitrary cutoff at 1024 GPUs
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


@custom_experiment(
    required_properties=[
        ModelName,
        TuningMethod,
        GPUModel,
        TokensPerSample,
        PerDeviceBatchSize,
        NumberGPUs,
    ],
    optional_properties=[GPUsPerWorker, MaxGPUs, ModelVersion],
    output_property_identifiers=["can_recommend", "gpus", "workers"],
    metadata={
        "description": "An AutoConf recommender that preserves the requested number of GPUs "
        "if it won't cause GPU OOM, otherwise recommends the minimum number of GPUs needed. "
        "Keeps the per-device batch size constant."
    },
    parameterization={},
)
def avoid_oom_recommender(
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    per_device_train_batch_size: int,
    number_gpus: int,
    gpus_per_worker: int = 8,
    max_gpus: int = 64,
    model_version: str = MODEL_VERSION,
) -> dict[str, int | bool]:

    result = {
        "can_recommend": False,
        "gpus": -1,
        "workers": -1,
    }
    try:
        # First, load the model
        predictor: TabularPredictor = load_model(model_version)
        configuration: dict = {
            "model_name": model_name,
            "method": method,
            "gpu_model": gpu_model,
            "tokens_per_sample": int(tokens_per_sample),
        }

        # Step 1: Check if the original number_gpus would work without OOM
        original_batch_size = per_device_train_batch_size * number_gpus
        configuration["batch_size"] = original_batch_size
        configuration["number_gpus"] = number_gpus
        moduleLog.debug(
            f"Step 1: Checking if original number_gpus={number_gpus} works with batch_size={original_batch_size}"
        )
        gpus_can_support_run, _ = get_model_prediction_and_metadata(
            configuration, predictor=predictor
        )

        if gpus_can_support_run == 1:
            # Original number_gpus works without OOM, preserve it
            workers = math.ceil(number_gpus / gpus_per_worker)
            gpus = math.ceil(number_gpus / workers)
            result["can_recommend"] = True
            result["gpus"] = gpus
            result["workers"] = workers
            moduleLog.debug(
                f"Original number_gpus={number_gpus} is valid (no OOM), returning workers:{workers} gpus:{gpus}"
            )
            return result

        # Step 2: Original number_gpus would cause OOM, find minimum GPUs needed
        moduleLog.debug(
            f"Step 2: Original number_gpus={number_gpus} would cause OOM, searching for minimum"
        )
        num_gpu_list = [2**i for i in range(int(math.log2(max_gpus)) + 1)]

        # Do not check for GPUs that are fewer than the original GPUs
        num_gpu_list = [x for x in num_gpu_list if x > number_gpus]
        result = {"can_recommend": False}

        for min_gpus in num_gpu_list:
            configuration["batch_size"] = per_device_train_batch_size * min_gpus
            configuration["number_gpus"] = min_gpus

            moduleLog.debug(
                f"Trying configuration with {min_gpus} GPUs, batch_size={configuration['batch_size']}"
            )
            gpus_can_support_run, _ = get_model_prediction_and_metadata(
                configuration, predictor=predictor
            )

            if gpus_can_support_run < 1:
                moduleLog.debug(
                    f"Configuration with {min_gpus} GPUs would cause OOM, trying next"
                )
                continue

            workers = math.ceil(min_gpus / gpus_per_worker)
            gpus = math.ceil(min_gpus / workers)
            result["can_recommend"] = True
            result["gpus"] = gpus
            result["workers"] = workers
            moduleLog.debug(
                f"Found minimum configuration: workers:{workers} gpus:{gpus}"
            )
            break

        return result

    except Exception as e:
        moduleLog.error(f"Error while trying to execute recommender:{e}")
        result["can_recommend"] = False
        return result
