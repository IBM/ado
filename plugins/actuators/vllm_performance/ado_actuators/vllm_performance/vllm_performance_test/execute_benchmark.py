# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import os
import subprocess
import time
import uuid
from typing import Any

from ado_actuators.vllm_performance.vllm_performance_test.get_benchmark_results import (
    VLLMBenchmarkResultReadError,
    get_results,
)

logger = logging.getLogger("vllm-bench")

default_geospatial_datasets_filenames = {
    "india_url_in_b64_out": "india_url_in_b64_out.jsonl",
    "valencia_url_in_b64_out": "valencia_url_in_b64_out.jsonl",
}


class VLLMBenchmarkError(Exception):
    """Raised if there was an issue when running the benchmark"""


def execute_benchmark(
    base_url: str,
    model: str,
    dataset: str,
    backend: str = "openai",
    interpreter: str = "python",
    num_prompts: int = 500,
    request_rate: int | None = None,
    max_concurrency: int | None = None,
    hf_token: str | None = None,
    benchmark_retries: int = 3,
    retries_timeout: int = 5,
    dataset_path: str | None = None,
    custom_args: dict[str, Any] | None = None,
    burstiness: float = 1,
) -> dict[str, Any]:
    """
    Execute benchmark
    :param base_url: url for vllm endpoint
    :param model: model
    :param dataset: data set name ["random"]
    :param backend: name of the vLLM benchmark backend to be used ["vllm", "openai", "openai-chat", "openai-audio", "openai-embeddings"]
    :param interpreter: name of Python interpreter
    :param num_prompts: number of prompts
    :param request_rate: request rate
    :param max_concurrency: maximum number of concurrent requests
    :param hf_token: huggingface token
    :param benchmark_retries: number of benchmark execution retries
    :param retries_timeout: timeout between initial retry
    :param dataset_path: path to the dataset
    :param custom_args: custom arguments to pass to the benchmark.
    :param burstiness: burstiness factor of the request generation, 0 < burstiness < 1
    keys are vllm benchmark arguments. values are the values to pass to the arguments

    :return: results dictionary

    :raises VLLMBenchmarkError if the benchmark failed to execute after
        benchmark_retries attempts
    """

    logger.debug(
        f"executing benchmark, invoking service at {base_url} with the parameters: "
    )
    logger.debug(
        f"model {model}, data set {dataset}, python {interpreter}, num prompts {num_prompts}"
    )
    logger.debug(
        f"request_rate {request_rate}, max_concurrency {max_concurrency}, benchmark retries {benchmark_retries}"
    )

    # get logger level and forward to subprocess
    log_level = logging.getLevelName(logger.getEffectiveLevel())
    request = f"export HF_TOKEN={hf_token} && " if hf_token is not None else ""
    f_name = f"{uuid.uuid4().hex}.json"
    # Propagate logger's log level to subprocess via env var (if supported)
    request = f"export VLLM_BENCH_LOGLEVEL={log_level} && " + request
    request += (
        f"vllm bench serve --backend {backend} --base-url {base_url} --dataset-name {dataset} "
        f"--model {model} --seed 12345 --num-prompts {num_prompts!s} --save-result --metric-percentiles "
        f'"25,75,99" --percentile-metrics "ttft,tpot,itl,e2el" --result-dir . --result-filename {f_name} '
        f"--burstiness {burstiness} "
    )

    if dataset_path is not None:
        request += f" --dataset-path {dataset_path} "
    if request_rate is not None:
        request += f" --request-rate {request_rate!s} "
    if max_concurrency is not None:
        request += f"--max-concurrency {max_concurrency!s} "
    if custom_args is not None:
        for key, value in custom_args.items():
            request += f" {key} {value!s} "
    timeout = retries_timeout

    logger.debug(f"Command line: {request}")

    for i in range(benchmark_retries):
        try:
            subprocess.check_call(request, shell=True)
            break
        except subprocess.CalledProcessError as e:
            logger.warning(f"Command failed with return code {e.returncode}")
            if i < benchmark_retries - 1:
                logger.warning(
                    f"Will try again after {timeout} seconds. {benchmark_retries - 1 - i} retries remaining"
                )
                time.sleep(timeout)
                timeout *= 2
            else:
                logger.error(
                    f"Failed to execute benchmark after {benchmark_retries} attempts"
                )
                raise VLLMBenchmarkError(f"Failed to execute benchmark {e}") from e

    try:
        retval = get_results(f_name=f_name)
    except VLLMBenchmarkResultReadError:
        raise VLLMBenchmarkError from VLLMBenchmarkResultReadError

    return retval


def execute_random_benchmark(
    base_url: str,
    model: str,
    dataset: str,
    num_prompts: int = 500,
    request_rate: int | None = None,
    max_concurrency: int | None = None,
    hf_token: str | None = None,
    benchmark_retries: int = 3,
    retries_timeout: int = 5,
    burstiness: float = 1,
    number_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    interpreter: str = "python",
) -> dict[str, Any]:
    """
    Execute benchmark with random dataset
    :param base_url: url for vllm endpoint
    :param model: model
    :param dataset: data set name ["random"]
    :param num_prompts: number of prompts
    :param request_rate: request rate
    :param max_concurrency: maximum number of concurrent requests
    :param hf_token: huggingface token
    :param benchmark_retries: number of benchmark execution retries
    :param retries_timeout: timeout between initial retry
    :param burstiness: burstiness factor of the request generation, 0 < burstiness < 1
    :param number_input_tokens: maximum number of input tokens for each request,
    :param max_output_tokens: maximum number of output tokens for each request,
    :param interpreter: name of Python interpreter

    :return: results dictionary
    """
    # Call execute_benchmark with the appropriate arguments
    return execute_benchmark(
        base_url=base_url,
        model=model,
        dataset=dataset,
        interpreter=interpreter,
        num_prompts=num_prompts,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        hf_token=hf_token,
        benchmark_retries=benchmark_retries,
        retries_timeout=retries_timeout,
        burstiness=burstiness,
        custom_args={
            "--random-input-len": number_input_tokens,
            "--random-output-len": max_output_tokens,
        },
    )


def execute_geospatial_benchmark(
    base_url: str,
    model: str,
    dataset: str,
    num_prompts: int = 500,
    request_rate: int | None = None,
    max_concurrency: int | None = None,
    hf_token: str | None = None,
    benchmark_retries: int = 3,
    retries_timeout: int = 5,
    burstiness: float = 1,
    interpreter: str = "python",
) -> dict[str, Any]:
    """
    Execute benchmark with random dataset
    :param base_url: url for vllm endpoint
    :param model: model
    :param dataset: data set name ["random"]
    :param num_prompts: number of prompts
    :param request_rate: request rate
    :param max_concurrency: maximum number of concurrent requests
    :param hf_token: huggingface token
    :param benchmark_retries: number of benchmark execution retries
    :param retries_timeout: timeout between initial retry
    :param burstiness: burstiness factor of the request generation, 0 < burstiness < 1
    :param interpreter: python interpreter to use

    :return: results dictionary
    """
    from pathlib import Path

    if dataset in default_geospatial_datasets_filenames:
        dataset_filename = default_geospatial_datasets_filenames[dataset]
        parent_path = Path(__file__).parents[1]
        dataset_path = parent_path / "datasets" / dataset_filename
    else:
        # This can only happen with the performance-testing-geospatial-full-custom-dataset
        # experiment, otherwise the dataset name is always one of the allowed ones.
        # Here the assumption is that the dataset file is placed in the  process working directory.
        ray_working_dir = Path.cwd()
        dataset_path = ray_working_dir / dataset

    if not dataset_path.is_file():
        error_string = (
            "The dataset filename provided does not exist or "
            f"does not point to a valid file: {dataset_path}"
        )
        logger.warning(error_string)
        raise ValueError(error_string)

    logger.debug(f"Dataset path {dataset_path}")

    return execute_benchmark(
        base_url=base_url,
        backend="io-processor-plugin",
        model=model,
        dataset="custom",
        interpreter=interpreter,
        num_prompts=num_prompts,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        hf_token=hf_token,
        benchmark_retries=benchmark_retries,
        retries_timeout=retries_timeout,
        burstiness=burstiness,
        custom_args={
            "--dataset-path": f"{dataset_path.resolve()}",
            "--endpoint": "/pooling",
            "--skip-tokenizer-init": True,
        },
    )


if __name__ == "__main__":
    results = execute_geospatial_benchmark(
        interpreter="python3.10",
        base_url="http://localhost:8000",
        model="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        request_rate=2,
        max_concurrency=10,
        hf_token=os.getenv("HF_TOKEN"),
        num_prompts=100,
        dataset="india_url_in_b64_out",
    )
    print(results)
