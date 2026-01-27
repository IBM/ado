# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import os
import subprocess
import time
import uuid
from typing import Any

from ado_actuators.vllm_performance.vllm_performance_test.benchmark_models import (
    BenchmarkResult,
)
from ado_actuators.vllm_performance.vllm_performance_test.get_benchmark_results import (
    VLLMBenchmarkResultReadError,
    get_results,
)
from pydantic import HttpUrl, TypeAdapter

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
) -> BenchmarkResult:
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

    :return: BenchmarkResult instance

    :raises VLLMBenchmarkError if the benchmark failed to execute after
        benchmark_retries attempts
    :raises ValueError: If base_url is not a valid URL format
    """

    # Validate URL format using Pydantic
    try:
        url_adapter = TypeAdapter(HttpUrl)
        url_adapter.validate_python(base_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {base_url}") from e

    logger.debug(
        f"executing benchmark, invoking service at {base_url} with the parameters: "
    )
    logger.debug(
        f"model {model}, data set {dataset}, python {interpreter}, num prompts {num_prompts}"
    )
    logger.debug(
        f"request_rate {request_rate}, max_concurrency {max_concurrency}, benchmark retries {benchmark_retries}"
    )

    # Copy over the environment and add required variables for the benchmark
    env = dict(os.environ)
    env["VLLM_BENCH_LOGLEVEL"] = logging.getLevelName(logger.getEffectiveLevel())

    if hf_token is not None:
        env["HF_TOKEN"] = hf_token

    # Output to a random file name
    f_name = f"{uuid.uuid4().hex}.json"

    # Build the vllm bench serve command
    command = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        backend,
        "--base-url",
        base_url,
        "--dataset-name",
        dataset,
        "--model",
        model,
        "--seed",
        "12345",
        "--num-prompts",
        f"{num_prompts!s}",
        "--save-result",
        "--metric-percentiles",
        "25,75,99",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--result-dir",
        ".",
        "--result-filename",
        f_name,
        "--burstiness",
        f"{burstiness!s}",
    ]

    if dataset_path is not None:
        command.extend(["--dataset-path", dataset_path])
    if request_rate is not None:
        command.extend(["--request-rate", f"{request_rate!s}"])
    if max_concurrency is not None:
        command.extend(["--max-concurrency", f"{max_concurrency!s}"])
    if custom_args is not None:
        for key, value in custom_args.items():
            command.extend([key, f"{value!s}"])

    logger.debug(f"Command line: {command}")

    timeout = retries_timeout
    for i in range(benchmark_retries):
        try:
            subprocess.check_call(command, env=env)  # noqa: S603
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
) -> BenchmarkResult:
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

    :return: BenchmarkResult instance
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
) -> BenchmarkResult:
    """
    Execute benchmark with geospatial dataset
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

    :return: BenchmarkResult instance
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
