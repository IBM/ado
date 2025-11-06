# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import os
import subprocess
import time
import uuid
from typing import Any

from ado_actuators.vllm_performance.vllm_performance_test.get_benchmark_results import (
    get_results,
)

logger = logging.getLogger("vllm-bench")

default_geospatial_datasets_filenames = {
    "india_url_in_b64_out": "india_url_in_b64_out.jsonl",
    "valencia_url_in_b64_out": "valencia_url_in_b64_out.jsonl",
}


def execute_benchmark(
    base_url: str,
    model: str,
    data_set: str,
    backend: str = "openai",
    interpreter: str = "python",
    num_prompts: int = 500,
    request_rate: int | None = None,
    max_concurrency: int | None = None,
    hf_token: str | None = None,
    benchmark_retries: int = 3,
    retries_timeout: int = 5,
    data_set_path: str | None = None,
    custom_args: dict[str, Any] | None = None,
    burstiness: float = 1,
) -> dict[str, Any]:
    """
    Execute benchmark
    :param base_url: url for vllm endpoint
    :param model: model
    :param data_set: data set name ["sharegpt", "sonnet", "random", "hf"]
    :param interpreter - name of Python interpreter
    :param num_prompts: number of prompts
    :param request_rate: request rate
    :param max_concurrency: max concurrency
    :param hf_token: huggingface token
    :param benchmark_retries: number of benchmark execution retries
    :param retries_timeout: timeout between initial retry
    :param data_set_path: path to the dataset
    :param custom_args: custom arguments to pass to the benchmark.
    keys are vllm benchmark arguments. values are the values to pass to the arguments
    :return: results dictionary
    """

    logger.debug(
        f"executing benchmark, invoking service at {base_url} with the parameters: "
    )
    logger.debug(
        f"model {model}, data set {data_set}, python {interpreter}, num prompts {num_prompts}"
    )
    logger.debug(
        f"request_rate {request_rate}, max_concurrency {max_concurrency}, benchmark retries {benchmark_retries}"
    )
    # The code below is commented as we are switching from a script invocation to command line
    # invocation. If we want to bring back script execution for any reason, this code must be
    # uncommented
    # parameters
    # code = os.path.abspath(
    #    os.path.join(os.path.dirname(__file__), "benchmark_serving.py")
    # )
    request = f"export HF_TOKEN={hf_token} && " if hf_token is not None else ""
    f_name = f"{uuid.uuid4().hex}.json"
    request += (
        # changing from script invocation to cli invocation
        # f"{interpreter} {code} --backend openai --base-url {base_url} --dataset-name {data_set} "
        f"vllm bench serve --backend {backend} --base-url {base_url} --dataset-name {data_set} "
        f"--model {model} --seed 12345 --num-prompts {num_prompts!s} --save-result --metric-percentiles "
        f'"25,75,99" --percentile-metrics "ttft,tpot,itl,e2el" --result-dir . --result-filename {f_name} '
        f"--burstiness {burstiness} "
    )

    if data_set_path is not None:
        request += f" --dataset-path {data_set_path} "
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
                time.sleep(timeout)
                timeout *= 2
            else:
                logger.warning("Failed to execute benchmark")
                raise Exception(f"Failed to execute benchmark {e}")

    return get_results(f_name=f_name)


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
    :param data_set: data set name ["sharegpt", "sonnet", "random", "hf"]
    :param hf_token: huggingface token
    :param benchmark_retries: number of benchmark execution retries
    :param retries_timeout: timeout between initial retry
    :param input_token_length: length of input tokens
    :param output_token_length: length of output tokens
    :return: results dictionary
    """
    # Call execute_benchmark with the appropriate arguments
    return execute_benchmark(
        base_url=base_url,
        model=model,
        data_set=dataset,
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
    :param data_set: data set name ["sharegpt", "sonnet", "random", "hf"]
    :param hf_token: huggingface token
    :param benchmark_retries: number of benchmark execution retries
    :param retries_timeout: timeout between initial retry
    :param input_token_length: length of input tokens
    :param output_token_length: length of output tokens
    :return: results dictionary
    """

    if dataset in default_geospatial_datasets_filenames:
        from pathlib import Path

        dataset_filename = default_geospatial_datasets_filenames[dataset]
        parent_path = Path(__file__).parents[1].absolute()
        data_set_path = os.path.join(parent_path, "datasets", dataset_filename)
    else:
        # This can only happen with the performance-testing-geospatial-full-custom-dataset
        # experiment, otherwise the dataset name is always one of the allowed ones.
        # Here the assumption is that the dataset file is placed in the  process working directory.
        ray_working_dir = os.getcwd()
        data_set_path = os.path.join(ray_working_dir, dataset)

    if not os.path.exists(data_set_path) or not os.path.isfile(data_set_path):
        logger.warning(
            f"The dataset filename provided does not exist or does not point to a valid file: {data_set_path}"
        )
        raise Exception(
            f"The dataset filename provided does not exist or does not point to a valid file: {data_set_path}"
        )

    logger.debug(f"Dataset path {data_set_path}")

    return execute_benchmark(
        base_url=base_url,
        backend="io-processor-plugin",
        model=model,
        data_set="custom",
        interpreter=interpreter,
        num_prompts=num_prompts,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        hf_token=hf_token,
        benchmark_retries=benchmark_retries,
        retries_timeout=retries_timeout,
        burstiness=burstiness,
        custom_args={
            "--dataset-path": data_set_path,
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
    )
    print(results)
