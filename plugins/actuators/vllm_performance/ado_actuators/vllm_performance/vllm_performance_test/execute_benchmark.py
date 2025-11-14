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


class VLLMBenchmarkError(Exception):
    """Raised if there was an issue when running the benchmark"""


def execute_benchmark(
    base_url: str,
    model: str,
    data_set: str,
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

    :raises VLLMBenchmarkError if the benchmark failed to execute after
        benchmark_retries attempts
    """
    logger = logging.getLogger("vllm-bench")

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
        f"vllm bench serve --backend openai --base-url {base_url} --dataset-name {data_set} "
        f"--model {model} --seed 12345 --num-prompts {num_prompts!s} --save-result --metric-percentiles "
        f'"25,75,99" --percentile-metrics "ttft,tpot,itl,e2el" --result-dir . --result-filename {f_name} '
        f"--burstiness {burstiness} "
    )

    if data_set_path is not None:
        request += f"--dataset-path {data_set_path} "
    if request_rate is not None:
        request += f"--request-rate {request_rate!s} "
    if max_concurrency is not None:
        request += f"--max-concurrency {max_concurrency!s} "
    if custom_args is not None:
        for key, value in custom_args.items():
            request += f"{key} {value!s} "
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
                raise VLLMBenchmarkError(f"Failed to execute benchmark {e}")

    try:
        retval = get_results(f_name=f_name)
    except VLLMBenchmarkResultReadError:
        raise VLLMBenchmarkError from VLLMBenchmarkResultReadError

    return retval


def execute_random_benchmark(
    base_url: str,
    model: str,
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
        data_set="random",
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


if __name__ == "__main__":
    results = execute_benchmark(
        interpreter="python3.10",
        base_url="http://localhost:28015",
        data_set="random",
        model="openai/gpt-oss-20b",
        request_rate=None,
        max_concurrency=None,
        hf_token=os.getenv("HF_TOKEN"),
        num_prompts=100,
    )
    print(results)
