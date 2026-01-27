# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import os

from .benchmark_models import BenchmarkResult


class VLLMBenchmarkResultReadError(Exception):
    """Raised if there was an issue reading benchmark results"""


def get_results(f_name: str = "random.json") -> BenchmarkResult:
    """
    Get benchmark results and validate with Pydantic model

    :param f_name: file containing results
    :return: BenchmarkResult instance
    """
    try:
        with open(f_name) as f:
            results = json.load(f)
        os.remove(f_name)
    except Exception as e:
        raise VLLMBenchmarkResultReadError(
            f"Failed to read benchmark result due to {e}"
        ) from e

    # Remove fields not needed for BenchmarkResult
    del results["date"]
    del results["endpoint_type"]
    del results["tokenizer_id"]
    del results["label"]

    # Validate and normalize using Pydantic model
    return BenchmarkResult.model_validate(results)


if __name__ == "__main__":
    get_results()
