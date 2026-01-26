# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import os
from typing import Any

from .benchmark_models import BenchmarkResult


class VLLMBenchmarkResultReadError(Exception):
    """Raised if there was an issue reading benchmark results"""


def get_results(f_name: str = "random.json") -> dict[str, Any]:
    """
    Get benchmark results and validate with Pydantic model

    :param f_name: file containing results
    :return: results dictionary (from BenchmarkResult.model_dump())
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
    # The CLI invocation does not return the same dict as the script invocation
    # so the following lines had to be commented
    # ---- uncomment if directly invoking script
    # del results["backend"]
    # del results["best_of"]
    # del results["request_goodput:"]
    # del results["input_lens"]
    # del results["output_lens"]
    # del results["ttfts"]
    # del results["itls"]
    # del results["generated_texts"]
    # del results["errors"]

    # Validate and normalize using Pydantic model
    benchmark_result = BenchmarkResult.model_validate(results)
    return benchmark_result.model_dump()


if __name__ == "__main__":
    get_results()
