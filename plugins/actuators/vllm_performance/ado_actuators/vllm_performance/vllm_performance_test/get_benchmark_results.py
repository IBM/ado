# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import os
from typing import Any


def get_results(f_name: str = "random.json") -> dict[str, Any]:
    """
    Get benchmark results
    :param f_name: file containing results
    :return: results dictionary
    """
    try:
        with open(f_name) as f:
            results = json.load(f)
        os.remove(f_name)
    except Exception as e:
        print(f"Failed to read benchmark result {e}")
        raise Exception(f"Failed to read benchmark result {e}")
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
    return results


if __name__ == "__main__":
    get_results()
