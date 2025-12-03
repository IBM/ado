# Copyright (c) IBM Corporation

# SPDX-License-Identifier: MIT

"""
Module for mapping between categorical variable values in the production data to the training data derived from benchmarking.
"""

# TODO(srikumarv): Need to make this formal somewhere so that it is resilient and adaptable to model and training data updates

import re

model_patterns = {
    "GRANITE_3_1_2B": r"^granite-(?:3\.[1-3]-)?2b(?:-(?:instruct|base))?(?:-|$)",
    "GRANITE_3_1_8B": r"^granite-(?:3\.[1-3]-)?8b(?:-(?:instruct|base))?$",
    "LLAMA_3_1_8B": r"llama-3.1-8b(?:-.*)?$",
    "GRANITE_4_SMALL": r"granite-4\.0(?:-h)?-(?:small)(?:-|$)",
    "GRANITE_4_TINY": r"granite-4\.0(?:-h)?-(?:tiny)(?:-|$)",
    "GRANITE_4_MICRO": r"granite-4\.0(?:-h)?-(?:micro)(?:-|$)",
}

# Models that we have chosen to map to

mapped_models = {
    "GRANITE_3_1_2B": "granite-3.1-2b",
    "GRANITE_3_1_8B": "granite-3.1-8b-instruct",
    "LLAMA_3_1_8B": "llama-3.1-8b",
    "GRANITE_4_SMALL": "granite-4.0-h-small",
    "GRANITE_4_TINY": "granite-4.0-h-tiny",
    "GRANITE_4_MICRO": "granite-4.0-h-micro",
}


def map_valid_model_name(model_name: str) -> str:

    mapped = [
        mapped_models[k] for k, v in model_patterns.items() if re.match(v, model_name)
    ]
    return model_name if len(mapped) == 0 else mapped[0]


def map_to_valid(input_dict: dict) -> dict:
    from copy import deepcopy

    mapped_dict = deepcopy(input_dict)
    key = "model_name"
    if key in mapped_dict:
        mapped_dict[key] = map_valid_model_name(mapped_dict[key])

    key = "gpu_model"
    if mapped_dict.get(key, None) == "NVIDIA A100-SXM4-80GB":
        mapped_dict[key] = "NVIDIA-A100-SXM4-80GB"

    return mapped_dict
