# Copyright (c) IBM Corporation

# SPDX-License-Identifier: MIT

from autoconf.utils.config_mapper import (
    map_valid_model_name,
    mapped_models,
)

invalid_model_names_transform_to_valid = {
    "granite-2b-base": mapped_models["GRANITE_3_1_2B"],
    "granite-3.2-8b-instruct": mapped_models["GRANITE_3_1_8B"],
    "granite-4.0-tiny": mapped_models["GRANITE_4_TINY"],
    "granite-4.0-small-base-prerelease-greylock": mapped_models["GRANITE_4_SMALL"],
    "llama-3-1-8b-instruct": mapped_models["LLAMA_3_1_8B"],
}

invalid_model_names_no_transform = {
    "gb_tuned_model_fw0u3wim_checkpoint-32637": "gb_tuned_model_fw0u3wim_checkpoint-32637",
    "granite-8b-code-instruct-128k": "granite-8b-code-instruct-128k",
    "devstral-small-182a9e3_24b": "devstral-small-182a9e3_24b",
    "granite-5.0-d453da": "granite-5.0-d453da",
}

valid_model_names_no_transform = {
    mapped_models["LLAMA_3_1_8B"]: mapped_models["LLAMA_3_1_8B"],
    mapped_models["GRANITE_4_TINY"]: mapped_models["GRANITE_4_TINY"],
    mapped_models["GRANITE_4_MICRO"]: mapped_models["GRANITE_4_MICRO"],
    mapped_models["GRANITE_3_1_8B"]: mapped_models["GRANITE_3_1_8B"],
}


def test_mapping_invalid_to_valid() -> None:
    assert all(
        map_valid_model_name(k) == v
        for k, v in invalid_model_names_transform_to_valid.items()
    )


def test_mapping_invalid_no_transform() -> None:
    assert all(
        map_valid_model_name(k) == v
        for k, v in invalid_model_names_no_transform.items()
    )


def test_mapping_valid_no_transform() -> None:
    assert all(
        map_valid_model_name(k) == v for k, v in valid_model_names_no_transform.items()
    )
