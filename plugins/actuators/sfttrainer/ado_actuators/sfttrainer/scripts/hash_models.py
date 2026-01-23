# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import hashlib
import json
import os
import sys
import typing

import ray

"""Discovers models which share the same model configuration
"""


@ray.remote(
    runtime_env={
        "pip": ["accelerate", "transformers>=4.40.0"],
        "env_vars": {
            "LOG_LEVEL": "debug",
            "LOGLEVEL": "debug",
            "HF_HOME": "/hf-models-pvc/huggingface_home",
        },
    },
)
def get_model_hash(path_model: str) -> dict[str, str | int | list[str]]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    with init_empty_weights():
        if os.path.exists(path_model) is False:
            import huggingface_hub

            print(
                "MODEL MAY NOT EXIST WILL TRY TO DOWNLOAD IT FROM HUGGINGFACE",
                file=sys.stderr,
            )
            huggingface_hub.snapshot_download(path_model)

        model_config = AutoConfig.from_pretrained(
            path_model, local_files_only=True, low_cpu_mem_usage=True
        )
        model = AutoModelForCausalLM.from_config(model_config)
        tokenizer = AutoTokenizer.from_pretrained(path_model)

    hash_info = []

    for x in model.children():
        for name, value in x.named_parameters():
            hash_info.append([name, value.numel(), list(value.size())])

    hash_info = sorted(hash_info, key=lambda i: i[0])
    architectures = model.config.to_dict().get("architectures")
    num_parameters = sum(m[1] for m in hash_info)
    hash_info.append(["architectures", architectures])

    the_hash = hashlib.md5(
        str(hash_info).encode("utf-8"), usedforsecurity=False
    ).hexdigest()

    print(
        path_model,
        "Architectures",
        architectures,
        "parameters",
        num_parameters,
        "hash",
        the_hash,
    )

    return {
        "hash": the_hash,
        "num_parameters": num_parameters,
        "architectures": architectures,
        "tokenizer_model_max_length": tokenizer.model_max_length,
    }


def main() -> None:
    ray.init()

    model_information: dict[str, dict[str, typing.Any]] = {}

    all_models = {
        "llama-7b": "/hf-models-pvc/LLaMa/models/hf/7B",
        "granite-13b-v2": "/hf-models-pvc/granite-13b-base-v2/step_300000_ckpt",
        "llama-13b": "/hf-models-pvc/LLaMa/models/hf/13B",
        "granite-20b-v2": "/hf-models-pvc/granite-20b-code-base-v2/step_280000_ckpt/",
        "granite-7b-base": "ibm-granite/granite-7b-base",
        "granite-8b-japanese": "/hf-models-pvc/granite-8b-japanese-base-v1-llama/",
        "granite-8b-code-base": "/hf-models-pvc/granite-8b-code-base/",
        "granite-34b-code-base": "/hf-models-pvc/granite-34b-code-base/",
        "mistral-7b-v0.1": "/hf-models-pvc/mistralai-mistral-7b-v0.1",
        "llama3-8b": "/hf-models-pvc/LLaMa/models/hf/llama3-8b",
        "llama3-70b": "/hf-models-pvc/LLaMa/models/hf/llama3-70b/",
        "mixtral-8x7b-instruct-v0.1": "/hf-models-pvc/Mixtral-8x7B-Instruct-v0.1/",
        "llama2-70b": "/hf-models-pvc/LLaMa/models/hf/llama2-70b/",
        "llama3.1-8b": "/hf-models-pvc/LLaMa/models/hf/llama3.1-8b",
        "llama3.1-70b": "/hf-models-pvc/LLaMa/models/hf/llama3.1-70b",
        "llama3.1-405b": "/hf-models-pvc/LLaMa/models/hf/llama3.1-405b",
        "granite-3b-code-base-128k": "ibm-granite/granite-3b-code-base-128k",
        "granite-8b-code-base-128k": "ibm-granite/granite-8b-code-base-128k",
    }

    for model_name, path_model in all_models.items():
        print(model_name, path_model)
        ret = ray.get(get_model_hash.remote(path_model))

        model_information[model_name] = ret
        print(model_name, "\n  ", json.dumps(ret))

        print("All so far")
        print(json.dumps(model_information, indent=2))

    unique_hashes: dict[str, list[str]] = {}

    for model_name, info in model_information.items():
        if info["hash"] not in unique_hashes:
            unique_hashes[info["hash"]] = []

        unique_hashes[info["hash"]].append(model_name)

    all_unique = True
    for model_names in unique_hashes.values():
        if len(model_names) > 1:
            print("These models are equivalent:", model_names)
            all_unique = False

    if all_unique:
        print("All models have unique hashes, they differ from each other")


if __name__ == "__main__":
    main()
