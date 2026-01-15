# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import logging
import os

import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.finetune as finetune
import ray

logging.basicConfig(level=10)

"""Computes statistics about the number of tokens per entry in the dataset.
It also populates the num_tokens_cache_dir with cached files containing the tokens per entry of a dataset
based on the tokenizer of a model

example output:

{
    "news-tokens-16384plus-entries-4096": {
        "llama3.1-8b": {
            "min": 17238,
            "max": 31959,
            "avg": 22518.065185546875
        },
        "llama3.1-70b": {
            "min": 17238,
            "max": 31959,
            "avg": 22518.065185546875
        },
        "llama3.1-405b": {
            "min": 17238,
            "max": 31959,
            "avg": 22518.065185546875
        }
    },
    "news-tokens-128kplus-entries-4096": {
        "llama3.1-8b": {
            "min": 154084,
            "max": 154434,
            "avg": 154241.513671875
        },
        "llama3.1-70b": {
            "min": 154084,
            "max": 154434,
            "avg": 154241.513671875
        },
        "llama3.1-405b": {
            "min": 154084,
            "max": 154434,
            "avg": 154241.513671875
        }
    }
}
"""


@ray.remote(
    runtime_env={
        "env_vars": {
            "LOG_LEVEL": "debug",
            "LOGLEVEL": "debug",
            "HF_HOME": "/hf-models-pvc/huggingface_home",
        },
    },
)
def tokenize_text(
    path_model: str,
    path_data: str,
    model_id: str | None,
    num_tokens_cache_dir: str | None,
) -> dict[str, float | str | None]:
    num_tokens = finetune.get_tokens_per_sample_in_dataset(
        path_model=path_model,
        path_data=path_data,
        model_id=model_id,
        num_tokens_cache_dir=num_tokens_cache_dir,
        dataset_text_field="output",
    )

    sum_tokens = sum(num_tokens)

    ret = {
        "min": min(num_tokens),
        "max": max(num_tokens),
        "avg": sum_tokens / len(num_tokens),
        "ds_file": os.path.splitext(os.path.basename(path_data))[0],
        "model_id": model_id,
    }

    logger = logging.getLogger("sft_trainer")
    logger.info(json.dumps(ret))

    return ret


def main() -> None:
    ray.init()
    root_data = os.environ.get("DATA_PATH", "/data/fms-hf-tuning/artificial-dataset/")

    # {dataset_id: {model_id: sum_tokens}}
    dataset_sizes = {}

    dataset_files = [
        # "news-chars-512-entries-4096.jsonl",
        # "news-chars-1024-entries-4096.jsonl",
        # "news-chars-2048-entries-4096.jsonl",
        "news-tokens-16384plus-entries-4096.jsonl",
    ]

    large_dataset_files = {
        "news-tokens-16384plus-entries-4096.jsonl",
        "news-tokens-128kplus-entries-4096.jsonl",
    }

    small_models = {
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
    }

    large_models = {
        "llama3.1-8b": "/hf-models-pvc/LLaMa/models/hf/llama3.1-8b",
        "llama3.1-70b": "/hf-models-pvc/LLaMa/models/hf/llama3.1-70b",
        "llama3.1-405b": "/hf-models-pvc/LLaMa/models/hf/llama3.1-405b",
    }

    cache_dir = os.path.join(root_data, "cache")

    os.makedirs(cache_dir, exist_ok=True)

    tasks = []

    for dataset_collection, models in [
        (dataset_files, small_models),
        (dataset_sizes, large_models),
        (large_dataset_files, large_models),
    ]:
        for ds_file in dataset_collection:
            path_data = os.path.join(root_data, ds_file)

            for model_id, path_model in models.items():
                tasks.append(
                    tokenize_text.remote(
                        path_model=path_model,
                        path_data=path_data,
                        model_id=model_id,
                        num_tokens_cache_dir=cache_dir,
                    )
                )

    results = ray.get(tasks)

    for ret in results:
        if ret["ds_file"] not in dataset_sizes:
            dataset_sizes[ret["ds_file"]] = {}

        dataset_sizes[ret["ds_file"]][ret["model_id"]] = {
            k: v for k, v in ret.items() if k not in ("model_id", "ds_file")
        }

    print(json.dumps(dataset_sizes, indent=2))


if __name__ == "__main__":
    main()
