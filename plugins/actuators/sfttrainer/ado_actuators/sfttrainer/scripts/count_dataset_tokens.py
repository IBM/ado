# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import logging
import os
import time
import typing

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


def _update_num_tokens_cache_for_model_and_dataset(
    cache_file: str,
    num_tokens: typing.List[int],
    model_id: str,
    path_data: str,
):
    import json

    logger = logging.getLogger("sft_trainer:cache")

    # VV: We know that we'd like to use the cache and that we could not find
    # useful data in it. Therefore, we populate the relevant cache file here
    try:
        for _ in range(5):
            with open(cache_file, "w") as f:
                json.dump(num_tokens, f)
            # VV: Verify that we actually stored what we think we stored (there could be multiple
            # tasks populating the cache and them corrupting each other's results)
            with open(cache_file, "r") as f:
                fresh = json.load(f)

            if fresh == num_tokens:
                logger.info(f"Populated the cache file {cache_file} successfully")
                break
            logger.warning(
                f"The cache file {cache_file} is corrupted, will try to recreate"
            )
            time.sleep(5)

    except Exception as e:
        logger.warning(
            f"Could not cache the num tokens using the tokenizer of {model_id} for dataset {path_data} due to {e}"
        )


def _load_num_tokens_cache_for_model_and_dataset(
    path_data: str,
    model_id: typing.Optional[str],
    num_tokens_cache_dir: typing.Optional[str],
) -> typing.Tuple[typing.Optional[str], typing.List[int]]:
    import json

    num_tokens = []
    cache_file = None

    logger = logging.getLogger("sft_trainer:cache")

    try:
        os.makedirs(num_tokens_cache_dir, exist_ok=True)

        # VV: since we may update the contents of a dataset
        # we use the md5 hash of the file as part of the cache id
        import hashlib

        digest = hashlib.md5()

        with open(path_data, "rb") as f:
            b = f.read(32768)
            while b:
                digest.update(b)
                b = f.read(32768)

        ds_name = os.path.splitext(os.path.basename(path_data))[0]
        cache_file = os.path.join(
            num_tokens_cache_dir,
            ".".join(
                (
                    "num-tokens",
                    model_id,
                    "for",
                    ds_name,
                    digest.hexdigest(),
                    "json",
                )
            ),
        )

        with open(cache_file, "rb") as f:
            num_tokens = json.load(f)
            if isinstance(num_tokens, list) is False:
                raise NotImplementedError(
                    f"Unknown type of num_tokens {type(num_tokens)}"
                )

        logger.info(
            f"Loaded cached num_tokens with tokenizer {model_id} and dataset {path_data}"
        )
    except FileNotFoundError:
        logger.info(
            f"No cached number of tokens with tokenizer {model_id} and dataset {path_data}"
        )
    except Exception as e:
        logger.warning(
            f"Could not parse the cached num tokens due to {e} - will compute number of tokens using "
            f"the tokenizer of {model_id} for dataset {path_data}"
        )

    return cache_file, num_tokens


def _get_tokens_of_dataset_entries(
    path_model: str,
    path_data: str,
    model_id: typing.Optional[str],
    num_tokens_cache_dir: typing.Optional[str],
) -> typing.List[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path_model)
    special_tokens_dict = {}
    logger = logging.getLogger("sft_trainer")

    DEFAULT_PAD_TOKEN = "<PAD>"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    if tokenizer.pad_token is None:
        logger.warning("PAD token set to default, missing in tokenizer")
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        logger.warning("EOS token set to default, missing in tokenizer")
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        logger.warning("BOS token set to default, missing in tokenizer")
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        logger.warning("UNK token set to default, missing in tokenizer")
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    import json

    # VV: It can be **pretty** expensive to tokenize the input dataset so we use a cache
    # to store the number of tokens per entry of a dataset based on the tokenizer of a model
    num_tokens = []
    cache_file = None

    import time

    if num_tokens_cache_dir and model_id:
        start = time.time()
        cache_file, num_tokens = _load_num_tokens_cache_for_model_and_dataset(
            path_data=path_data,
            model_id=model_id,
            num_tokens_cache_dir=num_tokens_cache_dir,
        )
        logger.info(
            f"It took {time.time() - start} seconds to search/load the num_tokens cache"
        )

    generated_tokens = False

    if not num_tokens:
        if cache_file is not None:
            logger.info(
                "Will tokenize the dataset and cache the results to speedup future measurements"
            )
        else:
            logger.info(
                "Will tokenize the dataset but the cache is disabled, future measurements will "
                "also tokenize the dataset"
            )

        # VV: Either the cache was empty, not found, or contained invalid data
        start = time.time()
        with open(path_data, "r") as f:
            for line in f:
                data = json.loads(line)
                decoded = tokenizer.encode(data["output"], padding=True)
                num_tokens.append(len(decoded))
                del decoded

        logger.debug(
            f"It took {time.time() - start} seconds to tokenize the dataset {path_data} "
            f"with the tokenizer of {model_id}"
        )
        generated_tokens = True

    if (
        num_tokens_cache_dir
        and model_id
        and cache_file is not None
        and generated_tokens
    ):
        _update_num_tokens_cache_for_model_and_dataset(
            cache_file=cache_file,
            num_tokens=num_tokens,
            model_id=model_id,
            path_data=path_data,
        )
    else:
        logger.info(f"Will not cache the tokens {cache_file} {model_id}")

    return num_tokens


@ray.remote(
    resources={"Tesla-V100-PCIE-16GB": 1},
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
    model_id: typing.Optional[str],
    num_tokens_cache_dir: typing.Optional[str],
):
    num_tokens = _get_tokens_of_dataset_entries(
        path_model=path_model,
        path_data=path_data,
        model_id=model_id,
        num_tokens_cache_dir=num_tokens_cache_dir,
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


def main():
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
