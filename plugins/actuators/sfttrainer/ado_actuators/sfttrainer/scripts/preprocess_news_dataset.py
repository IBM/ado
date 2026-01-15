# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""Preprocesses some of the artificial news dataset

File structure
- this_file.py
- common_en_news_combined_512.jsonl
- common_en_news_combined_1024.jsonl
- common_en_news_combined_512.jsonl
"""

# Standard
import os

# Third Party
import datasets


def format_alpaca_fn(example: dict[str, str]) -> dict[str, str]:
    prompt = (
        "Write a response that appropriately completes the remainder of the following input text.\n\n"
        "### Input:\n{input}\n\n### Response: {output}"
    )
    output = prompt.format_map(example)
    return {"output": output}


num_max_batch_size = 128
num_max_gpus = 8
num_gradient_accumulation = 4

enhanced_size = num_max_batch_size * num_max_gpus * num_gradient_accumulation

for num in [512, 1024, 2048]:
    path_dir = os.path.abspath(os.path.dirname(__file__))
    path_input = os.path.join(path_dir, f"common_en_news_combined_{num}.jsonl")

    path_output = os.path.join(
        path_dir, f"news-chars-{num}-entries-{enhanced_size}.jsonl"
    )

    ds: datasets.dataset_dict.DatasetDict = datasets.load_dataset(
        "json", data_files=path_input
    )

    alpaca_ds: datasets.arrow_dataset.Dataset = ds["train"].map(
        format_alpaca_fn, remove_columns=["input"]
    )

    x = alpaca_ds.to_list()

    replicated = [x[i % len(x)] for i in range(enhanced_size)]

    alpaca_ds = datasets.arrow_dataset.Dataset.from_list(replicated)

    alpaca_ds.to_json(path_output)
