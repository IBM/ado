# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""Uses the artificial dataset containing input and output strings consisting of 512 characters to produce
a dataset whose input string is a dataset of 512 characters and output contains (512+2) * 9 + 512 characters.
Then it converts this dataset to the alpaca format.

File structure
- this_file.py
- common_en_news_combined_1024.jsonl
"""

# Standard
import os

# Third Party
import datasets


def format_alpaca_fn(example: dict[str, str]) -> dict[str, str]:
    ex_input = example["input"]
    ex_output = ". ".join([example["output"] for _ in range(100)])

    return {"output": f"### Input:\n{ex_input}\n\n### Response: {ex_output}"}


num_max_batch_size = 128
num_max_gpus = 8
num_gradient_accumulation = 4

enhanced_size = num_max_batch_size * num_max_gpus * num_gradient_accumulation


path_dir = os.path.abspath(os.path.dirname(__file__))
path_input = os.path.join(path_dir, "common_en_news_combined_1024.jsonl")

path_output = os.path.join(
    path_dir, f"news-tokens-16384plus-entries-{enhanced_size}.jsonl"
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
