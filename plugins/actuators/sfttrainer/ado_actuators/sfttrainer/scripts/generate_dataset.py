# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""
# Generates a synthetic dataset from a seed file

## Documentation

The Seed file should be a human-readable text file. This script will parse the file and use it to build N entries of
a dataset. N is the product of num_max_batch_size * num_max_gpus * num_gradient_accumulation_steps

The motivation behind the formula of N is that the dataset should be large enough to support 1 full epoch for the
maximum considered: batch size, number of gpus, and number of gradient accumulation steps. If this property of the
dataset is violated, then the metrics that SFTTrainer from transformers produces can be inaccurate.

Each dataset entry is crafted such that:

1. it has about 10 tokens as its input (*)
2. it has about 18240 tokens (by default) as its output (*)
3. it follows the Alpaca format with an `input` and a `response` delimiter
4. it contains a stream of sequential words from the seed file (**)

## Caveats

- (*) The script assumes that each token is roughly 0.75 "words" (space separated text). The script keeps the count of
    the words it uses for each entry in the dataset and uses 0.75 words per token rule to estimate the number of
    tokens in that entry. This estimation is just a "good enough" guess. For example, different models may use
    different tokenizers and as a result they may produce a different number of tokens.
- (**) The script starts using the "words" from the beginning of the file and may wrap around the file if necessary.
    It keeps track of where it stopped so that different entries of the dataset do not all start from the exact same
    part of the seed file.

## Implications of the above

1. The synthetic dataset is dependent on the seed file you use. By default, the script uses itself as its seed file.
   My understanding is that 2 different seed files will produce "computationally equivalent" synthetic datasets.
   However, we have not formally verified this assumption.
2. When you load the synthetic dataset with your tokenizer, you will observe that the tokens for each entry in the
   dataset are actually different from what you requested. This is normal behaviour. We recommend asking for more
   tokens than you think you will need and controlling the actual tokens that your model trains on by setting the
   parameter --max_seq_length of fms-hf-tuning.
"""

import json
import logging
import os.path
import pathlib
import sys

import typer

app = typer.Typer(rich_markup_mode="markdown")


def generate_sequence(words: list[str], start: int, length: int) -> tuple[str, int]:
    ret = []
    idx = 0
    while idx < length:
        batch = min(length - idx, len(words) - start)

        ret.extend(words[start : start + batch])
        idx += batch

        start = (start + batch) % len(words)

    return " ".join(ret), start


def compute_population_size(
    num_max_batch_size=128, num_max_gpus=8, num_gradient_accumulation_steps=4
) -> int:
    return num_max_batch_size * num_max_gpus * num_gradient_accumulation_steps


def generate_dataset(
    seed_file: pathlib.Path,
    tokens_input: int = 10,
    tokens_output: int = 18240,
    words_per_token: float = 0.75,
    population: int = 4096,
    prompt: str = "### Input:\n",
    response_delimiter: str = "\n### Response:",
):
    with open(seed_file, encoding="utf-8") as f:
        words = [x for x in f.read().split() if len(x) > 0]

    dataset = []
    start = 0

    words_prompt = int(tokens_input * words_per_token + 0.5)
    words_response = int(tokens_output * words_per_token + 0.5)

    for _ in range(population):
        text_input, start = generate_sequence(words, start, words_prompt)
        text_output, start = generate_sequence(words, start, words_response)

        entry = {
            "output": " ".join((prompt, text_input, response_delimiter, text_output))
        }
        dataset.append(entry)

    return dataset


@app.command(
    no_args_is_help=True,
    help="Generates a synthetic dataset which should have at least @tokens_input+@tokens_output tokens in "
    "each entry. You can use the same dataset in experiment runs with different batch sizes, number of gpus, "
    "gradient accumulation steps. You just need to generate one dataset that contains enough entries "
    "for a full epoch of your largest experiment.",
    epilog=f"Examples\n\n"
    f"- python {sys.argv[0]} -o dataset.jsonl\n\n"
    f"- python {sys.argv[0]} -i path/to/some/file -o dataset.jsonl",
)
def main(
    output: pathlib.Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Where to store the dataset file.",
    ),
    input: pathlib.Path = typer.Option(
        pathlib.Path(__file__),
        "--input",
        "-i",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Input file to use as a source of sentences. For example you can use this python file, "
        "or some kind of publicly available document such as the Apache 2.0 License file. "
        "Smaller files should also work just as well.",
    ),
    tokens_input: int = typer.Option(
        10,
        help="How many tokens to have before the Response delimiter",
    ),
    tokens_output: int = typer.Option(
        18240,
        help="How many tokens to have after the Response delimiter",
    ),
    response_delimiter: str = typer.Option(
        "\n\n### Response:",
        help="The text to insert between the input and output tokens",
    ),
    prompt: str = typer.Option(
        "### Input:\n", help="The text to insert before the input tokens"
    ),
    words_per_token: float = typer.Option(
        0.75,
        help="How many tokens you expect to have per token - this is just an estimation",
    ),
    num_max_gpus: int = typer.Option(
        8, help="Maximum number of gpus in your experiment campaign"
    ),
    num_max_batch_size: int = typer.Option(
        128, help="Maximum batch size in your experiment campaign"
    ),
    num_gradient_accumulation_steps: int = typer.Option(
        4,
        help="Maximum num_gradient_accumulation_steps you will investigate with your experiment campaign",
    ),
    log_level: int = typer.Option(20, "--log-level", "-l", help="Log level"),
):
    logging.basicConfig(
        level=log_level,
        format="%(levelname)-9s %(name)-30s: %(funcName)-20s %(asctime)-15s: %(message)s",
    )
    population = compute_population_size(
        num_max_gpus=num_max_gpus,
        num_max_batch_size=num_max_batch_size,
        num_gradient_accumulation_steps=num_gradient_accumulation_steps,
    )
    logging.info(f"Generating the dataset with {population} entries")
    ds = generate_dataset(
        input,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        words_per_token=words_per_token,
        population=population,
        prompt=prompt,
        response_delimiter=response_delimiter,
    )

    logging.info(f"Saving file under {output}")
    if not os.path.isdir(output.parent.as_posix()):
        os.makedirs(output.parent.as_posix(), exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for entry in ds:
            json.dump(entry, f)
            f.write("\n")

    logging.info("Done")


if __name__ == "__main__":
    app()
