# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import os
import pathlib
from typing import Annotated

import ray
import typer
import yaml

app = typer.Typer(rich_markup_mode="markdown")


@ray.remote(
    runtime_env={
        "pip": ["accelerate", "transformers>=4.40.0"],
        "env_vars": {
            "LOG_LEVEL": "debug",
            "LOGLEVEL": "debug",
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        },
    },
)
def download_weights(path_model: str, hf_home: pathlib.Path) -> None:
    if os.path.isabs(path_model):
        print("Skipping download - model is stored locally")
        return

    os.makedirs(hf_home, exist_ok=True)

    import huggingface_hub

    huggingface_hub.snapshot_download(repo_id=path_model, cache_dir=hf_home / "hub")


@app.command(
    no_args_is_help=True,
    help="Caches HuggingFace model weights locally",
)
def main(
    path_to_models: Annotated[
        pathlib.Path,
        typer.Option(
            "--input",
            "-i",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Path to YAML file containing the models dictionary",
        ),
    ],
    hf_home: Annotated[
        pathlib.Path,
        typer.Option(
            "--hf_home",
            "-o",
            file_okay=False,
            dir_okay=True,
            help="The path to use as the HuggingFace cache home",
        ),
    ] = pathlib.Path(__file__),
) -> None:
    ray.init()

    """Keys are the names of models, values are dictionaries with keys indicating the type of the model weight,
    and values the HuggingFace identifier strings.

    Example:

    smollm2-135m:
        Vanilla: HuggingFaceTB/SmolLM2-135M
    """

    with open(path_to_models) as f:
        model_map: dict[str, dict[str, str]] = yaml.safe_load(f)

    for _model_name, items in model_map.items():  # noqa: PERF102
        for _model_type, model_path in items.items():  # noqa: PERF102
            print("Downloading", model_path)
            try:
                ray.get(download_weights.remote(model_path, hf_home))
                print("Success")
            except Exception as e:
                print(f"Unable to download weights due to {e} - ignoring error")


if __name__ == "__main__":
    app()
