# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
from pathlib import Path

import ray

"""Uses the dolomite engine to convert weights into HF formats
"""


@ray.remote(
    resources={"Tesla-V100-PCIE-16GB": 1},
    runtime_env={
        "pip": [
            "accelerate",
            "transformers>=4.40.0",
            "git+https://github.com/ibm-granite/dolomite-engine",
        ],
        "env_vars": {
            "LOG_LEVEL": "debug",
            "LOGLEVEL": "debug",
            "HF_HOME": "/hf-models-pvc/huggingface_home",
        },
    },
)
def convert_weights(
    path_model: Path, destination: Path, model_type: str = "llama"
) -> None:
    import json

    model_config_file = path_model / Path("config.json")
    config = json.loads(model_config_file.read_text())

    print("Model type", config["model_type"])
    orig_model_type = config["model_type"]

    if config["model_type"] != "gpt_dolomite":
        config["model_type"] = "gpt_dolomite"

        model_config_file.write_text(json.dumps(config, indent=2))

    print("converting to format", model_type)
    try:
        if destination.is_dir():
            print("Path", destination, "already exists, will remove it")
            import shutil

            shutil.rmtree(destination, ignore_errors=True)

        from dolomite_engine.hf_models import export_to_huggingface

        export_to_huggingface(str(path_model), str(destination), model_type=model_type)
    finally:
        if orig_model_type != config["model_type"]:
            print("Restoring original config.json")
            config["model_type"] = orig_model_type
            model_config_file.write_text(json.dumps(config, indent=2))

    print(json.dumps(config, indent=2))


ray.init()

linear_layers = {}

all_models = [
    (
        "granite-8b-japanese",
        Path(
            "/hf-models-pvc/granite-8b-japanese-base-v1/granite-8b-japanese-base-v1-20240806T153614"
        ),
        Path("/hf-models-pvc/granite-8b-japanese-base-v1-llama/"),
        "llama",
    ),
]

for model_name, path_model, destination, model_type in all_models:
    print(model_name, path_model, destination, model_type)

    ret = ray.get(convert_weights.remote(path_model, destination, model_type))
