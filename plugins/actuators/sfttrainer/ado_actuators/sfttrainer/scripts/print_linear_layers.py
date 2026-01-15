# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import os
import sys

import ray

# VV: You can import these if you also upload the sfttrainer actuator pip wheel,
# but it's not really necessary
# from ado_actuators.sfttrainer.experiments.common import (
#     WeightsFormat,
#     ModelMap,
# )

"""Prints the linear layers in a model
"""


@ray.remote(
    runtime_env={
        "pip": ["accelerate", "transformers>=4.40.0"],
        "env_vars": {
            "LOG_LEVEL": "debug",
            "LOGLEVEL": "debug",
            "HF_HOME": "/data/transformers_cache",
        },
    },
)
def get_linear_layers(path_model: str) -> set[str]:
    from accelerate import init_empty_weights
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
    )

    with init_empty_weights():
        if os.path.exists(path_model) is False:
            import huggingface_hub

            print(
                "MODEL MAY NOT EXIST WILL TRY TO DOWNLOAD IT FROM HUGGINGFACE",
                file=sys.stderr,
            )
            huggingface_hub.snapshot_download(path_model)

        with init_empty_weights():
            model_config = AutoConfig.from_pretrained(
                path_model, local_files_only=True, low_cpu_mem_usage=True
            )
            try:
                model = AutoModelForCausalLM.from_config(model_config)
            except ValueError as e:
                from transformers import AutoModelForImageTextToText

                if "Unrecognized configuration class" in str(e):
                    model = AutoModelForImageTextToText.from_config(model_config)

    # to get just linear layers
    import re

    model_modules = str(model.modules)
    pattern = r"\((\w+)\): Linear"
    linear_layer_names = re.findall(pattern, model_modules)
    return sorted(set(linear_layer_names))


ray.init()

linear_layers = {}

# VV: if you just added a model to `ModelMap` just copy/paste that entire dictionary here
# ModelMap = { ..... }
# (the version of the fms-hf-tuning Orchestrator Plugin on the ray cluster won't have the changes you just made)

ModelMap: dict[str, dict[str, str]] = {
    "llava-v1.6-mistral-7b": {
        "Vanilla": "llava-hf/llava-v1.6-mistral-7b-hf",
    }
}

all_models = {k: v["Vanilla"] for k, v in ModelMap.items() if "Vanilla" in v}

for model_name, path_model in all_models.items():
    print(model_name, path_model)
    ret = ray.get(get_linear_layers.remote(path_model))

    linear_layers[model_name] = ret
    print(model_name, "\n  ", json.dumps(ret))

    print("All so far")
    print(json.dumps(linear_layers, indent=2))
