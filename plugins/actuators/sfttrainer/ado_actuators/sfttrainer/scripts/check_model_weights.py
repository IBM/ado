# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import os
import sys
import traceback
import typing

import ray
from ado_actuators.sfttrainer.experiments.common import (
    ModelMap,
    WeightsFormat,
)

"""Checks whether models are load-able
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
def get_model_hash(path_model: str) -> bool:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    try:
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
            _model = AutoModelForCausalLM.from_config(model_config)
            _tokenizer = AutoTokenizer.from_pretrained(path_model)
    except Exception as e:
        print("Unable to load weights, due to", e)
        print(traceback.format_exc())
        return False

    return True


def main() -> None:
    ray.init()

    model_information: dict[str, dict[str, typing.Any]] = {}

    # VV: if you just added a model to `ModelMap` just copy/paste that entire dictionary here
    # ModelMap = { ..... }
    # (the version of the fms-hf-tuning Orchestrator Plugin on the ray cluster won't have the changes you just made)

    all_models = {
        k: v[WeightsFormat.Vanilla]
        for k, v in ModelMap.items()
        if WeightsFormat.Vanilla in v
    }

    for model_name, path_model in all_models.items():
        print(model_name, path_model)
        ret = ray.get(get_model_hash.remote(path_model))

        model_information[model_name] = ret
        print(model_name, "\n  ", json.dumps(ret))

        print("All so far")
        print(json.dumps(model_information, indent=2))

    missing = {k: False for k, v in model_information.items() if v is False}
    existing = {k: True for k, v in model_information.items() if v is True}

    print("Missing")
    print(json.dumps(missing, indent=2))

    print("Existing")
    print(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
