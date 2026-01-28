# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import copy
import typing

from . import common

if typing.TYPE_CHECKING:
    from orchestrator.modules.actuators.catalog import ExperimentCatalog
default_target_modules = {
    "smollm2-135m": ["q_proj", "v_proj"],
    "granite-3b-code-base-128k": ["q_proj", "v_proj"],
    "granite-7b-base": ["q_proj", "v_proj"],
    "granite-8b-code-base-128k": ["q_proj", "v_proj"],
    "granite-8b-code-base": ["q_proj", "v_proj"],
    "granite-8b-japanese": ["q_proj", "v_proj"],
    "granite-13b-v2": ["c_attn", "c_proj"],
    "granite-20b-v2": ["c_attn", "c_proj"],
    "granite-34b-code-base": ["c_attn", "c_proj"],
    "llama-7b": ["q_proj", "k_proj"],
    "llama-13b": ["q_proj", "k_proj"],
    "llama2-70b": ["q_proj", "v_proj"],
    "llama3-8b": ["q_proj", "k_proj"],
    "llama3-70b": ["q_proj", "v_proj"],
    "llama3.1-8b": ["q_proj", "v_proj"],
    "llama3.2-1b": ["q_proj", "v_proj"],
    "llama3.2-3b": ["q_proj", "v_proj"],
    "llama3.1-70b": ["q_proj", "v_proj"],
    "llama3.1-405b": ["q_proj", "v_proj"],
    "allam-1-13b": ["q_proj", "v_proj"],
    "hf-tiny-model-private/tiny-random-BloomForCausalLM": [
        "dense_h_to_4h",
        "dense_4h_to_4h",
    ],
    "mistral-7b-v0.1": ["q_proj", "v_proj"],
    "mistral-123b-v2": ["q_proj", "v_proj"],
    "mixtral-8x7b-instruct-v0.1": ["q_proj", "v_proj"],
    "granite-3-8b": ["q_proj", "v_proj"],
    "granite-3.1-2b": ["q_proj", "v_proj"],
    "granite-3.1-8b-instruct": ["q_proj", "v_proj"],
    "granite-4.0-micro": ["q_proj", "v_proj"],
    "granite-4.0-h-1b": ["q_proj", "v_proj"],
    "granite-4.0-350m": ["q_proj", "v_proj"],
    "granite-4.0-h-small": ["q_proj", "v_proj"],
    "granite-4.0-h-micro": ["q_proj", "v_proj"],
    "granite-4.0-h-tiny": ["q_proj", "v_proj"],
    "granite-3.0-1b-a400m-base": ["q_proj", "v_proj"],
    "granite-3.1-3b-a800m-instruct": ["q_proj", "v_proj"],
    "granite-vision-3.2-2b": ["q_proj", "v_proj"],
    "llava-v1.6-mistral-7b": ["q_proj", "v_proj"],
    "granite-3.3-8b": ["q_proj", "v_proj"],
}


def add_experiments(catalog: "ExperimentCatalog") -> None:

    method = "lora"
    version = "1.0.0"
    exp_name = f"finetune_{method}_benchmark"

    description = (
        "Measures the performance of LORA tuning a model for a given "
        "(GPU model, number GPUS, batch_size, model_max_length, number nodes) combination."
    )

    hardcoded_parameters: dict[str, typing.Any] = {
        "peft_method": method,
        "weights_format": common.WeightsFormat.Vanilla,
        "target_modules_map": default_target_modules,
        "purpose": common.ExperimentPurpose.Performance,
    }

    # VV: an EntityBase can represent many different kind of measurements.
    # This experiment is ONLY for LORA measurements
    override_propertydomains = {}

    default_params = copy.deepcopy(common.DEFAULT_PARAMS)
    default_params.update({"r": 4, "lora_alpha": 16})
    param_experiment = common.generate_parameterisable_finetune_experiment(
        hardcoded_parameters=hardcoded_parameters,
        default_params=default_params,
        override_propertydomains=override_propertydomains,
        version=version,
        method=method,
        description=description,
        exp_identifier=f"{exp_name}-v{version}",
        exp_name=exp_name,
        actuator_identifier=common.ACTUATOR_IDENTIFIER,
        fms_hf_tuning_versions=[".".join([str(d) for d in v]) for v in common.semvers],
        required_property_names=common.MINIMUM_PROPS,
    )

    catalog.addExperiment(param_experiment)
