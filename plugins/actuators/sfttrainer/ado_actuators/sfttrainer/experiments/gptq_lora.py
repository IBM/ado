# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import copy
import typing

from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum

from . import common

if typing.TYPE_CHECKING:
    from orchestrator.modules.actuators.catalog import ExperimentCatalog
default_target_modules = {
    "granite-7b-base": ["q_proj", "v_proj"],
    "granite-8b-code-instruct": ["q_proj", "v_proj"],
    "granite-20b-v2": ["c_attn", "c_proj"],
    "granite-34b-code-base": ["c_attn", "c_proj"],
    "llama-7b": ["q_proj", "k_proj"],
    "llama3-70b": ["q_proj", "v_proj"],
    "llama3.1-405b": ["q_proj", "v_proj"],
    "mistral-7b-v0.1": ["q_proj", "v_proj"],
    "mixtral-8x7b-instruct-v0.1": ["q_proj", "v_proj"],
    "allam-1-13b": ["q_proj", "v_proj"],
}


hardcoded_parameters: dict[str, typing.Any] = {
    "peft_method": "lora",
    "target_modules_map": default_target_modules,
    "purpose": common.ExperimentPurpose.Performance,
    "weights_format": common.WeightsFormat.GPTQQuantized,
    # VV: GPTQ-LoRA variables
    "auto_gptq": "triton_v2",
    "fp16": "yes",
    # VV: base_layer, fused_lora
    "fused_lora": ["auto_gptq", "True"],
}


def add_experiments(catalog: "ExperimentCatalog") -> None:

    # VV: GTPQ-LoRA has been known to work before support for padding_free
    # "padding_free": ["huggingface"],
    method = "gptq-lora"
    version = "1.0.0"
    exp_name = f"finetune_{method}_benchmark"

    description = (
        "Measures the performance of GPTQ-LORA tuning a model for a given "
        "(GPU model, number GPUS, batch_size, model_max_length, number nodes) combination."
    )

    # VV: Here configure any propertyDomains which differ from the default ones in EntitySpace
    override_propertydomains = {
        "torch_dtype": PropertyDomain(
            values=["float16"],
            variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        ),
    }

    default_params = copy.deepcopy(common.DEFAULT_PARAMS)
    # VV: GPTQ-Lora has a different (and only) option for torch_dtype
    default_params["torch_dtype"] = "float16"
    default_params.update(
        {
            "r": 4,
            "lora_alpha": 16,
            # VV: fast_loss, fast_rms_layernorm, fast_rope_embeddings
            "fast_kernels": ["True", "True", "True"],
        }
    )
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
