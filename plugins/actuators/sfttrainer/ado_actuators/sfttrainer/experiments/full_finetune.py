# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import copy
import typing

from . import common

if typing.TYPE_CHECKING:
    from orchestrator.modules.actuators.catalog import ExperimentCatalog


def add_full_experiments(
    catalog: "ExperimentCatalog",
) -> None:
    method = "full"
    version = "1.0.0"
    exp_name = f"finetune_{method}_benchmark"

    description = (
        "Measures the performance of full-finetuning a model for a given "
        "(GPU model, number GPUS, batch_size, model_max_length, number nodes) combination."
    )

    hardcoded_parameters: dict[str, typing.Any] = {
        "peft_method": method,
        "weights_format": common.WeightsFormat.Vanilla,
        "purpose": common.ExperimentPurpose.Performance,
    }

    # VV: Here configure any propertyDomains which differ from the default ones in EntitySpace
    override_propertydomains = {}

    default_params = copy.deepcopy(common.DEFAULT_PARAMS)
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


def add_full_stability_experiments(catalog: "ExperimentCatalog") -> None:

    method = "full"
    version = "1.0.0"
    purpose = common.ExperimentPurpose.Stability
    exp_name = f"finetune_{method}_{purpose.lower()}"

    description = (
        "Performs 5 full finetune runs of 5 steps each on a model "
        "and reports the fraction of those that resulted in GPU OOM, Other error, "
        "or No Error for a given (GPU model, number GPUS, batch_size, model_max_length) combination."
    )

    hardcoded_parameters: dict[str, typing.Any] = {
        "max_steps": 5,
        "weights_format": common.WeightsFormat.Vanilla,
        "peft_method": None,
        "purpose": purpose,
    }

    # VV: Here configure any propertyDomains which differ from the default ones in EntitySpace
    override_propertydomains = {}

    default_params = {
        x: v
        for x, v in copy.deepcopy(common.DEFAULT_PARAMS).items()
        if x not in hardcoded_parameters
    }
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
        properties=[
            "f_gpu_oom",
            "f_other_error",
            "f_no_error",
            "is_valid",
        ],
    )

    catalog.addExperiment(param_experiment)


def add_experiments(catalog: "ExperimentCatalog") -> None:
    add_full_experiments(catalog)
    add_full_stability_experiments(catalog)
