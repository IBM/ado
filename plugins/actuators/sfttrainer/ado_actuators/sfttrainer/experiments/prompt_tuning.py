# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import copy
import typing

from . import common

if typing.TYPE_CHECKING:
    from orchestrator.modules.actuators.catalog import ExperimentCatalog


def add_experiments(catalog: "ExperimentCatalog") -> None:

    method = "pt"
    version = "1.0.0"
    exp_name = f"finetune_{method}_benchmark"

    description = (
        "Measures the performance of prompt-tuning a model for a given "
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
