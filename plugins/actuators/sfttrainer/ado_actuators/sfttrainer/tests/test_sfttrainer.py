# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import dataclasses
import json
import os
import re
import typing

import ado_actuators.sfttrainer.actuators
import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.tuning_versions as tuning_versions
import pytest

import orchestrator.metastore.project
import orchestrator.modules.actuators.base
import orchestrator.modules.actuators.catalog
import orchestrator.modules.actuators.custom_experiments
import orchestrator.modules.actuators.replay
import orchestrator.modules.module
import orchestrator.schema.entity
import orchestrator.schema.experiment
import orchestrator.schema.property
import orchestrator.schema.property_value
import orchestrator.schema.reference
from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters


def try_instantiate_experiment(
    exp: "orchestrator.schema.entity.Experiment",
    entity_values: dict[str, typing.Any],
) -> "ado_actuators.sfttrainer.actuators.FinetuneContext":

    print("test", json.dumps(entity_values), "for", exp.identifier)

    values = [
        orchestrator.schema.property_value.ConstitutivePropertyValue(
            property=orchestrator.schema.property.ConstitutivePropertyDescriptor(
                identifier=k
            ),
            value=v,
        )
        for k, v in entity_values.items()
    ]

    entity = orchestrator.schema.entity.Entity(
        identifier="foo",
        generatorid="notsource",
        constitutive_property_values=tuple(values),
    )

    import ray

    actor = ado_actuators.sfttrainer.actuators.SFTTrainer.remote(
        queue=None,
        params=GenericActuatorParameters(),
    )
    context: ado_actuators.sfttrainer.actuators.FinetuneContext = ray.get(
        actor.prepare_finetune_context.remote(
            exp=exp, entity=entity, task_uid="hello", request_id="world"
        )
    )

    print("The arguments")
    print(json.dumps(dataclasses.asdict(context.args)))
    print("The EntitySpace")
    print(json.dumps(context.entity_space.model_dump()))

    return context


@pytest.mark.parametrize(
    "exp_name",
    [
        "finetune_full_benchmark",
        "finetune_full_stability",
        "finetune_lora_benchmark",
        "finetune_gptq-lora_benchmark",
        "finetune_pt_benchmark",
    ],
)
def test_sfttrainer_parameterised_experiment(exp_name: str) -> None:
    all_experiments = [
        e
        for e in ado_actuators.sfttrainer.actuators.catalog.experiments
        if e.metadata["experiment"] == exp_name and not e.deprecated
    ]

    assert len(all_experiments) > 0

    for exp in all_experiments:
        entity_values = {
            "model_name": "llama-7b",
            "number_gpus": 4,
            "model_max_length": 512,
            "batch_size": 4,
            "gpu_model": "NVIDIA-A100-80GB-PCIe",
        }

        if not exp.metadata["experiment"].endswith("_benchmark"):
            entity_values.update(
                {
                    "dataset_id": "news-tokens-16384plus-entries-4096",
                    "torch_dtype": (
                        "float16"
                        if exp.metadata["method"] == "gptq-lora"
                        else "bfloat16"
                    ),
                }
            )
            if exp.metadata["parameters"].get("multi_node"):
                entity_values["number_nodes"] = 2

        try_instantiate_experiment(exp=exp, entity_values=entity_values)

        if exp.metadata["experiment"].endswith("_benchmark"):
            entity_values = {
                "model_name": "llama-7b",
                "number_gpus": 1,
                "model_max_length": 512,
                "batch_size": 1,
                "gpu_model": "NVIDIA-A100-80GB-PCIe",
                "max_steps": 3,
            }

            context = try_instantiate_experiment(exp=exp, entity_values=entity_values)

            assert context.entity_space.max_steps == 3


def test_sfttrainer_fast_moe() -> None:
    # VV: The type of the experiment doesn't really matter
    all_full_experiments = [
        e
        for e in ado_actuators.sfttrainer.actuators.catalog.experiments
        if e.metadata["experiment"] == "finetune_full_benchmark" and not e.deprecated
    ]

    assert len(all_full_experiments) > 0

    for exp in all_full_experiments:
        entity_values = {
            "model_name": "llama-7b",
            "number_gpus": 4,
            "model_max_length": 512,
            "batch_size": 4,
            "gpu_model": "NVIDIA-A100-80GB-PCIe",
            "fast_moe": 2,
            # VV: fast moe is supported by fms-hf-tuning>=2.4.0
            "fms_hf_tuning_version": "2.4.0",
        }

        try_instantiate_experiment(exp=exp, entity_values=entity_values)

        # VV: fms_hf_tuning_version must be >= 2.4.0
        entity_values["fms_hf_tuning_version"] = "2.3.9"
        with pytest.raises(
            ado_actuators.sfttrainer.experiments.common.InvalidEntityError,
            match=re.escape(
                "fast_moe is supported for fms-hf-tuning version v2.4.0 and onwards but "
                "fms_hf_tuning_version is 2.3.9"
            ),
        ):
            try_instantiate_experiment(exp=exp, entity_values=entity_values)

        # VV: number_gpus % fast_moe must be 0
        entity_values["fms_hf_tuning_version"] = "2.4.0"
        entity_values["batch_size"] = 5
        entity_values["number_gpus"] = 5

        with pytest.raises(
            ado_actuators.sfttrainer.experiments.common.InvalidEntityError,
            match=re.escape("number_gpus is 5 but fast_moe[0] (i.e. ep_degrees) is 2"),
        ):
            try_instantiate_experiment(exp=exp, entity_values=entity_values)


def test_semver_cmp() -> None:
    assert tuning_versions.semver_cmp((3, 0, 0, 1), (3, 1, 0)) == -1
    assert tuning_versions.semver_cmp((3, 0, 0, 1), (3, 0, 0)) == 1
    assert tuning_versions.semver_cmp((3, 0, 0), (3, 0, 0, 1)) == -1
    assert tuning_versions.semver_cmp((3, 0, 0), (2, 0, 0, 1)) == 1
    assert tuning_versions.semver_cmp((3, 0, 0), (3, 0, 0, 0)) == 0


def test_select_wrapper() -> None:
    module_name = tuning_versions.get_wrapper_name_for_version(
        "3.1.0", os.path.dirname(tuning_versions.__file__)
    )
    assert module_name == "at_least_3_0_0_1"

    try:
        tuning_versions.import_tuning_version("3.1.0")
    except ModuleNotFoundError as e:
        # VV: fms-hf-tuning is not a required dependency of the SFTTrainer actuator
        if e.name != "tuning":
            raise
