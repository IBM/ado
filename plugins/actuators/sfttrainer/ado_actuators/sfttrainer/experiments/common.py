# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

# VV: @actuator_version Update this version according to semver.
# Consider "refactoring" the code as a "bug patch".
# Use `git-bisect` when looking for the implementation of a specific actuator version.
# Also remember to update the experiment descriptions in the `README.md` as bumping the version of the actuator
# also bumps the version of *all* currently supported experiments.
import copy
import enum
import functools
import logging
import os
import typing
from typing import Annotated

import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.constants as constants
import pydantic
import pydantic.fields
import pydantic_core
import yaml

import orchestrator.schema.domain
import orchestrator.schema.property_value
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.property import (
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
)
from orchestrator.schema.property_value import ConstitutivePropertyValue


class InternalInconsistencyError(ValueError):
    pass


class InvalidEntityError(ValueError):
    pass


ACTUATOR_IDENTIFIER = "SFTTrainer"
ACTUATOR_VERSION = "2.2.0"

FMS_HF_TUNING_REPOSITORY = "https://github.com/foundation-model-stack/fms-hf-tuning"
PACKAGES_DIR = f"{os.path.dirname(__file__)}/../packages"
CONFIG_DIR = f"{os.path.dirname(__file__)}/../config"

with open(os.path.join(CONFIG_DIR, "map_version_to_commit.yaml")) as f:
    FMS_HF_TUNING_COMMIT = yaml.safe_load(f)

FMS_HF_TUNING_VERSION = {
    version: f"{FMS_HF_TUNING_REPOSITORY}/tree/{commit}"
    for version, commit in FMS_HF_TUNING_COMMIT.items()
}


PATH_PINNED_PACKAGES = {
    version: f"{PACKAGES_DIR}/fms-hf-tuning_v{version}_{commit}.txt"
    for version, commit in FMS_HF_TUNING_COMMIT.items()
}


BACKEND_NAME_MAP = {
    "FSDP": "fsdp",
    "DDP": "ddp",
    None: "dp",
}


VALUES_TRANSFORMERS_ARGUMENT_OPTIM = [
    "adamw_hf",
    "adamw_torch",  # VV: transformers uses this as the default value of the --optim arg
    "adamw_torch_fused",
    "adamw_torch_xla",
    "adamw_torch_npu_fused",
    "adamw_apex_fused",
    "adafactor",
    "adamw_anyprecision",
    "adamw_torch_4bit",
    "ademamix",
    "sgd",
    "adagrad",
    "adamw_bnb_8bit",
    "adamw_8bit",
    "ademamix_8bit",
    "lion_8bit",
    "lion_32bit",
    "paged_adamw_32bit",  # VV: The blog-post references this one
    "paged_adamw_8bit",
    "paged_ademamix_32bit",
    "paged_ademamix_8bit",
    "paged_lion_32bit",
    "paged_lion_8bit",
    "rmsprop",
    "rmsprop_bnb",
    "rmsprop_bnb_8bit",
    "rmsprop_bnb_32bit",
    "galore_adamw",
    "galore_adamw_8bit",
    "galore_adafactor",
    "galore_adamw_layerwise",
    "galore_adamw_8bit_layerwise",
    "galore_adafactor_layerwise",
    "lomo",
    "adalomo",
    "grokadamw",
    "schedule_free_adamw",
    "schedule_free_sgd",
]


def packages_requiring_nvidia_development_binaries() -> list[str]:
    return [
        "fms-acceleration-foak",
        "fms-acceleration-moe",
        "triton",
        "flash_attn",
        "mamba-ssm",
        "causal-conv1d",
        "nvidia-cublas-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-nccl-cu12",
        "nvidia-nvjitlink-cu12",
        "nvidia-nvtx-cu12",
    ]


def parse_semver(what: str) -> list | None:
    import re

    p = re.compile(r"(\d+)\.(\d+)\.(\d+)")
    m = p.fullmatch(what)

    if m is None:
        return None

    return [int(x) for x in [m.group(1), m.group(2), m.group(3)]]


semvers = [x for x in map(parse_semver, FMS_HF_TUNING_VERSION) if x is not None]

# VV: We use these to instantiate Optional Parameters for all experiments (values are the defaults of the params)


class WeightsFormat(str, enum.Enum):
    Vanilla = "Vanilla"
    GPTQQuantized = "GPTQ-Quantized"

    def __str__(self) -> str:
        return self.value


class ExperimentPurpose(str, enum.Enum):
    Performance = "Performance"
    Stability = "Stability"

    def __str__(self) -> str:
        return self.value


@functools.cache
def load_model_map() -> dict[str, dict[WeightsFormat, str]]:
    from importlib.resources import read_text

    return yaml.safe_load(read_text("ado_actuators.sfttrainer.config", "models.yaml"))


def get_default_measured_properties() -> list[str]:
    return [
        "is_valid",
        "dataset_tokens_per_second_per_gpu",
        "train_runtime",
        "dataset_tokens_per_second",
        # VV: the next 4 are inaccurate when terminating the job early
        "train_samples_per_second",
        "train_steps_per_second",
        "train_tokens_per_second",
        "train_tokens_per_gpu_per_second",
        # VV: We no longer record the model_load_time
        # "model_load_time",
        # VV: CPU measurements
        "cpu_compute_utilization",
        "cpu_memory_utilization",
        # VV: GPU measurements - these will be 0 when running on a machine without GPUs
        "gpu_compute_utilization_min",
        "gpu_compute_utilization_avg",
        "gpu_compute_utilization_max",
        "gpu_memory_utilization_min",
        "gpu_memory_utilization_avg",
        "gpu_memory_utilization_max",
        "gpu_memory_utilization_peak",
        "gpu_power_watts_min",
        "gpu_power_watts_avg",
        "gpu_power_watts_max",
        "gpu_power_percent_min",
        "gpu_power_percent_avg",
        "gpu_power_percent_max",
    ]


ModelMap = load_model_map()

# VV: by convention, we use .jsonl for text datasets
DatasetMap = {
    x: f"{x}.jsonl"
    for x in (
        # VV: entries = max_batch_size * gradient_accumulation = 8 * 128 * 1
        "news-chars-512-entries-1024",
        "news-chars-1024-entries-1024",
        "news-chars-2048-entries-1024",
        # VV: entries = max_batch_size * gradient_accumulation = 8 * 128 * 4
        "news-chars-512-entries-4096",
        "news-chars-1024-entries-4096",
        "news-chars-2048-entries-4096",
        "news-tokens-16384plus-entries-4096",
        "news-tokens-128kplus-entries-4096",
        # VV: entries = max_batch_size * gradient_accumulation = 2 * 32 * 4
        "news-chars-512-entries-256",
        "news-chars-1024-entries-256",
        "news-chars-2048-entries-256",
        # SV: entries = max_batch_size * gradient_accumulation = 8 * 10 * 4
        "news-tokens-128kplus-entries-320",
    )
}


# VV: by convention, we use .parquet for vision datasets
DatasetMap.update(
    {
        x: f"{x}.parquet"
        for x in (
            "vision-384x384-16384plus-entries-4096",
            "vision-384x768-16384plus-entries-4096",
        )
    }
)


def get_fms_hf_tuning_package(commit: str) -> str:
    return f"fms-hf-tuning@git+{FMS_HF_TUNING_REPOSITORY}@{commit}"


def get_fms_hf_tuning_src(fms_hf_tuning_version: str) -> str:
    # VV: TODO: Generate links to pypi if fms_hf_tuning_version is a valid pypi version
    return FMS_HF_TUNING_VERSION[fms_hf_tuning_version]


def get_versioning_metadata(
    fms_hf_tuning_version: str | list[str],
) -> dict[str, str]:
    ret = {
        "actuator": ACTUATOR_VERSION,
    }
    if isinstance(fms_hf_tuning_version, str):
        ret.update(
            {
                "version": fms_hf_tuning_version,
                "src": get_fms_hf_tuning_src(
                    fms_hf_tuning_version=fms_hf_tuning_version
                ),
            }
        )
    else:
        ret.update(
            {
                "supported_fms_hf_tuning_versions": [
                    {
                        "fms-hf-tuning-src": get_fms_hf_tuning_src(
                            fms_hf_tuning_version=v
                        ),
                        "fms-hf-tuning": v,
                    }
                    for v in fms_hf_tuning_version
                ]
            }
        )

    return ret


def generate_parameterisable_finetune_experiment(
    description: str,
    method: str,
    exp_name: str,
    exp_identifier: str,
    version: str,
    actuator_identifier: str,
    override_propertydomains: dict[str, orchestrator.schema.domain],
    required_property_names: list[str],
    default_params: dict[str, str | float | bool | int],
    hardcoded_parameters: dict[str, typing.Any],
    fms_hf_tuning_versions: list[str] | str,
    properties: list[str] = ...,
) -> "Experiment":
    """Generates a finetune experiment

    Args:
        description:
            human readable description
        method:
            the finetune method
        exp_name:
            the name of the experiment (basically the identifier but without the version suffix)
        exp_identifier:
            the unique identifier of this experiment
        version:
            the experiment version (e.g. "1.1.1")
        actuator_identifier:
            the actuator identifier
        override_propertydomains:
            A dictionary whose keys are property names for which to use the value as the propertyDomain instead of
            the propertyDomain returned by EntitySpace.domain_for_constitutive_property().
            Use this for experiments which have their own custom rules for propertyDomain
        required_property_names:
            names of the required properties
        default_params:
            Key value pairs with default values of optional properties for experiment measurements
        fms_hf_tuning_versions:
            A list of supported versions of the fms-hf-tuning framework
        hardcoded_parameters:
            These are hardcoded parameters in the experiment which a user cannot change in any way
        properties:
            the names of the properties that this experiment measures. When not provided they default to
            the output of get_default_measured_properties()

    Returns:
        An experiment
    """
    import pydantic_core

    default_default_params = {
        "max_steps": -1.0,
        "num_train_epochs": 1.0,
        "stop_after_seconds": -1.0,
    }

    default_default_params = {
        k: v for k, v in default_default_params.items() if k not in hardcoded_parameters
    }
    default_default_params.update(default_params)
    default_params = default_default_params

    default_override_property_domains = {
        "max_steps": PropertyDomain(
            domainRange=[-1, 10000 + 1],
            interval=1,
            variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        ),
        "num_train_epochs": PropertyDomain(
            domainRange=[1.0, 10000.0 + 1.0],
            interval=1,
            variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        ),
        "stop_after_seconds": PropertyDomain(
            domainRange=[-1.0, 100_0000 + 1.0],
            interval=1,
            variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        ),
    }

    # VV: fill in common property domains
    default_override_property_domains = {
        k: v
        for k, v in default_override_property_domains.items()
        if k not in hardcoded_parameters
    }

    default_override_property_domains.update(override_propertydomains)
    override_propertydomains = default_override_property_domains

    def property_domain_for_prop(
        identifier: str,
    ) -> PropertyDomain:
        if identifier in override_propertydomains:
            return override_propertydomains[identifier]
        return EntitySpace.domain_for_constitutive_property(
            identifier, EntitySpace.model_fields[identifier]
        )

    optional_properties = [
        ConstitutiveProperty(
            identifier=identifier,
            propertyDomain=property_domain_for_prop(identifier),
            metadata={"description": EntitySpace.model_fields[identifier].description},
        )
        for identifier in set(default_params).union(override_propertydomains)
    ]

    optional_properties = sorted(optional_properties, key=lambda x: x.identifier)

    default_parameterisation = []

    default_parameterisation.extend(
        [
            ConstitutivePropertyValue(
                property=ConstitutivePropertyDescriptor(identifier=identifier),
                value=value,
            )
            for identifier, value in default_params.items()
        ]
    )

    required_properties = [
        p
        for p in EntitySpace.orch_experiment_required_properties()
        if p.identifier in required_property_names
    ]

    req_prop_identifiers = [
        p.identifier
        for p in required_properties
        # VV: All fields of the EntitySpace basemodel represent required properties, except for those which are:
        # 1. hardcoded by this experiment (e.g. a lora tuning method), or
        # 2. parametrisable by this experiment
        if (
            p.identifier not in hardcoded_parameters
            and p.identifier not in default_params
        )
    ]

    required_properties = [
        p for p in required_properties if p.identifier in req_prop_identifiers
    ]

    # VV: Sanity check rules:
    # 1. optional and required properties must **not** include any of the hard-coded params (i.e. base_params)
    # 2. optional properties + required properties + hard coded params = superset all fields of EntitySpace

    property_names = [p.identifier for p in required_properties] + [
        p.identifier for p in optional_properties
    ]
    cannot_be_hardcoded = {
        identifier
        for identifier in hardcoded_parameters
        if identifier in property_names
    }
    if cannot_be_hardcoded:
        raise ValueError(
            "The following hard coded parameters are also properties",
            cannot_be_hardcoded,
        )

    superset = property_names + list(hardcoded_parameters)
    missing = set(EntitySpace.model_fields).difference(superset)

    missing = {
        p
        for p in missing
        if isinstance(
            EntitySpace.model_fields[p].default,
            pydantic_core._pydantic_core.PydanticUndefinedType,
        )
    }

    if missing:
        raise ValueError(
            "The EntitySpace contains the following properties for which there "
            "is neither a property nor a hardcoded parameter",
            missing,
            {EntitySpace.model_fields[p].default for p in missing},
            {
                isinstance(
                    EntitySpace.model_fields[p].default,
                    pydantic_core._pydantic_core.PydanticUndefinedType,
                )
                for p in missing
            },
        )

    if properties is ...:
        properties = get_default_measured_properties()

    versioning = get_versioning_metadata(fms_hf_tuning_version=fms_hf_tuning_versions)
    versioning["experiment"] = version

    is_true_parameterised = False

    for p in optional_properties:
        prop_domain = property_domain_for_prop(p.identifier)

        # VV: An experiment is a true parameterised experimenti if it has at least 1 optional property which
        # can have multiple values
        if (
            prop_domain.domainRange
            and (prop_domain.domainRange[1] - prop_domain.domainRange[0] > 1)
        ) or (prop_domain.values is not None and len(prop_domain.values) > 1):
            is_true_parameterised = True
            break

    return Experiment(
        identifier=exp_identifier,
        actuatorIdentifier=actuator_identifier,
        metadata={
            "description": description,
            "method": method,
            "experiment": exp_name,
            "deprecated": False,
            "parameterised": is_true_parameterised,
            "parameters": copy.deepcopy(hardcoded_parameters),
            # VV: See @actuator_version
            "versioning": versioning,
        },
        targetProperties=[
            orchestrator.schema.property.AbstractPropertyDescriptor(identifier=prop)
            for prop in properties
        ],
        optionalProperties=optional_properties,
        requiredProperties=required_properties,
        defaultParameterization=default_parameterisation,
    )


class SFTTrainerCLIArgs(pydantic.BaseModel):
    """These are Entity properties which map to a CLI arg"""

    model_config = pydantic.ConfigDict(
        extra="forbid", protected_namespaces=(), use_enum_values=True
    )

    # VV: If you're updating these, then make sure you also update domain_for_constitutive_property()
    # the code uses `examples` to populate the categorical values of the constitutive property's domain
    model_name: Annotated[
        str,
        pydantic.Field(
            examples=sorted(ModelMap.keys()),
            description="The huggingface name or path to the model",
        ),
    ]
    model_max_length: Annotated[
        int,
        pydantic.Field(
            examples=[512, 2048, 8192],
            description="The maximum context size. Dataset entries with more tokens they are truncated. Entries with "
            "fewer are padded",
        ),
    ]

    dataset_id: Annotated[
        str,
        pydantic.Field(
            examples=sorted(DatasetMap.keys()),
            description="The identifier of the dataset to use for training",
        ),
    ] = "news-tokens-16384plus-entries-4096"

    batch_size: Annotated[
        int,
        pydantic.Field(
            examples=[1, 16, 128], description="The total batch size to use"
        ),
    ]

    # VV: These represent optional properties
    peft_method: Annotated[
        typing.Literal["pt", "lora"] | None,
        pydantic.Field(
            # VV: We auto convert `"full"` into `None`
            examples=["full", "lora", "pt"],
            description='The tuning method. Set to "full" to perform full finetuning',
        ),
    ]

    gradient_checkpointing: Annotated[
        bool,
        pydantic.Field(
            examples=[True, False],
            description="If True, use gradient checkpointing to save memory (i.e. higher batchsizes) at the expense "
            "of slower backward pass",
        ),
    ] = True

    torch_dtype: Annotated[
        str,
        pydantic.Field(
            examples=["bfloat16", "float16", "float32"],
            description="The torch datatype to use",
        ),
    ] = "bfloat16"

    gradient_accumulation_steps: Annotated[
        int,
        pydantic.Field(
            examples=[4],
            description="Number of update steps to accumulate before performing a backward/update pass.",
        ),
    ] = 4

    # VV: tuning.sft_trainer interprets max_steps and num_train_epochs together
    max_steps: Annotated[
        int,
        pydantic.Field(
            examples=[-1, 5],
            description="The number of optimization steps to perform. Set to -1 to respect num_train_epochs instead",
        ),
    ] = -1
    num_train_epochs: Annotated[
        float,
        pydantic.Field(
            examples=[1.0],
            description="How many epochs to run. Ignored if max_steps is greater than 0",
        ),
    ] = 1.0
    stop_after_seconds: Annotated[
        float,
        pydantic.Field(
            examples=[-1, 60 * 10],
            description="If set, the optimizer will be asked to stop after the specified time elapses. "
            "The check is performed after the end of each training step.",
        ),
    ] = -1.0

    auto_stop_method: Annotated[
        constants.AutoStopMethod | None,
        pydantic.Field(
            examples=[
                constants.AutoStopMethod.WARMUP_60S_STABLE_120S_OR_10_STEPS.value,
                None,
            ],
            description="The default value is `None`. This parameter defines the method used to automatically "
            "stop the fine-tuning job. Supported values are `WARMUP_60S_STABLE_120S_OR_10_STEPS` and "
            "`None`. If set to `WARMUP_60S_STABLE_120S_OR_10_STEPS`, the job stops after spending at least "
            "60 seconds in the warmup phase plus the longer of 120 seconds or the duration of 10 "
            "optimization steps. This method excludes the first 60 seconds of training when calculating "
            "throughput and system metrics.",
        ),
    ] = None

    # VV: lora specific parameters
    r: Annotated[
        int, pydantic.Field(examples=[4, 8, 16], description="The LORA rank")
    ] = 4

    lora_alpha: Annotated[
        int,
        pydantic.Field(
            examples=[16], description="LORA Alpha scales the learning weights"
        ),
    ] = 16

    fast_moe: Annotated[
        list[int] | None,
        pydantic.Field(
            # VV: Here "0" is a stand in for "disabled" -> we translate this into a None via pydantic
            examples=[0, 1, 2, 4, 8],
            description="Configures the amount of expert parallel sharding. number_gpus must be divisible by it",
        ),
    ] = 0

    fast_kernels: Annotated[
        list[str] | None,
        pydantic.Field(
            description="Switches on fast kernels, the value is a list with strings of boolean values for "
            "[fast_loss, fast_rms_layernorm, fast_rope_embeddings]",
            examples=[None, ["True", "True", "True"]],
        ),
    ] = None

    # VV: TAG: @HF_RAM_Efficient_Training
    # VV: These are all parameters that the huggingface blogpost on RAM efficient training uses
    # url: https://huggingface.co/blog/ram-efficient-pytorch-fsdp
    # The default values are all set to those we used for the default finetune v1.0.0 experiment i.e.
    # transformers v4.45.2 as that's the version fms-hf-tuning v2.1.2 uses

    optim: Annotated[
        str,
        pydantic.Field(
            description="The optimizer to use.",
            examples=VALUES_TRANSFORMERS_ARGUMENT_OPTIM,
        ),
    ] = "adamw_torch"

    bf16: Annotated[
        bool,
        pydantic.Field(
            description="Whether to use bf16 (mixed) precision instead of 32-bit. "
            "Requires Ampere or higher NVIDIA add bf16 mixed precision support for NPU "
            "architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change.",
            examples=[False, True],
        ),
    ] = False

    gradient_checkpointing_use_reentrant: Annotated[
        bool,
        pydantic.Field(
            description="Specify whether to use the activation checkpoint variant that requires reentrant autograd. "
            "This parameter should be passed explicitly. Torch version 2.5 will raise an exception "
            "if use_reentrant is not passed. If use_reentrant=False, checkpoint will use an implementation "
            "that does not require reentrant autograd. This allows checkpoint to support additional functionality, "
            "such as working as expected with torch.autograd.grad and support for keyword arguments input "
            "into the checkpointed function.",
            examples=[False, True],
        ),
    ] = False

    # VV: for image-to-text (vision) models
    dataset_text_field: Annotated[
        str | None,
        pydantic.Field(
            examples=["output", "messages"],
            description="Training dataset text field containing single sequence. "
            "Either the dataset_text_field "
            "or data_formatter_template need to be supplied. "
            "For running vision language model tuning pass the column name for text data.",
        ),
    ] = "output"

    dataset_image_field: Annotated[
        str | None,
        pydantic.Field(
            examples=[None, "images"],
            description="For running vision language model tuning pass "
            "the column name of the image data in the dataset.",
        ),
    ] = None

    remove_unused_columns: Annotated[
        bool | None,
        pydantic.Field(
            examples=[True, False],
            description="Remove columns not required by the model when using an nlp.Dataset.",
        ),
    ] = True

    dataset_kwargs_skip_prepare_dataset: Annotated[
        bool | None,
        pydantic.Field(
            examples=[True, False],
            description="When True, configures trl to skip preparing the dataset.",
        ),
    ] = False

    flash_attn: Annotated[
        bool,
        pydantic.Field(
            examples=[True, False],
            description="Use Flash attention v2 from transformers",
        ),
    ] = True

    @pydantic.field_validator("peft_method", mode="before")
    def upgrade_peft_method(cls, value: str) -> str | None:
        if isinstance(value, str) and value == "full":
            return None
        return value

    @pydantic.field_validator("fast_moe", mode="before")
    def upgrade_fast_moe(cls, value: int | list[int] | None) -> list[int] | None:
        # VV: Currently, fast_moe has a single argument in it (ep_degree). It's easier to describe the property
        # domain of discrete values so we're going to assume that this is a single integer for now
        if isinstance(value, (int, float)):
            if int(value) != value:
                # VV: ado converts integers to floats - if the value has decimal digits raise an exception
                return value
            value = int(value)

            if value == 0:
                return None
            return [value]
        return value


# VV: Different experiments have different required properties. Here we keep a record of their names
# We'll use this information to partition EntitySpace into:
#  1. hardcoded parameters in the Experiment metadata (options a user cannot override at all)
#  2. optional parameters (options a user may override but is not required to provide)
#  3. required parameters (options a user must provide)

MINIMUM_PROPS = [
    "model_name",
    "model_max_length",
    "batch_size",
    "number_gpus",
    # "gpu_model",
]

DEPRECATED_MINIMUM_PROPS_SINGLE_NODE = [*MINIMUM_PROPS, "dataset_id", "torch_dtype"]

DEPRECATED_MINIMUM_PROPS_MULTI_NODE = [
    *DEPRECATED_MINIMUM_PROPS_SINGLE_NODE,
    "number_nodes",
]


class EntitySpace(SFTTrainerCLIArgs):
    """This contains Entity properties which however do not map to a CLI argument"""

    # VV: If you're updating these, then make sure you also update domain_for_constitutive_property()
    # the code uses `examples` to populate the categorical values of the constitutive property's domain

    # VV: no examples as the number of gpus depends on whether you're using FSDP, DDP, or DP
    number_gpus: Annotated[
        int,
        pydantic.Field(
            examples=[0, 1, 2, 4, 8],
            description="The total number of GPUs to use",
            exclude=True,
        ),
    ]

    gpu_model: Annotated[
        str | None,
        pydantic.Field(
            examples=[
                None,
                "NVIDIA-A100-SXM4-80GB",
                "NVIDIA-A100-80GB-PCIe",
                "Tesla-T4",
                "L40S",
                "Tesla-V100-PCIE-16GB",
                "NVIDIA-H100-PCIe",
                "NVIDIA-H100-80GB-HBM3",
            ],
            description="The GPU model to use",
            exclude=True,
        ),
    ] = None

    distributed_backend: Annotated[
        typing.Literal["DDP", "FSDP"] | None,
        pydantic.Field(
            examples=["DDP", "FSDP", "None"],
            description="Which pytorch backend to use when training with multiple GPU devices",
            exclude=True,
        ),
    ] = "FSDP"

    number_nodes: Annotated[
        int,
        pydantic.Field(
            examples=[1, 2, 3, 4],
            description="If set, actuator distributes tasks on multiple nodes. "
            "Each Node will use number_gpus/number_nodes GPUs. "
            "Each Node will use 1 process for each GPU it uses",
            exclude=True,
        ),
    ] = 1

    # VV: In #1175 we decided that we're not going to update the default value of this experiment
    fms_hf_tuning_version: Annotated[
        str | None,
        pydantic.Field(
            examples=list(FMS_HF_TUNING_VERSION),
            description="The version of fms-hf-tuning to use - controls which wrapper to use "
            "as well as python dependencies",
            exclude=True,
        ),
    ] = "2.1.2"

    enable_roce: Annotated[
        bool,
        pydantic.Field(
            examples=[False, True],
            description="This setting is only in effect for multi-node runs. "
            "It controls whether RDMA over Converged Ethernet (RoCE) is switched on or not",
            exclude=True,
        ),
    ] = False

    # VV: TAG: @HF_RAM_Efficient_Training
    # VV: These are settings we need to replicate the huggingface post we store them in DistributedSettings
    fsdp_sharding_strategy: Annotated[
        typing.Literal[
            "FULL_SHARD",
            "SHARD_GRAD_OP",
            "NO_SHARD",
            "HYBRID_SHARD",
            "HYBRID_SHARD_ZERO2",
        ],
        pydantic.Field(
            description="[1] FULL_SHARD (shards optimizer states, gradients and parameters), "
            "[2] SHARD_GRAD_OP (shards optimizer states and gradients), "
            "[3] NO_SHARD (DDP), "
            "[4] HYBRID_SHARD (shards optimizer states, gradients and parameters within each node "
            "while each node has full copy - equivalent to FULL_SHARD for single-node runs), "
            "[5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients within each node while each node has "
            "full copy). For more information, please refer the official PyTorch docs.",
            examples=[
                "FULL_SHARD",
                "SHARD_GRAD_OP",
                "NO_SHARD",
                "HYBRID_SHARD",
                "HYBRID_SHARD_ZERO2",
            ],
            exclude=True,
        ),
    ] = "FULL_SHARD"

    fsdp_state_dict_type: Annotated[
        typing.Literal["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"],
        pydantic.Field(
            description="[1] FULL_STATE_DICT, [2] LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT",
            examples=["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"],
            exclude=True,
        ),
    ] = "FULL_STATE_DICT"

    fsdp_use_orig_params: Annotated[
        bool,
        pydantic.Field(
            description="If True, allows non-uniform `requires_grad` during init, which means support for "
            "interspersed frozen and trainable parameters. (useful only when `use_fsdp` flag is passed).",
            exclude=True,
            examples=[False, True],
        ),
    ] = True

    accelerate_config_mixed_precision: Annotated[
        typing.Literal["no", "fp16", "bf16", "fp8"],
        (
            pydantic.Field(
                examples=["no", "fp16", "bf16", "fp8"],
                description="Whether or not to use mixed precision training. Choose from 'no', 'fp16', 'bf16' or 'fp8'. "
                "'fp8' requires the installation of transformers-engine.",
                exclude=True,
            )
        ),
    ] = "no"

    accelerate_config_fsdp_transformer_layer_cls_to_wrap: Annotated[
        str | None,
        pydantic.Field(
            examples=[
                None,
                "GraniteDecoderLayer",
                "LlamaDecoderLayer",
                "MistralDecoderLayer",
                "GPTJBlock",
                "T5Block",
            ],
            description=(
                "List of transformer layer class names (case-sensitive) to wrap, e.g, BertLayer, "
                "GraniteDecoderLayer, GPTJBlock, T5Block ... (useful only when using FSDP)"
            ),
            exclude=True,
        ),
    ] = None

    @pydantic.field_validator("dataset_id")
    def val_dataset_id(cls, value: str) -> str:
        if value not in DatasetMap:
            raise ValueError("Unknown dataset")

        return value

    @classmethod
    def domain_for_constitutive_property(
        cls,
        property: str,
        field_info: pydantic.fields.FieldInfo,
    ) -> "PropertyDomain | None":
        # VV: number_gpus need to be divisible by fast_moe so we can assume that they have the same range
        # also, both fast_moe and number_gpus can be set to 0
        gpus_range = [0, 32 + 1]

        discrete_properties_with_range = {
            "number_gpus": gpus_range,
            "batch_size": [1, 4096 + 1],
            "model_max_length": [1, (2 << 16) + 1],
            "gradient_accumulation_steps": [1, 32 + 1],
            "number_nodes": [1, 8 + 1],
            "r": [1, 32 + 1],
            "lora_alpha": [1, 32 + 1],
            "fast_moe": gpus_range,
        }

        if property in discrete_properties_with_range:
            return PropertyDomain(
                domainRange=discrete_properties_with_range[property],
                interval=1,
                variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
            )

        values = field_info.examples

        if not values:
            raise ValueError(
                f"EntitySpace.{property} has no examples in the BaseModel - fix it"
            )

        if isinstance(values[0], bool):
            variable_type = VariableTypeEnum.BINARY_VARIABLE_TYPE
        elif isinstance(values[0], (int, float)):
            variable_type = VariableTypeEnum.DISCRETE_VARIABLE_TYPE
        else:
            variable_type = VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE

        return PropertyDomain(
            values=values,
            variableType=variable_type,
        )

    @pydantic.field_validator("peft_method", mode="before")
    def upgrade_peft_method(cls, value: str | None) -> str | None:
        if isinstance(value, str) and value == "full":
            return None

        return value

    @pydantic.field_validator(
        "distributed_backend",
        mode="before",
    )
    def set_none_string_to_none_literal(cls, value: str | None) -> str | None:
        if isinstance(value, str) and value == "None":
            return None

        return value

    @classmethod
    def orch_experiment_required_properties(
        cls,
    ) -> "list[ConstitutiveProperty]":

        return [
            ConstitutiveProperty(
                identifier=identifier,
                metadata={"description": field_info.description},
                propertyDomain=cls.domain_for_constitutive_property(
                    property=identifier, field_info=field_info
                ),
            )
            for identifier, field_info in cls.model_fields.items()
        ]

    def validate_and_update(
        self, exp_params: "ExperimentParameters", logger: logging.Logger
    ) -> None:
        """Method updates both @self and @exp_params to make the 2 consistent with each other, also validates them

        Args:
            exp_params:
                The experiment-specific parameters. May get updated in place
            logger:
                A logger object

        Raises:
            InvalidEntityError:
                When the experiment parameters or entity space contain invalid information
        """

        # VV: Here we convert the "effective" batch size to the per_device_train_batch_size.
        # Unfortunately, some values of batch_size and number_gpus are describing inconsistent runs, for example
        # batch_size = 1 and number_gpus = 2 would mean that 2 devices are expected to process 1 batch_size.
        # We could either artificially increase the batch_size to 2, reduce the number_gpus or just report
        # that this entity is not valid. Here, we opt for the latter.
        if self.batch_size % max(1, self.number_gpus) != 0:
            # VV: This marks the run as `success` - there will be just 1 measured property: `is_valid = 0`
            raise InvalidEntityError(
                f"batch_size is {self.batch_size} but number_gpus is {self.number_gpus}"
            )

        if exp_params.multi_node is not None and (
            self.number_nodes > 1 and not exp_params.multi_node
        ):
            raise InvalidEntityError(
                f"number_nodes is {self.number_nodes} but experiment is single node"
            )

        if self.distributed_backend is not None and self.number_gpus < 2:
            logger.info(
                f"Requested an {self.distributed_backend} experiment using {self.number_gpus} "
                f"gpus will set distributed_backend to None"
            )
            self.distributed_backend = None

        if self.number_nodes == 1 and exp_params.multi_node:
            logger.info(
                "Requested a multi_node experiment but number_nodes == 1, "
                "will run the equivalent single node experiment instead"
            )
            exp_params.multi_node = False

        if self.number_nodes > 1:
            exp_params.multi_node = True

        if max(self.number_gpus, 1) % self.number_nodes != 0:
            raise InvalidEntityError(
                f"number_gpus is {self.number_gpus} but number_nodes is {self.number_nodes}"
            )

        if self.fast_moe:
            ep_degrees = self.fast_moe[0]

            if self.number_gpus % ep_degrees != 0:
                raise InvalidEntityError(
                    f"number_gpus is {self.number_gpus} but fast_moe[0] (i.e. ep_degrees) is {ep_degrees}"
                )

            import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.tuning_versions as tv

            if (
                not self.fms_hf_tuning_version
                or tv.semver_cmp(tv.semver_parse(self.fms_hf_tuning_version), (2, 4, 0))
                == -1
            ):
                raise InvalidEntityError(
                    "fast_moe is supported for fms-hf-tuning version v2.4.0 and onwards but "
                    f"fms_hf_tuning_version is {self.fms_hf_tuning_version}"
                )


# VV: Get the defaults from the definition of the EntitySpace
DEFAULT_PARAMS = {
    k: v.default
    for k, v in EntitySpace.model_fields.items()
    if not isinstance(v.default, pydantic_core._pydantic_core.PydanticUndefinedType)
}


class ExperimentParameters(pydantic.BaseModel):
    # VV: Fields which do not map to a CLI arg of tuning.sft_trainer should come with exclude=True

    model_config = pydantic.ConfigDict(
        extra="forbid", protected_namespaces=(), use_enum_values=True
    )

    multi_node: Annotated[
        bool | None,
        pydantic.Field(
            description="Set for experiments which can only do one of multi-node or single-gpu measurements",
            exclude=True,
        ),
    ] = None

    # VV: For example 4.0
    num_train_epochs: Annotated[float, pydantic.Field()] = 1.0
    max_steps: Annotated[int, pydantic.Field()] = -1

    gradient_checkpointing: Annotated[bool, pydantic.Field()]

    # VV: Describes the goals of this experiment (e.g. measure performance metrics, get stability statistics, etc)
    purpose: Annotated[
        ExperimentPurpose,
        pydantic.Field(
            description="What this experiment is aiming to measure",
            exclude=True,
        ),
    ] = ExperimentPurpose.Performance

    # VV: Different experiments expect different weight formats
    weights_format: Annotated[
        WeightsFormat,
        pydantic.Field(
            description="The kind of weights this experiment can load",
            exclude=True,
        ),
    ]

    def args_for_entity_space(
        self, entity_space: EntitySpace, model_map: dict[WeightsFormat, str]
    ) -> dict[str, typing.Any]:

        args = self.model_dump()
        try:
            args["model_name"] = model_map[self.weights_format]
        except KeyError as error:
            raise NotImplementedError(
                f"Experiment expects weights_format {self.weights_format} but "
                f"the available weight formats for the model are {list(model_map)}"
            ) from error
        return args


class PromptTuningExperimentParameters(ExperimentParameters):
    peft_method: Annotated[
        typing.Literal["pt"],
        pydantic.Field(description="This is a prompt-tuning experiment"),
    ]


class FullFinetuningExperimentsParameters(ExperimentParameters):
    peft_method: Annotated[
        None, pydantic.Field(description="This is a full fine-tuning experiment")
    ]

    @pydantic.field_validator("peft_method", mode="before")
    def upgrade_peft_method(cls, value: str) -> str | None:
        if isinstance(value, str) and value == "full":
            return None
        return value


class LoraExperimentParameters(ExperimentParameters):
    peft_method: Annotated[
        typing.Literal["lora"], pydantic.Field(description="This is a LORA experiment")
    ]

    # r: int
    #
    # lora_alpha: int

    target_modules_map: Annotated[
        dict[str, list[str]],
        pydantic.Field(
            description="A map of model_name to list of target_modules for LORA",
            exclude=True,
        ),
    ]

    def args_for_entity_space(
        self, entity_space: EntitySpace, model_map: dict[WeightsFormat, str]
    ) -> dict[str, typing.Any]:
        args = super().args_for_entity_space(
            entity_space=entity_space, model_map=model_map
        )
        try:
            args["target_modules"] = self.target_modules_map[entity_space.model_name]

        except KeyError as error:
            raise NotImplementedError(
                f"No target_modules mapping for model {entity_space.model_name}"
            ) from error

        return args


class GPTQLoraExperimentParameters(LoraExperimentParameters):
    auto_gptq: Annotated[str, pydantic.Field()]

    fp16: Annotated[str, pydantic.Field()]

    fast_kernels: Annotated[list[str], pydantic.Field()]

    fused_lora: Annotated[list[str], pydantic.Field()]

    torch_dtype: Annotated[typing.Literal["float16"], pydantic.Field()]

    # VV: Anh Uong said that padding_free is not required as GPTQ-LoRA was tested before support for
    # padding_free was available
    # padding_free: typing.List[str]

    def args_for_entity_space(
        self, entity_space: EntitySpace, model_map: dict[WeightsFormat, str]
    ) -> dict[str, typing.Any]:

        if entity_space.torch_dtype != "float16" and self.auto_gptq == "triton_v2":
            raise InvalidEntityError("triton_v2 only supports torch_dtype=float16")

        return super().args_for_entity_space(
            entity_space=entity_space, model_map=model_map
        )


def experiment_parameters_from_experiment(
    exp: "Experiment",
    entity_values: dict[str, typing.Any],
) -> ExperimentParameters:
    import json

    method = exp.metadata.get("method")

    parameters = exp.metadata.get("parameters") or {}

    try:
        model_class: type[pydantic.BaseModel] = {
            "pt": PromptTuningExperimentParameters,
            "lora": LoraExperimentParameters,
            "full": FullFinetuningExperimentsParameters,
            "full-error": FullFinetuningExperimentsParameters,
            "gptq-lora": GPTQLoraExperimentParameters,
        }[method]
    except KeyError as error:
        raise InternalInconsistencyError(
            f"There is no ExperimentParameters schema for experiment {exp.identifier} with method {method}"
        ) from error

    # VV: Copy in any fields that the experiment sets and override the defaults from the experiment metadata
    # this also fills in any missing fields from the experiment metadata
    if entity_values:
        for name, value in entity_values.items():
            if name in model_class.model_fields:
                parameters[name] = value

    try:
        return model_class.model_validate(parameters)
    except Exception:
        print(
            "instantianting",
            exp.identifier,
            model_class.__name__,
            "with",
            json.dumps(parameters),
        )
        raise
