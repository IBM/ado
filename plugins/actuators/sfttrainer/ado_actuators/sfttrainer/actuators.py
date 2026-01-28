# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import asyncio
import copy
import dataclasses
import importlib
import json
import logging
import math
import os
import traceback
import typing
from collections.abc import Callable
from typing import Annotated, Any

import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.callbacks.metrics_tracker as metrics_tracker
import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.finetune as finetune
import pydantic
import pydantic.typing
import ray
import ray.util
import ray.util.placement_group
import ray.util.state
import yaml
from ado_actuators.sfttrainer.experiments import (
    full_finetune,
    gptq_lora,
    lora,
    prompt_tuning,
)
from ado_actuators.sfttrainer.experiments.common import (
    ACTUATOR_IDENTIFIER,
    FMS_HF_TUNING_COMMIT,
    PATH_PINNED_PACKAGES,
    DatasetMap,
    EntitySpace,
    ExperimentParameters,
    InternalInconsistencyError,
    InvalidEntityError,
    ModelMap,
    WeightsFormat,
    experiment_parameters_from_experiment,
    get_fms_hf_tuning_package,
)
from ado_actuators.sfttrainer.ray_env import utils as ray_env_utils
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import orchestrator.modules.actuators.catalog
import orchestrator.schema.property_value
import orchestrator.schema.request
from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters

# VV: If you do import orchestrator.actuators.base and then use
# orchestrator.actuators.base.ActuatorBase
# as the base class of your Actuator class ray (for an unknown reason) rejects your Actuator
# the first time you try to use it. It claims that it does not have any `async` methods (i.e. coroutines)
# Using `from ... import` fixes this behaviour however we don't know why.
from orchestrator.modules.actuators.base import ActuatorBase, DeprecatedExperimentError
from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.observed_property import ObservedPropertyValue
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from orchestrator.schema.result import InvalidMeasurementResult, ValidMeasurementResult
from orchestrator.utilities.environment import enable_ray_actor_coverage

if typing.TYPE_CHECKING:
    from orchestrator.modules.actuators.measurement_queue import MeasurementQueue

# VV: Required module variables
identifier = ACTUATOR_IDENTIFIER

catalog = orchestrator.modules.actuators.catalog.ExperimentCatalog(
    catalogIdentifier=identifier
)


def _init_catalog(
    catalog: orchestrator.modules.actuators.catalog.ExperimentCatalog,
) -> None:
    full_finetune.add_experiments(catalog=catalog)
    prompt_tuning.add_experiments(catalog=catalog)
    lora.add_experiments(catalog=catalog)
    gptq_lora.add_experiments(catalog=catalog)

    # VV: Technically we can still run these old experiments so we won't be marking them as deprecated.
    # However, there's really no reason to include them in the actuator anymore. So I'll just move them
    # to a plugin for the actuator plugin.
    # There are also experiments which we no longer support at all. Those will show up as "deprecated".

    for old_experiment_type in ["full", "lora", "gptq_lora", "pt", "unsupported"]:
        try:
            name = f"ado_sfttrainer_deprecated_experiments.{old_experiment_type}"
            module = importlib.import_module(name)
        except (ModuleNotFoundError, ImportError):  # noqa: PERF203
            continue
        else:
            module.inject_deprecated_experiments(catalog=catalog)


_init_catalog(catalog)


def model_dump_all(model: pydantic.BaseModel) -> dict[str, Any]:
    """Recursively dumps all fields of a pydantic model ignoring exclude directives

    Args:
        model:
            The pydantic model to dump.

    Returns:
        A dictionary containing the dumped fields and values.
    """
    ret = {}
    pending = [(ret, model)]

    while pending:
        parent, model = pending.pop(0)

        for key, _value_info in model.model_fields.items():  # noqa: PERF102
            value = model.__getattribute__(key)

            if isinstance(value, pydantic.BaseModel):
                parent[key] = {}
                pending.append((parent[key], value))
            else:
                parent[key] = value

    return ret


class ActuatorParameters(
    orchestrator.core.actuatorconfiguration.config.GenericActuatorParameters
):
    model_config = pydantic.ConfigDict(
        extra="forbid", use_enum_values=True, protected_namespaces=()
    )

    match_exact_dependencies: Annotated[
        bool,
        pydantic.Field(
            description="If True, runs the measurement in a virtual environment that exactly matches the Python "
            "packages of the selected fms-hf-tuning version, enabling all optional features like "
            "fast_kernels, fast_moe, and flash_attn. If False, the system checks whether the machine initiating "
            "the measurement has NVIDIA development binaries or an ARM CPU, and excludes incompatible packages "
            "and features. Useful for running on limited-support devices like MacBooks.",
        ),
    ] = True

    output_dir: Annotated[
        str,
        pydantic.Field(
            description="The prefix directory path, under which "
            "to store the finetuned weights.",
        ),
    ] = "output"

    data_directory: Annotated[
        str,
        pydantic.Field(
            description="The directory that contains the data files",
        ),
    ] = "/data/fms-hf-tuning/artificial-dataset/"

    aim_dashboard_url: Annotated[
        str | None,
        pydantic.Field(
            description="The AIM Dashboard endpoint. When set, the actuator inserts the aim_url field "
            "in the MeasurementResult.metadata object that is associated with the measurement.",
        ),
    ] = None

    aim_db: Annotated[
        str | None,
        pydantic.Field(
            description="The AIM server endpoint. When set to None the "
            "measurement will use a temporary AIM repository that will be garbage collected after the termination "
            "of the measurement.",
        ),
    ] = None

    hf_home: Annotated[
        str,
        pydantic.Field(
            description="To configure where huggingface_hub will locally store data. In particular, "
            "your token and the cache will be stored in this folder.",
        ),
    ] = "/hf-models-pvc/huggingface_home"

    model_map: Annotated[
        dict[str, dict[WeightsFormat, str]],
        pydantic.Field(
            default_factory=lambda: copy.deepcopy(ModelMap),
            description="Maps model identifiers to their corresponding Hugging Face model ids and absolute paths."
            "The contents of this dictionary will override the defaults that ship with the Actuator.",
        ),
    ]

    num_tokens_cache_directory: Annotated[
        str | None,
        pydantic.Field(
            description="Use this to cache the number of tokens in the dataset so that we don't compute it over and over. "
            "It can take a few minutes to compute how many tokens are in a dataset, and that number depends on "
            "which model you're using, as well as the effective max sequence length (i.e. the min of the "
            "inherent max sequence length of the model and the value of --max_seq_length). "
            "The value is treated as a path to a directory that is relevant to the value of @data_dir. "
            "If num_tokens_cache_directory is None then the cache will not be used.",
        ),
    ] = "cache"

    @pydantic.field_validator("model_map", mode="before")
    @classmethod
    def upgrade_simple_model_map(
        cls,
        values: dict[str, str | dict[WeightsFormat, str]],
    ) -> dict[str, dict[WeightsFormat, str]]:
        """Auto converts model_map entries whose values are strings to {"Vanilla": <the str>}"""
        if not isinstance(values, dict):
            return values

        for k in values:
            if isinstance(values[k], str):
                values[k] = {WeightsFormat.Vanilla: values[k]}

        return values


def prepare_runtime_environment(
    actuator_parameters: ActuatorParameters,
    log: logging.Logger,
    space: EntitySpace,
    args: "finetune.FineTuneArgs",
) -> dict[str, Any]:
    exclude_packages = []

    if not actuator_parameters.match_exact_dependencies:
        if ray_env_utils.is_using_arm_cpu():
            exclude_packages.append("bitsandbytes")

        if not ray_env_utils.is_nvcc_available():
            exclude_packages.extend(
                ray_env_utils.packages_requiring_nvidia_development_binaries()
            )

    if exclude_packages:
        log.info(
            f"Because match_exact_dependencies=False we will exclude the packages {exclude_packages}"
        )

    # VV: Users that switch off match_exact_dependencies are likely in an "exploration" mode.
    # Help them out a bit by explaining why their measurement cannot work instead of asking them to
    # manually investigate the exception that pip raises.
    msg_unsupported_feat = (
        "The measurement requires {feature}, but the required NVIDIA development "
        "binaries are not supported on this platform. When using match_exact_dependencies=False you cannot use: "
        "fast_moe, fast_kernels, flash_attn"
    )
    if "fms-acceleration-foak" in exclude_packages and "true" in [
        str(x).lower() for x in space.fast_kernels or []
    ]:
        raise ValueError(msg_unsupported_feat.format(feature="fast_kernels"))

    if "fms-acceleration-moe" in exclude_packages and space.fast_moe not in [
        [0],
        0,
        None,
    ]:
        raise ValueError(msg_unsupported_feat.format(feature="fast_moe"))

    if "flash_attn" in exclude_packages and space.flash_attn:
        raise ValueError(msg_unsupported_feat.format(feature="flash_attn"))

    log.info(f"Excluded packages {exclude_packages}")
    packages = ray_env_utils.get_pinned_packages(
        path_requirements=PATH_PINNED_PACKAGES[space.fms_hf_tuning_version],
        override_fms_hf_tuning=get_fms_hf_tuning_package(
            commit=FMS_HF_TUNING_COMMIT[space.fms_hf_tuning_version]
        ),
        exclude_packages=exclude_packages,
        ensure_aim=True,
    )

    # VV: Detect any extra wheels and propagate them to the job e.g. sfttrainer
    import ray.runtime_context

    context = ray.get_runtime_context()

    pip_config_packages = context.runtime_env.pip_config().get("packages", [])
    uv_config_packages = context.runtime_env.uv_config().get("packages", [])

    ordered_pip = context.runtime_env.get("ordered_pip", {})
    ordered_pip_packages = []
    for phase in ordered_pip.get("phases", []):
        if isinstance(phase, dict):
            ordered_pip_packages.extend(phase.get("packages"), [])
        elif isinstance(phase, list):
            ordered_pip_packages.extend(phase)

    additional_packages = (
        pip_config_packages + uv_config_packages + ordered_pip_packages
    )

    additional_wheels = [
        x
        for x in additional_packages
        # VV: Do not install ado wheels other than sfttrainer. Their dependencies may conflict with
        # those in fms-hf-tuning
        if x.endswith(".whl")
        and not (
            os.path.basename(x).startswith("ado_")
            and not os.path.basename(x).startswith("ado_sfttrainer-")
        )
    ]

    if additional_wheels:
        log.info(
            "Discovered custom wheels which will be propagated to dynamic virtual environment: "
            f"{[os.path.basename(x) for x in additional_wheels]}"
        )
        packages.extend(additional_wheels)

    # VV: Get a ray runtime-environment which contains packages that this version of fms-hf-tuning imports

    # VV: Propagate environment variables that are related to pip
    # for example, PIP_FIND_LINKS for installing packages from a URL/directory.
    # This is useful for packages that take too long to compile from source like mamba-ssm
    env_vars = {key: name for key, name in os.environ.items() if key.startswith("PIP_")}

    runtime_env = ray_env_utils.get_ray_environment(
        packages=packages,
        packages_requiring_extra_phase=[ray_env_utils.packages_depending_on_torch()],
        env_vars=env_vars,
    )

    # VV: Need HF_HOME set so the tokenize_text() method in finetune.py can access
    # the same transformers cache that the fms-hf-tuning.sft_trainer.py script uses
    # This is useful for handling models we download from huggingface
    runtime_env["env_vars"] = runtime_env.get("env_vars", {})
    runtime_env["env_vars"]["HF_HOME"] = args.hf_home

    return runtime_env


def dynamic_name_function(
    function: Callable[..., Any], new_name: str
) -> Callable[[tuple[Any, ...], dict[str, Any]], Any]:
    """Returns a new function identical to the original, but with a new name.

    Parameters:
        function:
            The original function
        new_name:
            The name for the new function
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return function(*args, **kwargs)

    wrapper.__name__ = new_name

    # VV: Also update the __qualname__ so that the __repr__ method of the wrapper function object returns the new name
    if wrapper.__qualname__:
        parts = wrapper.__qualname__.split(".")
        parts = [*parts[:-1], new_name]
        wrapper.__qualname__ = ".".join(parts)

    return wrapper


class FinetuneContext:
    def __init__(
        self,
        args: "finetune.FineTuneArgs",
        runtime_env: dict[str, Any],
        exp: "Experiment",
        exp_params: ExperimentParameters,
        entity_space: EntitySpace,
        aim_metadata: dict[str, Any],
        log_level: int,
        extra: dict[str, Any],
        actuator_params: ActuatorParameters,
        request_id: str,
    ) -> None:
        """Helper class that holds all information related to 1 measurement on 1 entity

        Args:
            args:
                The per-worker arguments to the sfttrainer_wrapper.py for instantiating the tuning job worker(s)
            runtime_env:
                The Ray runtime environment to use for the measurement
            exp:
                The experiment to apply
            exp_params:
                The parameterisation of the experiment
            entity_space:
                The base entity definition
            aim_metadata:
                Extra metadata to store in AIM
            log_level:
                The logging level to use during the measurement
            extra:
                Additional arguments to ray.remote() when instantiating the measurement
            actuator_params:
                The actuator parameters
            request_id:
                The identifier of the MeasurementRequest object that owns the execution of this experiment
                on this entity
        """
        self.runtime_env = runtime_env
        self.exp = exp
        self.exp_params = exp_params
        self.aim_metadata = aim_metadata
        self.log_level = log_level
        self.args = args
        self.extra = extra or {}
        self.typed_parameters = actuator_params
        self.entity_space = entity_space
        self.request_id = request_id

        self.number_cpus = max(self.entity_space.number_gpus, 1) * 2

        if self.typed_parameters.num_tokens_cache_directory is not None:
            self.num_tokens_cache_dir = os.path.join(
                self.typed_parameters.data_directory,
                self.typed_parameters.num_tokens_cache_directory,
            )
        else:
            self.num_tokens_cache_dir = None

    def generate_distributed_settings(
        self,
    ) -> finetune.DistributedSettings:
        return finetune.DistributedSettings(
            backend=self.entity_space.distributed_backend,
            fsdp_sharding_strategy=self.entity_space.fsdp_sharding_strategy,
            fsdp_state_dict_type=self.entity_space.fsdp_state_dict_type,
            accelerate_config_mixed_precision=self.entity_space.accelerate_config_mixed_precision,
            accelerate_config_fsdp_transformer_layer_cls_to_wrap=(
                self.entity_space.accelerate_config_fsdp_transformer_layer_cls_to_wrap
            ),
        )

    def postprocess_metrics_tracker_metrics(
        self,
        metrics: "metrics_tracker.Metrics",
    ) -> dict[str, Any]:
        world_size = max(1, self.entity_space.number_gpus)

        return metrics.to_scalar_observations(
            distributed_backend=self.entity_space.distributed_backend,
            world_size=world_size,
        )

    def generate_method_call(
        self,
        scheduling_strategy: PlacementGroupSchedulingStrategy | None = None,
        multi_node: finetune.MultiNodeSettings | None = None,
    ) -> "ray.Actor":
        # VV: FIXME this is not a ray.Actor, what is it ?
        extra = self.extra.copy()

        if scheduling_strategy:
            extra["scheduling_strategy"] = scheduling_strategy

        number_cpus = self.number_cpus

        if multi_node:
            number_cpus //= multi_node.num_machines
            if "resources" not in extra:
                extra["resources"] = {}

            # VV: This is just for completeness as we're manually assigning tasks to bundle indices
            extra["resources"]["full-worker"] = 1

            if self.entity_space.enable_roce:
                # VV: We want to use a worker capable of RDMA over Converged Ethernet (RoCE)
                extra["resources"]["RoCE"] = 1

            if "num_gpus" in extra:
                extra["num_gpus"] //= multi_node.num_machines
                extra["resources"][
                    self.entity_space.gpu_model
                ] //= multi_node.num_machines

        return ray.remote(
            num_cpus=number_cpus,
            runtime_env=self.runtime_env,
            max_retries=0,
            **extra,
        )(
            # VV: Associate a call to launch_finetune() with its request_id.
            # This makes it trivial to find the logs of the respective Ray Task
            dynamic_name_function(
                finetune.launch_finetune,
                f"launch_finetune_{self.request_id}",
            )
        )


def get_host(pod_ip: str) -> str:
    with open(
        "/var/run/secrets/kubernetes.io/serviceaccount/namespace",
        encoding="utf-8",
    ) as f:
        namespace = f.read().strip()

    pod_ip = pod_ip.replace(".", "-")

    return f"{pod_ip}.{namespace}.pod.cluster.local"


def get_ip(host: str) -> str:
    import socket

    return socket.gethostbyname(host)


def update_dict(
    target: dict[Any, dict[Any, Any]],
    updates: dict[Any, dict[Any, Any]],
) -> None:
    """Merges 2 dictionaries of dictionaries

    Args:
        target:
            The dictionary to update
        updates:
            The dictionary containing the updated values
    """

    for key, nested_dict in updates.items():
        existing = target.get(key, {})

        for inner_key, value in nested_dict.items():
            existing[inner_key] = value

        # VV: Cover the scenario where updates contains novel keys
        target[key] = existing


@ray.remote
class SFTTrainer(ActuatorBase):
    _dir = os.path.abspath(os.path.dirname(__file__))
    identifier = identifier
    parameters_class = ActuatorParameters

    def __init__(
        self,
        queue: "MeasurementQueue",
        params: "GenericActuatorParameters",
    ) -> None:
        enable_ray_actor_coverage("sfttrainer")
        super().__init__(queue, params)
        self.log = logging.getLogger("SFTTrainer")

        try:
            # VV: HACK This is a temporary hack because the actuator parameters feature in ADO is not working atm
            if "SFTTRAINER_PARAMETERS_FILE" in os.environ:
                path = os.environ["SFTTRAINER_PARAMETERS_FILE"]

                self.log.info(f"Loading actuator parameters from {path}")
                with open(path, encoding="utf-8") as f:
                    params = yaml.safe_load(f)
                params = self.parameters_class.model_validate(params)

            # VV: Layer model_map parameters over the default ones.
            # Every other field of ActuatorConfiguration stays as is, this enables users to just override models
            # instead of specifying every single model.
            params = params.model_dump()
            default_models = copy.deepcopy(ModelMap)
            update_dict(default_models, params.get("model_map", {}))
            params["model_map"] = default_models

            self.typed_parameters = self.parameters_class.model_validate(params)

            self.log.info(f"The actuator parameters are {self.typed_parameters}")
        except pydantic.ValidationError as e:
            self.log.warning(
                f"Invalid configuration for Actuator. Pydantic error was {e}. "
                f"All experiments will be marked as Success with is_valid=False"
            )

        # VV: Keeps track of running experiments that this Actuator has launched and have not completed yet
        self.running_tasks: set[asyncio.Task] = set()

    def model_map(self) -> dict[str, dict[WeightsFormat, str]]:
        return self.typed_parameters.model_map.copy()

    @classmethod
    def _cli_args_for_measurement(
        cls,
        space: EntitySpace,
        exp_params: ExperimentParameters,
        task_uid: str,
        entity_identifier: str,
        actuator_parameters: ActuatorParameters,
    ) -> "finetune.FineTuneArgs":
        """Returns the arguments for the remote task that executes the actuator

        Args:
            space:
                The entity configuration to measure
            exp_params:
                The parameters of the experiment to use for measuring the entity
            task_uid:
                The unique identifier of the task that describes running the experiment on this entity
            entity_identifier:
                The entity identifier
            actuator_parameters:
                An instance of the ActuatorParameters() that parametrizes this measurement

        Returns:
            An instance of finetune.FineTuneArgs

        Raises:
            InvalidEntityError:
                If the entity should never be evaluated (i.e. `is_valid=0`)
            InternalInconsistencyError:
                If there's some unexpected exception - this likely means there's a bug in this code
            NotImplementedError:
                If the current implementation of the actuator cannot handle the requested entity
        """
        try:
            data_path = os.path.join(
                actuator_parameters.data_directory, DatasetMap[space.dataset_id]
            )
        except KeyError as error:
            raise NotImplementedError(
                f"References unknown dataset {space.dataset_id}"
            ) from error

        try:
            model_map = actuator_parameters.model_map[space.model_name]

        except KeyError as error:
            raise NotImplementedError(
                f'Entity is referencing unknown model "{space.model_name}", '
                f"supported models are {list(actuator_parameters.model_map)}"
            ) from error

        kwargs: dict[str, Any] = actuator_parameters.model_dump(
            exclude_none=True,
            # VV: Manually exclude fields which are not CLI args of the fms-hf-tuning wrapper
            exclude={
                "data_directory",
                "aim_dashboard_url",
                "model_map",
                "num_tokens_cache_directory",
                "match_exact_dependencies",
            },
        )

        kwargs.update(exp_params.args_for_entity_space(space, model_map=model_map))

        # VV: Rename model_max_length as max_seq_length
        space_args = space.model_dump(
            exclude={"model_max_length", "dataset_id", "model_name", "number_nodes"}
        )
        space_args["max_seq_length"] = space.model_max_length
        kwargs.update(space_args)

        # VV: Map model_name to model_name_or_path so that the
        # entity identifier does not differ when the source of the model changes (e.g. different path)
        model_name_or_path = kwargs.pop("model_name")

        kwargs["training_data_path"] = data_path

        kwargs["model_name_or_path"] = model_name_or_path

        # VV: EntitySpace has `batch_size` but FineTuneArgs has `per_device_train_batch_size` instead
        kwargs["per_device_train_batch_size"] = kwargs.pop("batch_size") // max(
            1, space.number_gpus or 1
        )

        kwargs["use_flash_attn"] = kwargs.pop("flash_attn")

        try:
            args = finetune.FineTuneArgs(**kwargs)
        except Exception as e:
            # VV: This means there's a bug with the actuator - we should always be able to parse the args following
            # the above error checks
            raise InternalInconsistencyError(str(e)) from e

        # VV: Here we fill in fields which need to propagate to FinetuneArgs BUT their definition in the
        # pydantic model (e.g. ExperimentParameter, or EntitySpace) contains `exclude=True`
        args.output_dir = os.path.join(args.output_dir, task_uid)
        args.aim_experiment = entity_identifier

        args.aim_db = actuator_parameters.aim_db
        args.number_gpus = space.number_gpus // space.number_nodes
        args.fms_hf_tuning_version = space.fms_hf_tuning_version

        return args

    async def _launch_explore_errors_runs(
        self,
        entity_id: str,
        exp_id: str,
        args: "finetune.FineTuneArgs",
        aim_metadata: dict[str, Any],
        method: "ray.actor.ActorMethod",
        distributed_settings: finetune.DistributedSettings,
        multi_node: finetune.MultiNodeSettings | None = None,
        log_level: int | None = None,
    ) -> dict[str, float]:
        metrics = {
            "f_gpu_oom": 0,
            "f_other_error": 0,
            "f_no_error": 0,
            "is_valid": True,
        }
        number_sample_runs = 5

        aim_metadata["explore_errors"] = {}

        for idx in range(number_sample_runs):
            aim_metadata["explore_errors"]["sample_idx"] = idx
            self.log.info(
                f"Launching {idx+1}/{number_sample_runs} for {exp_id} on {entity_id}"
            )
            try:
                # VV: We could run these in parallel
                _ = await method.remote(
                    args=args,
                    aim_metadata=aim_metadata,
                    count_dataset_tokens=False,
                    distributed_settings=distributed_settings,
                    log_level=log_level,
                    multi_node=multi_node,
                )
            except finetune.OutOfGPUMemoryError:
                metrics["f_gpu_oom"] += 1.0 / number_sample_runs
            except finetune.ExperimentError:
                metrics["f_other_error"] += 1.0 / number_sample_runs
            except Exception as e:
                self.log.info(
                    f"Unknown exception {e} for experiment {exp_id}[{idx}] on entity "
                    f"{entity_id} - will increase f_other_error"
                )
                self.log.info(traceback.format_exc())
                metrics["f_other_error"] += 1.0 / number_sample_runs
            else:
                metrics["f_no_error"] += 1.0 / number_sample_runs

        return metrics

    def prepare_finetune_context(
        self,
        exp: "Experiment | ParameterizedExperiment",
        entity: "Entity",
        task_uid: str,
        request_id: str,
    ) -> FinetuneContext:
        entity_values = exp.propertyValuesFromEntity(entity)

        self.log.debug(
            f"Entity {entity.identifier} translates to {json.dumps(entity_values)}"
        )

        # VV: This may raise one of the exceptions that _observations_for_experiment() may raise
        exp_params = experiment_parameters_from_experiment(
            exp, entity_values=entity_values
        )
        self.log.debug(f"With exp_params: {json.dumps(dict(exp_params))} ")

        # VV: dict(exp_params) includes fields which would have been excluded
        # by an exp_params.model_dump().
        # By definition
        for key, value in dict(exp_params).items():
            if key not in EntitySpace.model_fields:
                # VV: This is an Experiment parameter for which there is no field in the EntitySpace
                # i.e. it controls something inside the experiment which users cannot set in any way
                continue

            if key in entity_values:
                # VV: The entity overrides information in the experiment.
                # Some experiments (e.g. finetune-lora-default-r-4-a-16-fsdp-v2.1.0) have a hard-coded num_train_epochs
                # which cannot be overridden by the entity. Others support overriding the experiment defaults
                # (e.g. finetune_lora_benchmark-v1.0.0 for num_train_epochs)
                continue
            entity_values[key] = value

        self.log.info(
            f"After customization for experiment {exp.identifier} entity {entity.identifier} "
            f"translates to {json.dumps(entity_values)}"
        )

        space = EntitySpace.model_validate(entity_values)
        space.validate_and_update(exp_params=exp_params, logger=self.log)

        args = self._cli_args_for_measurement(
            space=space,
            exp_params=exp_params,
            task_uid=task_uid,
            entity_identifier=entity.identifier,
            actuator_parameters=self.typed_parameters,
        )

        import dataclasses

        self.log.info(
            f"Prepare context for experiment {exp.identifier} entity {entity.identifier} "
            f"and args {dataclasses.asdict(args)}"
        )

        runtime_env = prepare_runtime_environment(
            actuator_parameters=self.typed_parameters,
            log=self.log,
            space=space,
            args=args,
        )

        number_cpus = max(space.number_gpus, 1) * 2

        aim_metadata = {
            p.property.identifier: p.value for p in entity.constitutive_property_values
        }

        aim_metadata["experiment"] = exp.identifier
        aim_metadata["entity"] = entity.identifier

        self.log.info(
            f"Starting {exp.identifier} for {entity.identifier} with gpus={space.number_gpus}, cpus={number_cpus}, "
            f"environment {runtime_env} and args {args}"
        )

        # VV: This is so that we can query AIM for `run.$field` e.g. `run.model_name` etc
        aim_metadata = {"metadata": aim_metadata}

        extra = {}
        if space.number_gpus > 0:
            extra["num_gpus"] = space.number_gpus
            extra["resources"] = {space.gpu_model: space.number_gpus}

        self.log.info(f"The environment is {json.dumps(runtime_env)}")

        log_level = None
        if os.environ.get("LOGLEVEL"):
            loglevel = os.environ.get("LOGLEVEL")
            loglevel = {
                "DEBUG": 10,
                "INFO": 20,
                "WARNING": 30,
                "ERROR": 40,
                "CRITICAL": 50,
            }.get(loglevel, loglevel)
            log_level = loglevel

        # VV: Inject the response_template when dealing with text datasets
        if args.training_data_path.endswith(".jsonl"):
            args.response_template = "\n### Response:"

        return FinetuneContext(
            args=args,
            runtime_env=runtime_env,
            exp=exp,
            entity_space=space,
            exp_params=exp_params,
            aim_metadata=aim_metadata,
            log_level=log_level,
            extra=extra,
            actuator_params=self.typed_parameters,
            request_id=request_id,
        )

    async def _multi_node_run(
        self,
        context: FinetuneContext,
        entity: "Entity",
    ) -> "metrics_tracker.Metrics":
        if context.exp.metadata["method"] == "full-error":
            raise NotImplementedError(
                "Haven't implemented multi-node support for full-error yet"
            )

        # VV: full-worker=1 ensures that gpu worker pods land on different nodes.
        # Only nodes which request all the GPUs on the baremetal nodes have this custom resource
        bundle_resources = {
            "CPU": context.number_cpus // context.entity_space.number_nodes,
            "full-worker": 1,
        }

        if context.entity_space.number_gpus:
            bundle_resources.update(
                {
                    "GPU": context.entity_space.number_gpus
                    // context.entity_space.number_nodes,
                    context.entity_space.gpu_model: context.entity_space.number_gpus
                    // context.entity_space.number_nodes,
                }
            )

        if context.entity_space.enable_roce:
            # VV: We want to use a worker capable of RDMA over Converged Ethernet (RoCE)
            bundle_resources["RoCE"] = 1

        # VV: The idea is to schedule N tasks in parallel. One of them will act as the "main" worker.
        # On that node, we need to find a free port and then configure all workers (including the main one) to
        # connect to that port. There are actually going to be multiple workers per node 1 per GPU
        pg = ray.util.placement_group(
            [bundle_resources.copy() for _ in range(context.entity_space.number_nodes)],
            strategy="STRICT_SPREAD",
        )

        tasks = []

        self.log.info(f"Starting a multi-node run {ray.util.placement_group_table(pg)}")
        try:
            await pg.ready()

            self.log.info(
                f"Placement group for multi-node run is ready {ray.util.placement_group_table(pg)}"
            )

            method_port = ray.remote(finetune.get_available_open_port)
            port = await method_port.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                )
            ).remote()

            # VV: Find the IP of the GPU node on which we found an available port to listen on
            nodes = ray.nodes()
            pg_table = ray.util.placement_group_table(pg)

            bundles_to_node_id: dict[int, str] = {
                int(k): str(v) for k, v in pg_table["bundles_to_node_id"].items()
            }

            pod_ip = None
            for w in nodes:
                if w["NodeID"] == bundles_to_node_id[0]:
                    pod_ip = w["NodeManagerAddress"]

            if not pod_ip:
                raise ValueError(
                    f"Could not find ip of worker in bundle 0 with node id {bundles_to_node_id[0]}"
                )

            host = get_host(pod_ip)
            ip = get_ip(host)

            self.log.info(f"The host is {host} its ip is {ip} and the port is {port}")

            # VV: TODO how do we deal with errors here ?
            orig_aim_metadata = context.aim_metadata

            for bundle_idx in range(context.entity_space.number_nodes):
                context.aim_metadata = copy.deepcopy(orig_aim_metadata)
                context.aim_metadata["rank"] = bundle_idx

                pgs = PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=bundle_idx
                )
                multi_node = finetune.MultiNodeSettings(
                    port_is_local=bundle_idx == 0,
                    port=port,
                    ip=ip,
                    num_machines=context.entity_space.number_nodes,
                    machine_rank=bundle_idx,
                    nccl_ib_disable=int(not context.entity_space.enable_roce),
                )

                tasks.append(
                    self._measurement(
                        context=context,
                        entity=entity,
                        method=context.generate_method_call(
                            scheduling_strategy=pgs, multi_node=multi_node
                        ),
                        multi_node=multi_node,
                    )
                )
            results: list[metrics_tracker.Metrics] = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                ray.cancel(task, force=True, recursive=True)
            ray.util.remove_placement_group(pg)

        # VV: At this point we'll have N metrics one for each of the nodes we used. To combine them we'll:
        # 1. use the fields training and model from the main worker at rank 0
        # 2. concatenate the gpu fields
        # 3. get rid of aim_run_hash altogether (there are multiple aim hashes, one for each machine)
        # 4. combine the system metrics such as the resulting min,max,avg values correspond to min(min, min, min...)
        #      max(max, max, max...), avg(avg, avg, avg...)

        gpus = []
        for idx, r in enumerate(results):
            self.log.info(f"Raw metrics for machine {idx} are {dataclasses.asdict(r)}")
            r.filter_unused_gpus()

            # VV: Use `Raw metrics are` instead of `raw metrics are` so that we can quickly search for this
            # in the ray dashboard
            self.log.info(
                f"After filtering out unused GPUs for machine {idx} Raw metrics are {dataclasses.asdict(r)}"
            )
            gpus.extend(r.gpus)

        sm = {}

        aim_run_hash = results[0].aim_run_hash

        for name in dataclasses.asdict(results[0].system):
            _min = []
            _max = []
            _avg = []

            for r in results:
                _min.append(dataclasses.asdict(r.system)[name]["min"])
                _max.append(dataclasses.asdict(r.system)[name]["max"])
                _avg.append(dataclasses.asdict(r.system)[name]["avg"])
            sm[name] = metrics_tracker.AggregatedValues(
                avg=sum(_avg) / len(_avg), min=min(_min), max=max(_max)
            )
        system = metrics_tracker.SystemMetrics(**sm)

        hostname_gpus = {}
        for m in results:
            hostname_gpus.update(m.hostname_gpus)

        # VV: Assume that all jobs started/finished using the GPUs at the same time
        combined_metrics = copy.deepcopy(results[0])

        combined_metrics.gpus = gpus
        combined_metrics.system = system
        combined_metrics.aim_run_hash = aim_run_hash
        combined_metrics.hostname_gpus = hostname_gpus

        self.log.info(
            f"Combined Raw metrics are {dataclasses.asdict(combined_metrics)}"
        )

        return combined_metrics

    def _measurement(
        self,
        context: FinetuneContext,
        entity: "Entity",
        method: "ray.Actor",
        multi_node: finetune.MultiNodeSettings | None,
    ) -> typing.Coroutine:
        distributed_settings = context.generate_distributed_settings()

        if context.exp.metadata["method"] == "full-error":
            return self._launch_explore_errors_runs(
                args=context.args,
                aim_metadata=context.aim_metadata,
                entity_id=entity.identifier,
                exp_id=context.exp.identifier,
                method=method,
                distributed_settings=distributed_settings,
                log_level=context.log_level,
                multi_node=multi_node,
            )
        return method.remote(
            args=context.args,
            aim_metadata=context.aim_metadata,
            distributed_settings=distributed_settings,
            count_dataset_tokens=True,
            num_tokens_cache_dir=context.num_tokens_cache_dir,
            model_id=context.entity_space.model_name,
            multi_node=multi_node,
        )

    async def _observations_for_experiment(
        self,
        entity: Entity,
        exp: Experiment,
        context: FinetuneContext,
    ) -> dict[str, Any]:
        """Runs an experiment on an identity and returns the measured properties

        Args:
            entity:
                The entity to process
            exp:
                The experiment to apply on the entity
            context:
                The FinetuneContext describing the measurement

        Returns:
            A dictionary whose keys are identifiers of targetProperties that were measured (i.e. observedProperties)
            and values are the values of the target properties.

        Raises:
            InvalidEntityError:
                If the entity should never be evaluated (i.e. `is_valid=0`)
            InternalInconsistencyError:
                If there's some unexpected exception - this likely means there's a bug in this code
            NotImplementedError:
                If the current implementation of the actuator cannot handle the requested entity
        """
        try:
            if not os.path.isfile(context.args.training_data_path):
                raise NotImplementedError(
                    f"training_data_path points to path {context.args.training_data_path} which is not a file. "
                    f"Double check your DiscoverySpace, ActuatorParameters, and the file storage of your cluster. "
                    f"This measurement will be flagged as Failed so that you have the chance to repeat it in the "
                    f"future after you address this problem."
                )

            self.log.info(
                f"Launch {model_dump_all(context.entity_space)} "
                f"- each task has the args {dataclasses.asdict(context.args)}"
            )

            if not context.exp_params.multi_node:
                metrics: dict[str, Any] = await self._measurement(
                    context=context,
                    entity=entity,
                    method=context.generate_method_call(),
                    multi_node=None,
                )
            else:
                metrics: metrics_tracker.Metrics = await self._multi_node_run(
                    context=context, entity=entity
                )
        except InvalidEntityError:
            raise
        except finetune.ExperimentError as e:
            raise InvalidEntityError(str(e)) from e
        except Exception as e:
            # VV: Can't tell what the issue is, we're probably missing a feature
            raise NotImplementedError(str(e)) from e

        if isinstance(metrics, dict):
            raw_metrics = metrics
        else:
            raw_metrics = dataclasses.asdict(metrics)
        self.log.info(
            f"Raw metrics for {exp.identifier} on entity {entity.identifier} are {raw_metrics}"
        )

        if isinstance(metrics, dict):
            return metrics

        # VV: This is for experiments that measure system metrics
        try:
            return context.postprocess_metrics_tracker_metrics(metrics=metrics)
        except Exception as e:
            raise InternalInconsistencyError(
                "Exception when decoding metrics for experiment "
                + exp.identifier
                + " on entity "
                + entity.identifier
                + " error was "
                + str(e)
            ) from e

    async def _evaluate_one_entity(
        self,
        entity: Entity,
        task_uid: str,
        request: MeasurementRequest,
    ) -> str:
        """Measures one identity

        Args:
            entity:
                The entity to measure
            task_uid:
                The unique identifier of this task
            request:
                The measurement request for this particular experiment

        Returns:
            Nothing
        """
        exp_name = "*UnknownExperiment*"

        # VV: This is just to make the linter happy
        exp = None
        context: FinetuneContext | None = None
        scalar_observations: dict[str, Any] = {}

        try:

            if self.typed_parameters is None:
                # VV: Can't tell what the exact issue is here, so we're basically asking the user to inspect the logs
                # then re-run the experiments.
                raise NotImplementedError(
                    "Invalid ActuatorParameters (see earlier Warning)"
                )

            try:
                exp = catalog.experimentForReference(request.experimentReference)
                if exp is None:
                    raise KeyError(request.experimentReference)
                ## Check if the experiment has parameterization fields and create the right ParameterizedExperiment instance
                ## Note: this is necessary for getting values of optional properties that the discoverySpace overrides
                if request.experimentReference.parameterization:
                    exp = ParameterizedExperiment(
                        parameterization=request.experimentReference.parameterization,
                        **exp.model_dump(),
                    )

            except Exception as e:
                self.log.debug(f"Exception while discovering experiment: {e}")
                raise InternalInconsistencyError(e) from e
            exp_name = exp.identifier
            if exp is None:
                # VV: Pretty sure this can never happen
                raise NotImplementedError(
                    f"Unknown experiment {request.experimentReference.experimentIdentifier}"
                )

            try:
                context = self.prepare_finetune_context(
                    exp=exp,
                    entity=entity,
                    task_uid=task_uid,
                    request_id=request.requestid,
                )
            except InvalidEntityError:
                raise
            except finetune.ExperimentError as e:
                raise InvalidEntityError(str(e)) from e
            except Exception as e:
                # VV: Can't tell what the issue is, we're probably missing a feature
                raise NotImplementedError(str(e)) from e

            scalar_observations = await self._observations_for_experiment(
                entity=entity,
                exp=exp,
                context=context,
            )
            # VV: We got some measurements therefore this is a valid experiment. Also, we should not
            # re-evaluate this entity for this experiment ever again
            scalar_observations["is_valid"] = True
            request.status = MeasurementRequestStateEnum.SUCCESS
        except InvalidEntityError as e:
            self.log.warning(
                f"Experiment {exp_name} for {entity} is invalid, due to {e}. Traceback with INFO loglevel"
            )
            self.log.info(traceback.format_exc())
            # VV: Do not try to re-evaluate this point, it is definitely invalid
            scalar_observations = {"is_valid": False}
            request.status = MeasurementRequestStateEnum.SUCCESS
        except (NotImplementedError, InternalInconsistencyError, Exception) as e:
            # VV: The entity requires some yet to be implemented feature
            # OR There's a bug in the actuator.
            # OR This is an unexpected exception (this could be either of the above 2)
            # Mark the entity as failed (i.e. no properties) so that the user can retry in the future
            if isinstance(e, InternalInconsistencyError):
                self.log.warning(
                    f"Internal inconsistency error, for Experiment {exp_name} for {entity} will mark task as "
                    f"failed. If you have not enabled DEBUG logs, please do and retry. "
                    f"Record this message and the traceback that DEBUG logs print. Then contact the developers of this "
                    f"software for more information. The exception was {e}"
                )
            else:
                self.log.warning(
                    f"Experiment {exp_name} for {entity} raised an Exception, however it may succeed in "
                    f"the future, marking the entity as Failed. Enable debug level logs for more info. "
                    f"The exception was {e}. Traceback with INFO loglevel"
                )
            self.log.info(traceback.format_exc())
            # VV: mark the entity as Failed -> the user can re-evaluate this point sometime in the future
            # Don't register any properties
            request.status = MeasurementRequestStateEnum.FAILED

            # VV: TODO Figure out how to improve the reason we log for Failed measurements
            request.measurements = [
                InvalidMeasurementResult(
                    entityIdentifier=entity.identifier,
                    reason=str(e),
                    experimentReference=exp.reference,
                )
            ]
            await self._stateUpdateQueue.put_async(request, block=False)
            return request.requestid

        if context is not None and (
            (context.args.stop_after_seconds > 0.0)
            or (context.args.auto_stop_method is not None)
        ):
            # VV: Dynamically terminating the training job confuses transformers and causes it to report the wrong
            # throughput. So here we're getting rid of these values.
            scalar_observations = {
                k: v
                for k, v in scalar_observations.items()
                if k
                not in [
                    "train_tokens_per_second",
                    "train_tokens_per_gpu_per_second",
                    "train_samples_per_second",
                    "train_steps_per_second",
                ]
            }

        # VV: We got measurements -> keep those which map to targetProperties of this experiment
        measurements = []
        recorded_property_names = []
        for prop in exp.observedProperties:
            try:
                # VV: The scalar observations do not use the observedProperty identifier but rather the targetProperty
                # identifier as multiple experiments can use the exact same method to extract the same measurements
                value = scalar_observations[prop.targetProperty.identifier]
            except KeyError:
                continue

            try:
                is_finite = math.isfinite(value)
            except Exception:
                is_finite = False

            if is_finite is False:
                self.log.info(
                    f"Measured property {prop.identifier}={value} for "
                    f"the entity {entity.identifier} but the value is not finite"
                )

            self.log.info(
                f"Measured property {prop.identifier}={value} for "
                f"the entity {entity.identifier}"
            )
            recorded_property_names.append(prop.targetProperty.identifier)
            property_value = ObservedPropertyValue(
                valueType=orchestrator.schema.property_value.ValueTypeEnum.NUMERIC_VALUE_TYPE,
                value=value,
                property=prop,
            )
            measurements.append(property_value)

        metadata = {
            "aim_hash": scalar_observations.get("aim_run_hash"),
            # VV: The 3 fields below are adequate information to link this experiment with
            # metadata stored on prometheus. See metrics_tracker.py for details on field contents
            "train_time_start": scalar_observations.get("train_time_start"),
            "train_time_stop": scalar_observations.get("train_time_stop"),
            "hostname_gpus": scalar_observations.get("hostname_gpus"),
            "total_steps": scalar_observations.get("total_steps"),
        }

        if metadata["aim_hash"] and self.typed_parameters.aim_dashboard_url:
            aim_dashboard_url = "/".join(
                (
                    self.typed_parameters.aim_dashboard_url.rstrip("/"),
                    "runs",
                    metadata["aim_hash"],
                )
            )

            metadata["aim_dashboard_url"] = aim_dashboard_url

        measurement_result = ValidMeasurementResult(
            entityIdentifier=entity.identifier,
            measurements=measurements,
            metadata=metadata,
        )
        request.measurements = [measurement_result]

        if not recorded_property_names:
            self.log.warning(
                f"THIS EXPERIMENT RUN IS WEIRD, IT DID NOT RECORD ANY OBSERVED PROPERTIES AND YOU SHOULD "
                f"DEBUG IT {exp.identifier} for {entity.identifier} - "
                f"will mark it as failed"
            )
            request.status = MeasurementRequestStateEnum.FAILED

        await self._stateUpdateQueue.put_async(request, block=False)

        return request.requestid

    async def submit(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
    ) -> list[str]:
        """Submits the entities for measurement by experiment via the receiver

        :param entities: A list of Entity representing the entities to be measured
        :param experimentReference: An ExperimentReference defining the
            experiment to run on the entities
        :param requesterid: Unique identifier of the requester of this measurement
        :param requestIndex: The index of this request i.e. this is the ith request by
            the requester in the current run

        Returns:

            A list with the ids of the experiments it submitted.
            NOTE: An actuator may not submit all entities in one batch.
        """

        import uuid

        requests = []

        # AP 29/08/2024:
        # Before doing anything, we must ensure the experiment is not deprecated
        experiment = catalog.experimentForReference(experimentReference)
        if experiment.deprecated:
            raise DeprecatedExperimentError(
                f"Experiment {experiment.identifier} is deprecated"
            )

        for index in range(len(entities)):
            entity = entities[index]

            request = MeasurementRequest(
                operation_id=requesterid,
                requestIndex=requestIndex,
                experimentReference=experimentReference,
                entities=[entity],
                requestid=str(uuid.uuid4())[:8],
            )
            # VV: This is to support executing the same operation on multiple experiments at the same time
            task_uid = os.path.join(requesterid, f"{index}-{request.requestid}")
            task = asyncio.create_task(
                self._evaluate_one_entity(entity, task_uid, request)
            )

            # VV: Record task and have it auto-remove itself from the set of running tasks when it's done
            self.running_tasks.add(task)
            task.add_done_callback(lambda t: self.running_tasks.remove(t))

            requests.append(request)

        return [r.requestid for r in requests]

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> orchestrator.modules.actuators.catalog.ExperimentCatalog:
        return catalog
