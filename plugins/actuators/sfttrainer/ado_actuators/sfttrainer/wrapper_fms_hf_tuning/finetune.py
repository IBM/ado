# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import copy
import enum
import functools
import logging
import sys
import time
import typing

if typing.TYPE_CHECKING:
    import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.constants as constants
    import transformers
    from transformers.tokenization_utils_base import BatchEncoding

    from .callbacks import metrics_tracker

import dataclasses
import os

# VV: Env vars this script uses:
# HOME -> Must set this to something like `/tmp` because aim is attempting to generate files under `~/.aim_profile`
#         when importing `aim.utils.tracking` and that fails as the Process in the container does not have
#         write permissions to the home folder.


class ExperimentError(Exception):
    """Raising this exception indicates that this Measurement will always fail. The measurement will be
    marked as "invalid" i.e. its observed property `is_valid` will be set to 0.

    Create more exceptions that inherit this one. Only have 1 parameter to the __init__() method and make sure
    you return a human-readable string in your `__str__()` implementation.

    If you include more than 1 parameter to your __init__() method make sure you **only** raise your exception
    by using positional parameters instead of named ones. Otherwise, ray will raise this kind of exceptions when
    pickling/unpickling your exceptions:

    RuntimeError: Failed to unpickle serialized exception
    ....
    TypeError: YourException.__init__() missing 2 required positional arguments: 'first_arg' and 'second_arg'
    """


class NumberOfExpertsNotDivisibleByEpDegreeError(ExperimentError):
    def __init__(self, underlying_error: str) -> None:
        self.underlying_error = underlying_error

    def __str__(self) -> str:
        return self.underlying_error


class AccelerateError(ExperimentError):
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


class OutOfGPUMemoryError(ExperimentError):
    def __init__(self, underlying_error: Exception | str | None = None) -> None:
        self.underlying_error = underlying_error

    def __str__(self) -> str:
        return f"Out of GPU memory, underlying error was {self.underlying_error}"


class NCCLError(ExperimentError):
    def __init__(self, underlying_error: Exception | str | None = None) -> None:
        self.underlying_error = underlying_error

    def __str__(self) -> str:
        return f"NCCL error, underlying error was {self.underlying_error}"


class UnhandledError(NotImplementedError):
    def __init__(self, underlying_error: Exception | str | None = None) -> None:
        self.underlying_error = underlying_error

    def __str__(self) -> str:
        return (
            f"Unhandled experiment error, underlying error was {self.underlying_error}"
        )


@dataclasses.dataclass
class MultiNodeSettings:
    port_is_local: bool = False
    num_machines: int = 1
    machine_rank: int = 0
    ip: str | None = None
    port: int | None = None
    nccl_ib_disable: int = 0


@dataclasses.dataclass
class DistributedSettings:
    backend: typing.Literal["FSDP", "DDP"] | None = dataclasses.field(
        default=None,
        metadata={
            "help": "Which backend to use. FSDP or DDP for multi-process experiments and "
            "None for single-process experiments"
        },
    )

    fsdp_sharding_strategy: typing.Literal[
        "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD", "HYBRID_SHARD_ZERO2"
    ] = dataclasses.field(
        default="FULL_SHARD",
        metadata={
            "help": "[1] FULL_SHARD (shards optimizer states, gradients and parameters), "
            "[2] SHARD_GRAD_OP (shards optimizer states and gradients), "
            "[3] NO_SHARD (DDP), "
            "[4] HYBRID_SHARD (shards optimizer states, gradients and parameters within each node "
            "while each node has full copy - equivalent to FULL_SHARD for single-node runs), "
            "[5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients within each node while each node has "
            "full copy). For more information, please refer the official PyTorch docs."
        },
    )

    fsdp_state_dict_type: (
        typing.Literal["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"]
        | None
    ) = dataclasses.field(
        default="FULL_STATE_DICT",
        metadata={
            "help": "[1] FULL_STATE_DICT, [2] LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT"
        },
    )

    fsdp_use_orig_params: bool = dataclasses.field(
        default=True,
        metadata={
            "help": "If True, allows non-uniform `requires_grad` during init, which means support for "
            "interspersed frozen and trainable parameters. (useful only when `use_fsdp` flag is passed)."
        },
    )

    accelerate_config_mixed_precision: typing.Literal["no", "fp16", "bf16", "fp8"] = (
        dataclasses.field(
            default="no",
            metadata={
                "help": "Whether or not to use mixed precision training. Choose from 'no', 'fp16', 'bf16' or 'fp8'. "
                "'fp8' requires the installation of transformers-engine."
            },
        )
    )

    accelerate_config_fsdp_transformer_layer_cls_to_wrap: str | None = (
        dataclasses.field(
            default=None,
            metadata={
                "help": "List of transformer layer class names (case-sensitive) to wrap, e.g, GraniteDecoderLayer, "
                "BertLayer, GPTJBlock, T5Block ... (useful only when fsdp flag is passed)"
            },
        )
    )


@dataclasses.dataclass
class FineTuneArgs:
    """The actuator wrapper uses these options to control the experiment"""

    # VV: Misc options
    hf_home: str = dataclasses.field(
        default="/hf-models-pvc/huggingface_home",
        metadata={
            "help": "To configure where huggingface_hub will locally store data. In particular, "
            "your token and the cache will be stored in this folder. "
            "This is actually an environment variable, not a command-line argument (HF_HOME)"
        },
    )

    aim_db: str | None = dataclasses.field(
        default=None,
        metadata={"help": "The AIM endpoint"},
    )

    aim_experiment: str = dataclasses.field(
        default=None,
        metadata={"help": "The name of the AIM experiment"},
    )

    fms_hf_tuning_version: str = dataclasses.field(
        default=None,
        metadata={
            "help": "The version of fms-hf-tuning to use - controls which wrapper to use "
            "as well as python dependencies"
        },
    )

    # VV: Ray options
    number_gpus: int = dataclasses.field(
        default=1, metadata={"help": "How many GPUs to use"}
    )

    # VV: Options that match arguments of stf_trainer.py
    gradient_accumulation_steps: int = dataclasses.field(
        default=4,
        metadata={
            "help": "Number of update steps to accumulate before performing a backward/update pass."
        },
    )

    gradient_checkpointing: bool | None = dataclasses.field(
        default=None,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of "
            "slower backward pass."
        },
    )

    max_steps: int = dataclasses.field(
        default=-1,
        metadata={
            "help": "If > 0: set total number of training steps to perform. Override num_train_epochs."
        },
    )
    num_train_epochs: float = dataclasses.field(
        default=1.0, metadata={"help": "Total number of training epochs to perform."}
    )
    stop_after_seconds: float | None = dataclasses.field(
        default=None,
        metadata={
            "help": "If set, the optimizer will be asked to stop after the specified time elapses. "
            "The check is performed after the end of each training step."
        },
    )

    auto_stop_method: "constants.AutoStopMethod | None" = dataclasses.field(
        default=None,
        metadata={
            "help": "The default value is `None`. This parameter defines the method used to automatically "
            "stop the fine-tuning job. Supported values are `WARMUP_60S_STABLE_120S_OR_10_STEPS` and "
            "`None`. If set to `WARMUP_60S_STABLE_120S_OR_10_STEPS`, the job stops after spending at least "
            "60 seconds in the warmup phase plus the longer of 120 seconds or the duration of 10 "
            "optimization steps. This method excludes the first 60 seconds of training when calculating "
            "throughput and system metrics."
        },
    )

    peft_method: str | None = dataclasses.field(
        default="pt",
        metadata={
            "help": "The method to use, either pt, lora, or None (full fine tune)"
        },
    )

    model_name_or_path: str = dataclasses.field(
        default="hf-tiny-model-private/tiny-random-BloomForCausalLM",
        metadata={"help": "Path or huggingface id of model to use"},
    )

    per_device_train_batch_size: int = dataclasses.field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."},
    )
    torch_dtype: str = dataclasses.field(
        default="float32", metadata={"help": "Torch DTYPE"}
    )
    max_seq_length: int = dataclasses.field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    training_data_path: str = dataclasses.field(
        default="dataset/artificial/common_en_news_combined_512-preprocessed.jsonl",
        metadata={"help": "Path to the training data in JSONL format."},
    )
    output_dir: str = dataclasses.field(
        default="output",
        metadata={
            "help": "The output directory where the model predictions and "
            "checkpoints will be written."
        },
    )

    fp16: bool | None = dataclasses.field(
        default=None,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )

    use_flash_attn: bool = dataclasses.field(
        default=True,
        metadata={"help": "Use Flash attention v2 from transformers, default is True"},
    )

    # VV: Options for lora
    r: int = 8
    lora_alpha: int = 32
    target_modules: list[str] = dataclasses.field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={
            "help": "The names of the modules to apply LORA to. LORA selects modules which either completely match or "
            'end with one of the strings. If the value is ["all-linear"], then LORA selects all linear and Conv1D '
            "modules except for the output layer."
        },
    )
    # bias = "none" # VV: This is a hardcoded constant in LoraConfig, we cannot set it here
    lora_dropout: float = 0.05

    # VV: Options for fused_ops_and_kernels
    fast_kernels: list[str] | None = dataclasses.field(default=None)
    fused_lora: list[str] | None = dataclasses.field(default=None)

    # VV: options for AttentionAndDistributedPackingConfig
    padding_free: list[str] | None = None

    # VV: This is a [str, bool] for the parameters (kernel, from_quantized)
    auto_gptq: list[str | bool] | None = None

    # VV: This is a [int] for the parameters (ep_degree)
    fast_moe: list[int] | None = dataclasses.field(
        default=None,
        metadata={
            "help": "Configures the amount of expert parallel sharding. "
            "world_size (i.e. number_gpus) must be divisible by it"
        },
    )

    # VV: TAG: @HF_RAM_Efficient_Training
    # VV: These are arguments we added to replicate huggingface blogpost
    optim: str | None = dataclasses.field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )

    bf16: bool | None = dataclasses.field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )

    gradient_checkpointing_use_reentrant: bool | None = dataclasses.field(
        default=None,
        metadata={
            "help": "Specify whether to use the activation checkpoint variant that requires reentrant autograd. "
            "This parameter should be passed explicitly. In version 2.5 we will raise an exception "
            "if use_reentrant is not passed. If use_reentrant=False, checkpoint will use an implementation "
            "that does not require reentrant autograd. This allows checkpoint to support additional functionality, "
            "such as working as expected with torch.autograd.grad and support for keyword arguments input "
            "into the checkpointed function."
        },
    )

    # VV: For image-to-text (vision) models
    dataset_text_field: str | None = dataclasses.field(
        default=None,
        metadata={
            "help": "Training dataset text field containing single sequence. \
                        Either the dataset_text_field \
                        or data_formatter_template need to be supplied. \
                        For running vision language model tuning pass the column name for text data."
        },
    )

    dataset_image_field: str | None = dataclasses.field(
        default=None,
        metadata={"help": "For running vision language model tuning pass \
                the column name of the image data in the dataset."},
    )

    remove_unused_columns: bool | None = dataclasses.field(
        default=True,
        metadata={
            "help": "Remove columns not required by the model when using an nlp.Dataset."
        },
    )

    dataset_kwargs_skip_prepare_dataset: bool | None = dataclasses.field(
        default=False,
        metadata={"help": "When True, configures trl to skip preparing the dataset"},
    )

    # VV: Data args
    response_template: str | None = dataclasses.field(
        default=None,
        metadata={"help": "Response template, separator to train on completions only"},
    )


@dataclasses.dataclass
class HardcodedArgs:
    """Bunch of HardcodedArgs which I got from a script that Anh used, we might want to turn
    some of these into proper arguments, like for example output_dir and data_path
    """

    # VV: Training args
    log_level: str | None = dataclasses.field(
        default="info",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'."
            ),
        },
    )

    eval_strategy: "transformers.IntervalStrategy | str" = dataclasses.field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    save_strategy: "transformers.IntervalStrategy | str" = dataclasses.field(
        default="no",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    learning_rate: float = dataclasses.field(
        default=1e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = dataclasses.field(
        default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    warmup_ratio: float = dataclasses.field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    lr_scheduler_type: "transformers.SchedulerType | str" = dataclasses.field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )
    logging_steps: float = dataclasses.field(
        default=1,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    include_tokens_per_second: bool | None = dataclasses.field(
        default=True,
        metadata={
            "help": "If set to `True`, the speed metrics will include `tgs` (tokens per second per device)."
        },
    )
    packing: bool = dataclasses.field(
        default=False,
        metadata={"help": "Packing to be enabled in SFT Trainer, default is False"},
    )


def get_available_open_port() -> int:
    """Utility method to find a socket port that is available on the system"""
    import contextlib
    import socket

    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("0.0.0.0", 0))  # noqa: S104
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def extract_metrics(aim_info_path: str, number_gpus: int) -> "metrics_tracker.Metrics":
    import json

    with open(aim_info_path, encoding="utf-8") as f:
        aim_info: dict[str, typing.Any] = json.load(f)

    if "error" in aim_info:
        exc = aim_info.get("exception")
        if aim_info["error"] == "OutOfGPUMemoryError":
            raise OutOfGPUMemoryError(underlying_error=exc)
        if aim_info["error"] == "NCCLError":
            raise NCCLError(underlying_error=exc)
        raise UnhandledError(underlying_error=exc)

    import logging

    from .callbacks import metrics_tracker

    log = logging.getLogger("actuator")
    log.info(f"The metrics were {json.dumps(aim_info)}")

    return metrics_tracker.Metrics.from_aim_info_dict(
        aim_info=aim_info, num_gpus=number_gpus
    )


def _finetune_launch_kernel(
    args: FineTuneArgs,
    aim_metadata: dict[str, typing.Any] | None,
    multi_node: MultiNodeSettings,
    distributed_settings: DistributedSettings,
    working_directory: str,
) -> "metrics_tracker.Metrics":
    log = logging.getLogger("launch")

    if args.fast_moe and isinstance(args.fast_moe[0], int) and args.fast_moe[0] > 0:
        ep_degree = args.fast_moe[0]

        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(
                args.model_name_or_path, low_cpu_mem_usage=True
            )
            num_local_experts = config.num_local_experts
        except Exception as e:
            # VV: something went wrong when we tried to extract the number of local experts, ignore the error
            # and let fms-hf-tuning deal with it
            log.info(
                f"Ran into exception {e} while checking whether num_local_experts is divisible by {ep_degree=} "
                f"- ignoring error"
            )
        else:
            if num_local_experts % ep_degree != 0:
                raise NumberOfExpertsNotDivisibleByEpDegreeError(
                    f"The {num_local_experts=} are not divisible by {ep_degree=}"
                )

    # VV: Convert args to a dictionary which we'll then use to put together the commandline of
    # `accelerate launch` or python

    if args.aim_db is None:
        args = copy.deepcopy(args)
        args.aim_db = os.path.join(working_directory, "ephemeral_aim")
        log.info(
            f"aim_db is unset, will use an ephemeral aim repository at {args.aim_db}"
        )

    cmdline_args = generate_arguments_sft_trainer(args)
    log.info(
        f"Evaluate {dataclasses.asdict(args)} and multi_node {dataclasses.asdict(multi_node)}"
    )

    import json
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="aim_info_",
        delete=True,
        dir=working_directory,
    ) as f:
        aim_info_path = f.name

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="aim_metadata_",
        delete=False,
        dir=working_directory,
    ) as f:
        aim_metadata_path = f.name
        json.dump(aim_metadata, f)

    # VV: We're using this file as a thin wrapper to the main method in sft_trainer.py to catch
    # exceptions (e.g. GPU OOM) and report system metrics, it parses cmdline-args and invokes train()
    wrapper_script = os.path.join(
        os.path.dirname(__file__), "scripts", "wrapper_sfttrainer.py"
    )

    num_processes = max(1, args.number_gpus) * max(1, multi_node.num_machines)

    if distributed_settings.backend is not None:
        if multi_node.port is None:
            multi_node.port = get_available_open_port()
        elif multi_node.port_is_local:
            # VV: double check that the port is available
            import contextlib
            import socket

            # VV: If this node cannot bind the chosen port we'll treat this as a Transient exception
            with contextlib.closing(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ) as s:
                s.bind(("0.0.0.0", multi_node.port))  # noqa: S104

        # VV: Accelerate refers to DDP with the name "MULTI_GPU"
        backend_name_map = {"FSDP": "FSDP", "DDP": "MULTI_GPU"}[
            distributed_settings.backend
        ]
        accelerate_config = {
            "compute_environment": "LOCAL_MACHINE",
            "debug": False,
            # VV: this seems to be related to TPUs:
            # https://github.com/huggingface/accelerate/blob/4b6be8991059f39a8df8893333d11c54bc51fc60/
            # src/accelerate/commands/config/config_args.py#L205
            "downcast_bf16": "no",
            "distributed_type": backend_name_map,
            "machine_rank": multi_node.machine_rank,
            "main_training_function": "main",
            "mixed_precision": distributed_settings.accelerate_config_mixed_precision,
            "num_machines": multi_node.num_machines,
            "rdzv_backend": "static",
            "same_network": True,
            "tpu_env": [],
            "tpu_use_cluster": False,
            "tpu_use_sudo": False,
            "use_cpu": False,
            "main_process_port": multi_node.port,
            "num_processes": num_processes,
        }

        if multi_node.ip:
            accelerate_config["main_process_ip"] = multi_node.ip

        if distributed_settings.backend == "FSDP":
            accelerate_config["fsdp_config"] = {
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "fsdp_backward_prefetch": "BACKWARD_PRE",
                "fsdp_forward_prefetch": False,
                "fsdp_offload_params": False,
                "fsdp_sharding_strategy": distributed_settings.fsdp_sharding_strategy,
                "fsdp_state_dict_type": distributed_settings.fsdp_state_dict_type,
                "fsdp_cpu_ram_efficient_loading": True,
                "fsdp_sync_module_states": True,
                "fsdp_use_orig_params": distributed_settings.fsdp_use_orig_params,
            }

            if (
                distributed_settings.accelerate_config_fsdp_transformer_layer_cls_to_wrap
                is not None
            ):
                accelerate_config["fsdp_config"][
                    "fsdp_transformer_layer_cls_to_wrap"
                ] = (
                    distributed_settings.accelerate_config_fsdp_transformer_layer_cls_to_wrap
                )

        log.info(f"Using the accelerate config {json.dumps(accelerate_config)}")

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="accelerate_config_",
            delete=False,
            dir=working_directory,
        ) as config_file:
            config_file.write(json.dumps(accelerate_config))

        command = [
            "accelerate",
            "launch",
            "--config_file",
            config_file.name,
            "--num_processes",
            str(num_processes),
            wrapper_script,
        ]
    else:
        command = ["python", wrapper_script]

    cmdline_args["aim_metadata_path"] = aim_metadata_path

    for key, value in cmdline_args.items():
        if isinstance(value, enum.Enum):
            command.extend((f"--{key}", value.value))
        elif isinstance(value, dict):
            # VV: see https://github.com/huggingface/transformers/pull/30227/
            command.extend((f"--{key}", json.dumps(value)))
        elif isinstance(value, list) and not isinstance(value, str):
            # VV: this is to convert things like `target_modules = ["q_proj", "c_proj"]`
            # into `--target_modules q_proj c_proj`
            command.extend(
                [f"--{key}"]
                + [x.value if isinstance(x, enum.Enum) else str(x) for x in value]
            )
        else:
            command.extend((f"--{key}", str(value)))

    command.extend(
        [
            "--aim_info_path",
            aim_info_path,
            "--aim_info_aggregate_metrics",
            "true",
        ]
    )

    log.info(f"Command is {command}")

    import importlib.metadata

    installed_packages = importlib.metadata.distributions()
    installed_packages = sorted(
        [
            "{}=={}".format(pkg.metadata["Name"], pkg.metadata["Version"])
            for pkg in installed_packages
        ]
    )
    log.info(f"Installed packages {installed_packages}")

    import subprocess

    env = os.environ.copy()
    env["HF_HOME"] = args.hf_home

    # VV: When this is 1, RDMA over Converged Ethernet (RoCE) is Disabled
    env["NCCL_IB_DISABLE"] = str(multi_node.nccl_ib_disable)

    if "LOGLEVEL" in env:
        env["LOGLEVEL"] = env["LOGLEVEL"].upper()

    log.info(f"Environment variables {env}")
    proc = subprocess.Popen(  # noqa: S603
        command,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
        cwd=working_directory,
    )

    proc.wait()

    if proc.returncode == 0:
        return extract_metrics(aim_info_path, args.number_gpus)

    # VV: if we got here then something broke
    errors = {}

    for worker in range(num_processes):
        try:
            if distributed_settings.backend is None:
                # VV: the DP backend doesn't use accelerate
                path = aim_info_path
            else:
                path = "_".join((aim_info_path, str(worker)))

            m = extract_metrics(path, args.number_gpus)
        except FileNotFoundError:  # noqa: PERF203
            log.info(f"Worker {worker} did not record any error under {path}")
        except ExperimentError as e:
            log.warning(f"Worker {worker} ran into {e}")
            errors[worker] = e
        except UnhandledError as e:
            log.warning(f"Worker {worker} ran into unhandled {e}")
            errors[worker] = e
        except Exception as e:
            log.warning(f"Worker {worker} ran into unknown {e}")
            errors[worker] = e
        else:
            raise NotImplementedError(
                f"Expected worker {worker} to fail but it actually produced {dataclasses.asdict(m)}"
            )

    if not errors:
        raise UnhandledError("Experiment failed but no worker reported any error")

    for worker, e in errors.items():
        if isinstance(e, ExperimentError):
            log.info(f"Reporting ExperimentError {e} from worker {worker}")
            raise e

    # VV: No worker ran out of GPU memory just report any error
    log.info(
        "No worker ran out of a known ExperimentError, propagate "
        "the error that a random worker reported"
    )
    raise next(iter(errors.values()))


def _update_num_tokens_cache_for_model_and_dataset(
    cache_file: str,
    tokens_per_sample: list[int],
    model_id: str,
    path_data: str,
) -> None:
    import json

    parent_dir = os.path.dirname(cache_file)

    log = logging.getLogger("tokens_cache")
    # VV: We know that we'd like to use the cache and that we could not find
    # useful data in it. Therefore, we populate the relevant cache file here
    try:
        for _ in range(5):
            if not os.path.isdir(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            with open(cache_file, "w") as f:
                json.dump(tokens_per_sample, f)
            # VV: Verify that we actually stored what we think we stored (there could be multiple
            # tasks populating the cache and them corrupting each other's results)
            with open(cache_file) as f:
                fresh = json.load(f)

            if fresh == tokens_per_sample:
                log.info(f"Populated the cache file {cache_file} successfully")
                break
            log.warning(
                f"The cache file {cache_file} is corrupted, will try to recreate"
            )
            time.sleep(5)

    except Exception as e:
        log.warning(
            f"Could not cache the num tokens using the tokenizer of {model_id} for "
            f"dataset {path_data} due to {e}"
        )


def _load_num_tokens_cache_for_model_and_dataset(
    path_data: str,
    cache_file: str,
    model_id: str | None,
    num_tokens_cache_dir: str | None,
) -> list[int]:
    import json

    num_tokens = []

    log = logging.getLogger("sft_trainer:cache")

    try:
        os.makedirs(num_tokens_cache_dir, exist_ok=True)

        with open(cache_file, "rb") as f:
            num_tokens = json.load(f)
            if isinstance(num_tokens, list) is False:
                raise NotImplementedError(
                    f"Unknown type of num_tokens {type(num_tokens)}"
                )

        log.info(
            f"Loaded cached num_tokens with tokenizer {model_id} and dataset {path_data}"
        )
    except FileNotFoundError:
        log.info(
            f"No cached number of tokens with tokenizer {model_id} and dataset {path_data} in {cache_file}"
        )
    except Exception as e:
        log.info(
            f"Could not parse the cached num tokens due to {e} - will compute number of tokens using "
            f"the tokenizer of {model_id} for dataset {path_data}",
        )

    return num_tokens


def calculate_tokens_in_image_text_dataset(
    path_model: str,
    path_data: str,
    dataset_text_field: str,
) -> list[int]:
    import pandas as pd
    from datasets import Dataset
    from transformers import AutoProcessor

    df = pd.read_parquet(path_data)
    dataset = Dataset.from_pandas(df)

    processor = AutoProcessor.from_pretrained(path_model)

    def tokenize_samples(sample: dict) -> "BatchEncoding":
        return processor.apply_chat_template(
            sample[dataset_text_field],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    tokenized_dataset = dataset.map(lambda x: tokenize_samples(x), batched=False)
    return [len(sample["input_ids"][0]) for sample in tokenized_dataset]


def calculate_tokens_in_text_dataset(
    path_model: str,
    path_data: str,
) -> list[int]:
    from transformers import AutoTokenizer

    log = logging.getLogger("sft_trainer")
    tokenizer = AutoTokenizer.from_pretrained(path_model)
    special_tokens_dict = {}

    DEFAULT_PAD_TOKEN = "<PAD>"  # noqa: S105
    DEFAULT_EOS_TOKEN = "</s>"  # noqa: S105
    DEFAULT_BOS_TOKEN = "<s>"  # noqa: S105
    DEFAULT_UNK_TOKEN = "<unk>"  # noqa: S105

    if tokenizer.pad_token is None:
        log.warning("PAD token set to default, missing in tokenizer")
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        log.warning("EOS token set to default, missing in tokenizer")
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        log.warning("BOS token set to default, missing in tokenizer")
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        log.warning("UNK token set to default, missing in tokenizer")
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    import json

    import tqdm

    num_tokens = []

    with open(path_data) as f:
        for line in tqdm.tqdm(
            f,
            desc="Counting tokens in samples of dataset",
            file=sys.stdout,
            mininterval=5.0,
        ):
            data = json.loads(line)
            decoded = tokenizer.encode(data["output"], padding=True)
            num_tokens.append(len(decoded))
            del decoded

    return num_tokens


@functools.cache
def get_cache_file_for_tokens_per_sample(
    path_data: str,
    num_tokens_cache_dir: str | None,
    model_id: str,
) -> str | None:
    if num_tokens_cache_dir is None:
        return None

    # VV: since we may update the contents of a dataset
    # we use the md5 hash of the file as part of the cache id
    import hashlib

    digest = hashlib.md5(usedforsecurity=False)

    with open(path_data, "rb") as f:
        b = f.read(32768)
        while b:
            digest.update(b)
            b = f.read(32768)

    ds_name = os.path.splitext(os.path.basename(path_data))[0]

    return os.path.join(
        num_tokens_cache_dir,
        f"num-tokens.{model_id}.for.{ds_name}.{digest.hexdigest()}.json",
    )


def get_tokens_per_sample_in_dataset(
    path_model: str,
    path_data: str,
    model_id: str | None,
    num_tokens_cache_dir: str | None,
    dataset_text_field: str,
) -> list[int]:
    """Returns the tokens per sample for each sample in a dataset

    Args:
        path_model:
            path or name of model, the method uses the model's tokenizer
        path_data:
            path to the dataset
        model_id:
            the identifier of the model (i.e. the name that the SFTTrainer actuator uses to refer to1 his model)
        num_tokens_cache_dir:
            directory in which we store cached information about the number of tokens for
            the entries of a dataset. If this is not None and there's no cache file already, then this method
            will produce one
        dataset_text_field:
            Training dataset text field containing single sequence.  Either the dataset_text_field
            or data_formatter_template need to be supplied.
            For running vision language model tuning pass the column name for text data.
    Returns:
        An array containing the number of samples for each sample in a dataset
    """
    log = logging.getLogger("sft_trainer")
    # VV: Calculating the tokens for every measurement can be expensive, so we use a cache of number of tokens
    cache_file = get_cache_file_for_tokens_per_sample(
        path_data=path_data,
        num_tokens_cache_dir=num_tokens_cache_dir,
        model_id=model_id,
    )
    tokens_per_sample = []

    if cache_file and os.path.exists(cache_file):
        start = time.time()
        tokens_per_sample = _load_num_tokens_cache_for_model_and_dataset(
            path_data=path_data,
            model_id=model_id,
            cache_file=cache_file,
            num_tokens_cache_dir=num_tokens_cache_dir,
        )
        log.warning(
            f"It took {time.time() - start} seconds to search/load the num_tokens cache"
        )

    if not tokens_per_sample:
        start = time.time()
        if cache_file:
            log.info(
                "Will tokenize the dataset and cache the results to speedup future measurements"
            )
        else:
            log.info(
                "Will tokenize the dataset but the cache is disabled, future measurements will "
                "also tokenize the dataset"
            )
        if path_data.endswith(".jsonl"):
            tokens_per_sample = calculate_tokens_in_text_dataset(
                path_model=path_model,
                path_data=path_data,
            )
        elif path_data.endswith(".parquet"):
            tokens_per_sample = calculate_tokens_in_image_text_dataset(
                path_model=path_model,
                path_data=path_data,
                dataset_text_field=dataset_text_field,
            )
        else:
            raise NotImplementedError(
                f"Unsupported file extension for dataset {path_data}"
            )

        log.debug(
            f"It took {time.time() - start} seconds to tokenize the dataset {path_data} "
            f"with the tokenizer of {model_id}"
        )

        if cache_file and tokens_per_sample:
            _update_num_tokens_cache_for_model_and_dataset(
                cache_file=cache_file,
                tokens_per_sample=tokens_per_sample,
                model_id=model_id,
                path_data=path_data,
            )
        else:
            log.info(f"Will not cache the tokens_per_sample in {cache_file}")

    return tokens_per_sample


def tokenize_text(
    path_model: str,
    path_data: str,
    max_seq_length: int,
    model_id: str | None,
    num_tokens_cache_dir: str | None,
    num_entries: int,
    skip_entries: int,
    dataset_text_field: str,
) -> int:
    """Counts the tokens in a dataset that was used to train a model

    This method takes into account how many entries the experiment processed

    Args:
        path_model:
            path or name of model, the method uses the model's tokenizer
        path_data:
            path to the dataset
        max_seq_length:
            the size of the context (i.e. the param of --max_seq_length for sft_trainer.py)
        model_id:
            the identifier of the model (i.e. the name that the SFTTrainer actuator uses to refer to1 his model)
        num_tokens_cache_dir:
            directory in which we store cached information about the number of tokens for
            the entries of a dataset. If this is not None and there's no cache file already, then this method
            will produce one
        num_entries:
            How many entries of the dataset to consider during the computation of the number of tokens
            in the dataset. If this value is less than 0 then num_entries is set to the total number of entries
            in the dataset
        skip_entries:
            Number of first entries to skip. For example, to account for a warmup phase
        dataset_text_field:
            Training dataset text field containing single sequence.  Either the dataset_text_field
            or data_formatter_template need to be supplied.
            For running vision language model tuning pass the column name for text data.
    Returns:
        How many tokens were used to train a model
    """
    from transformers import AutoTokenizer

    log = logging.getLogger("sft_trainer")
    log.info(
        f"Tokenizing dataset {path_data} for model {path_model} with num_entries={num_entries}"
    )

    tokens_per_sample = get_tokens_per_sample_in_dataset(
        path_model=path_model,
        path_data=path_data,
        model_id=model_id,
        num_tokens_cache_dir=num_tokens_cache_dir,
        dataset_text_field=dataset_text_field,
    )

    tokenizer = AutoTokenizer.from_pretrained(path_model)
    # VV: When a model doesn't have an inherent model_max_length HF sets this value to `VERY_LARGE_INTEGER=1e30`
    tokenizer_model_max_length = tokenizer.model_max_length

    if num_entries < 0:
        num_entries = len(tokens_per_sample)

    all_tokens = aggregate_tokens_of_samples(
        tokens_per_sample=tokens_per_sample,
        samples=num_entries,
        max_seq_length=max_seq_length,
        tokenizer_model_max_length=tokenizer_model_max_length,
    )

    if skip_entries:
        all_tokens -= aggregate_tokens_of_samples(
            tokens_per_sample=tokens_per_sample,
            samples=skip_entries,
            max_seq_length=max_seq_length,
            tokenizer_model_max_length=tokenizer_model_max_length,
        )

    return all_tokens


def aggregate_tokens_of_samples(
    tokens_per_sample: list[int],
    samples: int,
    max_seq_length: int,
    tokenizer_model_max_length: int,
) -> int:
    """Calculate the total number of tokens for a subset of a dataset.

    Args:
        tokens_per_sample:
            A list where each element represents the number of tokens in a sample.
        samples:
            The total number of samples to aggregate.
        max_seq_length:
            The maximum sequence length allowed.
        tokenizer_model_max_length:
            The maximum length supported by the tokenizer model.

    Returns:
        int: The total number of tokens for the given samples.
    """

    full_epochs = samples // len(tokens_per_sample)

    sum_tokens = 0

    if full_epochs > 0:
        sum_tokens = (
            sum(
                min(entry, max_seq_length, tokenizer_model_max_length)
                for entry in tokens_per_sample
            )
            * full_epochs
        )

    if samples % len(tokens_per_sample) != 0:
        sum_tokens += sum(
            min(entry, max_seq_length, tokenizer_model_max_length)
            for entry in tokens_per_sample[: (samples % len(tokens_per_sample))]
        )

    return sum_tokens


def launch_finetune(
    args: FineTuneArgs,
    aim_metadata: dict[str, typing.Any] | None,
    count_dataset_tokens: bool,
    distributed_settings: DistributedSettings,
    multi_node: MultiNodeSettings | None,
    model_id: str | None = None,
    num_tokens_cache_dir: str | None = None,
    log_level: int | None = None,
) -> "metrics_tracker.Metrics":
    from .callbacks import metrics_tracker

    if log_level is None:
        log_level = 20

    logging.basicConfig(level=log_level)

    if multi_node is None:
        multi_node = MultiNodeSettings(num_machines=1)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="sfttrainer_") as working_directory:
        metrics = _finetune_launch_kernel(
            args=args,
            aim_metadata=aim_metadata,
            distributed_settings=distributed_settings,
            multi_node=multi_node,
            working_directory=working_directory,
        )

    if count_dataset_tokens:
        world_size = max(1, args.number_gpus) * max(1, multi_node.num_machines)

        entries_in_one_step = (
            args.per_device_train_batch_size
            * world_size
            * max(1, args.gradient_accumulation_steps)
        )

        # VV: training_steps does not include warmup_steps
        warmup_steps = metrics.warmup_steps or 0
        total_steps = warmup_steps + metrics.training_steps

        num_entries = entries_in_one_step * total_steps

        # VV: We want to get rid of the first X entries which correspond to optimization steps
        # that took place during warmup
        skip_entries = entries_in_one_step * warmup_steps

        ds = tokenize_text(
            path_model=args.model_name_or_path,
            path_data=args.training_data_path,
            max_seq_length=args.max_seq_length,
            num_tokens_cache_dir=num_tokens_cache_dir,
            model_id=model_id,
            num_entries=num_entries,
            skip_entries=skip_entries,
            dataset_text_field=args.dataset_text_field or "output",
        )

        metrics.training.dataset_tokens = metrics_tracker.AggregatedValues(
            avg=ds, min=ds, max=ds
        )

    return metrics


def generate_arguments_sft_trainer(args: FineTuneArgs) -> dict[str, typing.Any]:
    """Generate arguments for tuning.sft_trainer.train_wrapper() from FineTuneArgs plus some HardcodedArgs"""
    excluded = {
        "gradient_checkpointing_use_reentrant",
        "number_gpus",
        "hf_home",
        "dataset_kwargs_skip_prepare_dataset",
    }
    cmdline_args = {
        k: v
        for k, v in dataclasses.asdict(args).items()
        if k not in excluded and v is not None
    }

    # VV: Some arguments are actually nested inside dictionaries

    if args.dataset_kwargs_skip_prepare_dataset is not None:
        cmdline_args["dataset_kwargs"] = {
            "skip_prepare_dataset": args.dataset_kwargs_skip_prepare_dataset
        }

    if args.gradient_checkpointing_use_reentrant is not None:
        cmdline_args["gradient_checkpointing_kwargs"] = {
            "use_reentrant": args.gradient_checkpointing_use_reentrant
        }

    if args.model_name_or_path == "hf-tiny-model-private/tiny-random-BloomForCausalLM":
        # VV: Flash attention is not supported for bloom models
        cmdline_args["use_flash_attn"] = False

    hard_coded = HardcodedArgs()
    accumulated_args = dataclasses.asdict(hard_coded)
    accumulated_args.update(cmdline_args)

    if args.peft_method == "pt":
        lora_or_pt = {"peft_method": "pt"}
    elif args.peft_method == "lora":
        lora_or_pt = {
            "r": args.r,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
            "lora_dropout": args.lora_dropout,
            "peft_method": "lora",
        }
    elif args.peft_method is None:
        lora_or_pt = {"peft_method": "none"}
    else:
        raise ValueError(f'Unknown peft_method="{args.peft_method}"')

    accumulated_args.update(lora_or_pt)

    return {k: v for k, v in accumulated_args.items() if v is not None}
