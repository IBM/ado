# Copyright The IBM Tuning Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import datetime
import os
import sys
import typing

# Standard
from typing import Any

import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.tuning_versions as tuning_versions
import aim

# Third Party
from aim.hugging_face import AimCallback
from transformers import TrainerControl, TrainerState, TrainingArguments


def has_gathered_enough_samples(
    duration_of_optimization_steps: list[float],
    warmup_seconds: float,
    meaningful_samples_seconds: float,
    meaningful_samples_amount: int,
) -> tuple[bool, int]:
    """Determines if the current process should stop based on accumulated steps and time.

    Args:
        duration_of_optimization_steps:
            A list of durations of each optimization step.
        warmup_seconds:
            The duration of the warmup period.
        meaningful_samples_seconds:
            The minimum duration required for meaningful samples.
        meaningful_samples_amount:
            The minimum number of meaningful samples required.

    Returns:
        A tuple containing a boolean indicating if the process should stop and the index of the first step after the warmup period.
        - The boolean value is True if the process should stop, False otherwise.
        - The index indicates the position of the first step after the warmup period in the original 'steps' list. -1 if the stopping criteria are not met.
    """
    if not duration_of_optimization_steps:
        return False, -1

    import numpy as np

    cs = np.cumsum(duration_of_optimization_steps)
    running_for = cs[-1]

    if running_for <= warmup_seconds + meaningful_samples_seconds:
        return False, -1

    first_after_warmup = int(np.searchsorted(cs, warmup_seconds, side="right")) + 1

    if first_after_warmup >= len(duration_of_optimization_steps):
        # VV: we need at least 1 more step
        return False, -1

    steps_after_warmup = duration_of_optimization_steps[first_after_warmup:]
    duration_of_meaningful_samples = cs[-1] - cs[first_after_warmup - 1]

    # Stop when there are at least @meaningful_samples_amount samples post AND
    # we've spent at least @meaningful_samples_seconds seconds drawing these meaningful samples
    return (
        (
            (len(steps_after_warmup) >= meaningful_samples_amount)
            and duration_of_meaningful_samples >= meaningful_samples_seconds
        ),
        first_after_warmup,
    )


def get_cuda_uuid_to_index() -> dict[str, int]:
    """Returns a dictionary mapping GPU device UUIDs to their index numbers"""
    try:
        import aim.ext.pynvml as nvml

        nvml.nvmlInit()
    except Exception as e:
        print(
            f"Unable to initialize nvml when mapping cuda uuid to AIM gpu indices due to {e} - "
            f"will skip mapping the uuids"
        )
        return {}

    gpu_device_count = nvml.nvmlDeviceGetCount()

    ret = {
        str(nvml.nvmlDeviceGetUUID(nvml.nvmlDeviceGetHandleByIndex(i))): i
        for i in range(gpu_device_count)
    }

    nvml.nvmlShutdown()

    return ret


def get_cuda_device_indices(cuda_visible_devices: str) -> list[int]:
    """Returns the indices of cuda devices

    Args:
        cuda_visible_devices: The value of the CUDA_VISIBLE_DEVICES environment
        variable. It represents the devices that should be made visible to the
        current process.

    Returns:
        a list of integers representing the device indices that should be made visible to
        the current process.
    """
    if not cuda_visible_devices:
        return []

    try:
        return [int(x) for x in cuda_visible_devices.split(",") if len(x) > 0]
    except ValueError:
        # VV: these are cuda device UIDs, need to decode them
        pass

    cuda_mapping = get_cuda_uuid_to_index()
    return [cuda_mapping.get(uuid, uuid) for uuid in cuda_visible_devices.split(",")]


def calculate_gpu_power_percent(
    run_metrics: list[tuple[str, dict[str, int], list[float]]],
):
    """Calculates __system__gpu_power_percent using __system__gpu_power_watts and inserts it into existing run metrics

    Args:
        run_metrics:
            The run metrics collected from AIM. The method updates this array in memory

    Returns:
        Nothing
    """
    from aim.ext.resource.utils import round10e5

    try:
        import aim.ext.pynvml as nvml

        nvml.nvmlInit()
    except Exception as e:
        print(
            f"Unable to instantiate nvml due to {e} - will not record power measurements",
            file=sys.stderr,
        )
        return []

    for name, context, values in run_metrics:
        # VV: aim reports: gpu_info['gpu_power_watts'] = round10e5(nvml.nvmlDeviceGetPowerUsage(handle) / 1000)
        if name == "__system__gpu_power_watts" and "gpu" in context:
            handle = nvml.nvmlDeviceGetHandleByIndex(context["gpu"])
            # VV: nvmlDeviceGetEnforcedPowerLimit is in Milliwatts:
            # https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html

            power_cap = nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000

            # VV: The range is [0, 100]
            gpu_percent = [round10e5(v * 100.0 / power_cap) for v in values]

            run_metrics.append(
                (
                    "__system__gpu_power_percent",
                    context.copy(),
                    gpu_percent,
                )
            )

    nvml.nvmlShutdown()

    return run_metrics


class CustomAimCallback(AimCallback):
    # VV: Set this after training starts and never delete it
    the_run_hash = None
    the_experiment: "aim.Run | None" = None
    training_steps = 0

    def __init__(
        self,
        repo: str | None = None,
        experiment: str | None = None,
        system_tracking_interval: int | None = 10,
        log_system_params: bool | None = True,
        capture_terminal_logs: bool | None = True,
        additional_metrics: dict[str, Any] | None = None,
        aim_info_path: str | None = None,
        aim_info_aggregate_metrics: bool = False,
        aim_metadata: dict[str, Any] | None = None,
        stop_after_seconds: float = -1.0,
        auto_stop_method: int | None = None,
    ):

        self._additional_metrics = additional_metrics or {}
        self._aim_info_path = aim_info_path
        self._aim_info_aggregate_metrics = aim_info_aggregate_metrics
        self._aim_metadata = aim_metadata or {}

        self._stop_after_seconds = stop_after_seconds
        self._time_started: datetime.datetime | None = None

        self._optimization_step_started: datetime.datetime | None = None

        # VV: When auto_stop_method is not None, we will also maintain a list of the durations of each
        # optimization step. At the end of each optimization step we'll invoke the auto stop method.
        # If the auto stop method signals that training should stop we will also drop the system metrics we extracted
        # during the warmup phase.
        self._post_warmup_optimization_step_index: int | None = None
        self._optimization_step_durations = []
        self._auto_stop_method = auto_stop_method

        super().__init__(
            repo,
            experiment,
            system_tracking_interval,
            log_system_params,
            capture_terminal_logs,
        )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model, **kwargs)

        self._time_started = datetime.datetime.now()

        if state.is_local_process_zero:
            run: aim.Run = self.experiment
            CustomAimCallback.the_experiment = run
            CustomAimCallback.the_run_hash = run.hash

            for k, v in self._additional_metrics.items():
                run.track(v, name=k, context={"scope": "additional_metrics"})

            for k, v in self._aim_metadata.items():
                run[k] = v

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            self._optimization_step_started = datetime.datetime.now()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_step_end(args, state, control, **kwargs)

        CustomAimCallback.training_steps += 1

        sys.stderr.flush()
        sys.stdout.flush()

        if state.is_world_process_zero:
            dt = (
                datetime.datetime.now() - self._optimization_step_started
            ).total_seconds()

            self.experiment.track(value=dt, name="optimization_step_duration")

        if state.is_local_process_zero and self._auto_stop_method is not None:
            # VV: In multi-node runs the local process zero dumps a JSON file with information we collected.
            # When auto_stop_method is on, all local zero processes smust get rid of the system metrics of the warmup
            # phase. This means that all local zero processes must keep track of the durations for optimization steps
            # so that they can compute the duration of the warmup phase.
            self._optimization_step_durations.append(dt)

        running_for = (datetime.datetime.now() - self._time_started).total_seconds()

        if (
            state.is_local_process_zero
            and 0.0 <= self._stop_after_seconds <= running_for
        ):
            print(
                "Triggering experiment to stop after running for",
                running_for,
                f"seconds due to stop_after_seconds={self._stop_after_seconds}",
            )
            control.should_training_stop = True

        if self._auto_stop_method is not None:
            if self._auto_stop_method == 1:
                gathered_enough, post_warmup_step_idx = has_gathered_enough_samples(
                    duration_of_optimization_steps=self._optimization_step_durations,
                    warmup_seconds=60,
                    meaningful_samples_seconds=120,
                    meaningful_samples_amount=10,
                )
            else:
                raise NotImplementedError(
                    "Unknown auto_stop_method", self._auto_stop_method
                )

            if gathered_enough:
                self._post_warmup_optimization_step_index = post_warmup_step_idx
                if state.is_local_process_zero:
                    print(
                        "Triggering experiment to stop after running for",
                        running_for,
                        f"seconds due to auto_stop_method={self._auto_stop_method}.",
                        "Post warmup step index is",
                        post_warmup_step_idx,
                    )
                    control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        try:
            if self._aim_info_path and state.is_local_process_zero:
                format_time = "%d%m%y-%H%M%S"
                train_time_stop = datetime.datetime.now().strftime(format_time)

                run: aim.Run = self.experiment

                for k, v in self._additional_metrics.items():
                    run.track(v, name=k, context={"scope": "additional_metrics"})

                cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

                cuda_visible_devices = get_cuda_device_indices(cuda_visible_devices)

                metrics = []

                run_metrics = [
                    (m.name, m.context.to_dict(), m.values.values_list())
                    for m in run.metrics()
                ]

                run_metrics.extend(calculate_gpu_power_percent(run_metrics=run_metrics))

                skip_first_system_metrics, warmup_seconds = None, None

                if self._post_warmup_optimization_step_index:
                    warmup_seconds = sum(
                        self._optimization_step_durations[
                            : self._post_warmup_optimization_step_index
                        ]
                    )

                    skip_first_system_metrics = int(
                        warmup_seconds / self._system_tracking_interval
                    )

                for name, context, values in run_metrics:
                    if self._aim_info_aggregate_metrics:
                        try:
                            values = list(values)

                            if (
                                name.startswith("__system")
                                and skip_first_system_metrics is not None
                            ):
                                values = values[skip_first_system_metrics:]

                            values = [x for x in values if x is not None]
                            len_values = len(values)

                            if len_values > 0:
                                _sum = sum(values)
                                avg = _sum / len_values
                                _max = max(values)
                                _min = min(values)
                            else:
                                avg, _max, _min = None, None, None

                            values = {
                                "avg": avg,
                                "max": _max,
                                "min": _min,
                            }
                        except ValueError:
                            # Don't aggregate properties that are weird
                            pass

                    metrics.append(
                        {
                            "name": name,
                            "values": values,
                            "context": context,
                        }
                    )

                # Standard
                import json

                if self._time_started is not None:
                    train_time_start = self._time_started.strftime(format_time)
                else:
                    train_time_start = None

                with open(self._aim_info_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "run_hash": run.hash,
                            "metrics": metrics,
                            "hostname": os.environ.get("HOSTNAME"),
                            "train_time_start": train_time_start,
                            "train_time_stop": train_time_stop,
                            "cuda_visible_devices": cuda_visible_devices,
                            "world_rank": os.environ.get("RANK", "0"),
                            "world_size": os.environ.get("WORLD_SIZE", "1"),
                            "training_steps": CustomAimCallback.training_steps,
                            "post_warmup_index_for_stable_properties": self._post_warmup_optimization_step_index,
                            "warmup_seconds": warmup_seconds,
                        },
                        f,
                    )
        finally:
            super().on_train_end(args=args, state=state, control=control, **kwargs)
            if CustomAimCallback.the_experiment:
                CustomAimCallback.the_experiment = None


@dataclasses.dataclass
class CustomArgs:
    aim_metadata_path: str | None = dataclasses.field(
        default=None,
        metadata={
            "help": "Path to JSON file containing metadata that sft_trainer.py will store in AIM"
        },
    )

    aim_info_path: str | None = dataclasses.field(
        default=None,
        metadata={
            "help": "The path to a JSON file that sft_trainer.py will use to store the metrics that AIM captures. "
            "If unset, the script will not produce the file"
        },
    )

    aim_info_aggregate_metrics: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "Whether to store the mean values of the metrics that AIM measures in the aim_info_path file"
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

    stop_after_seconds: float = dataclasses.field(
        default=-1.0,
        metadata={
            "help": "If set, the optimizer will be asked to stop after the specified time elapses. "
            "The check is performed after the end of each training step."
        },
    )

    auto_stop_method: int | None = dataclasses.field(
        default=None,
        metadata={
            "help": "The method to use for automatically stopping the finetuning job. "
            "1: Stops after the job has performed 60+max(120 seconds, duration of 10 optimization steps). "
            "This method will ignore the first 60 seconds of the training for both the throughput and the "
            "recorded system metrics."
        },
    )


def main():
    """Utility method that invokes the main() method of sft_trainer.py, catches GPU OOM exceptions and logs them
    to the --aim_info_path JSON file as well as STDERR"""
    import json
    import sys

    import transformers

    parser = transformers.HfArgumentParser(dataclass_types=(CustomArgs,))

    (
        custom_args,
        remaining_args,
    ) = parser.parse_args_into_dataclasses(
        args=list(sys.argv[1:]),
        return_remaining_strings=True,
    )

    sys.argv = [sys.argv[0], *remaining_args]

    custom_args = typing.cast(CustomArgs, custom_args)

    if custom_args.fms_hf_tuning_version is None:
        raise ValueError("must set --fms_hf_tuning_version")

    if custom_args.aim_metadata_path:
        with open(custom_args.aim_metadata_path) as f:
            aim_metadata = json.load(f)
    else:
        aim_metadata = {}

    import json

    import torch.cuda
    import tuning.sft_trainer

    metadata = aim_metadata.get("metadata", {})

    try:
        measurement_id = "/".join((metadata["experiment"], metadata["entity"]))
    except KeyError as e:
        print("Could not construct measurement id due to", e, file=sys.stderr)
        measurement_id = "unknown/unknown"

    def report_error(exception: Exception, warning: str, exception_type: str):
        print(warning, file=sys.stderr)
        # Standard
        import traceback

        print(traceback.format_exc(), file=sys.stderr)
        import os

        # VV: 'accelerate' injects this env-var
        if os.environ.get("RANK", ""):
            rank = os.environ["RANK"]
            path = "_".join((custom_args.aim_info_path, rank))
        else:
            path = custom_args.aim_info_path
            rank = "?"

        report = {
            "error": exception_type,
            "exception": str(exception),
            "run_hash": CustomAimCallback.the_run_hash,
            "training_steps": CustomAimCallback.training_steps,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                report,
                f,
            )

        print(
            f"Worker {rank} of {measurement_id} dumped {report} in {path}",
            file=sys.stderr,
        )

    def report_oom(exception: Exception):
        return report_error(
            exception,
            warning="SFTTRAINER_EXCEPTION: OUT_OF_MEMORY",
            exception_type="OutOfGPUMemoryError",
        )

    def report_nccl_error(exception: Exception):
        return report_error(
            exception,
            warning="SFTTRAINER_EXCEPTION: NCCL_ERROR",
            exception_type="NCCLError",
        )

    try:
        print("Worker started", file=sys.stderr)

        if not custom_args.aim_info_path:
            raise ValueError("must set --aim_info_path")

        job_config = tuning.sft_trainer.get_json_config()

        callbacks = [
            CustomAimCallback(
                repo=custom_args.aim_db,
                experiment=custom_args.aim_experiment,
                additional_metrics={},
                aim_info_path=custom_args.aim_info_path,
                aim_info_aggregate_metrics=custom_args.aim_info_aggregate_metrics,
                aim_metadata=aim_metadata,
                stop_after_seconds=custom_args.stop_after_seconds,
                auto_stop_method=custom_args.auto_stop_method,
            )
        ]
        module = tuning_versions.import_tuning_version(
            version=custom_args.fms_hf_tuning_version
        )
        module.parse_arguments_and_execute_wrapper(
            callbacks=callbacks, job_config=job_config
        )

    except torch.cuda.OutOfMemoryError as e:
        report_oom(e)
        raise
    except RuntimeError as e:
        if (
            "CUDA error: out of memory".lower() in str(e).lower()
            or "CUDA error: an illegal memory access was encountered".lower()
            in str(e).lower()
        ):
            report_oom(e)
            # elif "NCCL Error".lower() in str(e).lower():
            #     report_nccl_error(e)
        else:
            report_error(
                e,
                warning=f"SFTTRAINER_EXCEPTION: UNHANDLED {type(e)}",
                exception_type=f"Unhandled({type(e)})",
            )
        raise
    except BaseException as e:
        report_error(
            e,
            warning=f"SFTTRAINER_EXCEPTION: UNHANDLED {type(e)}",
            exception_type=f"Unhandled({type(e)})",
        )
        raise
    finally:
        print("Worker stopped", file=sys.stderr)
        if CustomAimCallback.the_experiment:
            print(
                "AIM Run is not closed, will close it now. Measurement id:",
                measurement_id,
                "AIM run hash",
                CustomAimCallback.the_experiment.hash,
                file=sys.stderr,
            )
            CustomAimCallback.the_experiment.close()


if __name__ == "__main__":
    main()
