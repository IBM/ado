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
from typing import Any, Optional

import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.tuning_versions as tuning_versions
import aim

# Third Party
from aim.hugging_face import AimCallback
from transformers import TrainerControl, TrainerState, TrainingArguments


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
    the_experiment: "typing.Optional[aim.Run]" = None
    training_steps = 0

    def __init__(
        self,
        repo: Optional[str] = None,
        experiment: Optional[str] = None,
        system_tracking_interval: Optional[int] = 10,
        log_system_params: Optional[bool] = True,
        capture_terminal_logs: Optional[bool] = True,
        additional_metrics: Optional[dict[str, Any]] = None,
        aim_info_path: Optional[str] = None,
        aim_info_aggregate_metrics: bool = False,
        aim_metadata: Optional[dict[str, Any]] = None,
        stop_after_seconds: float = -1.0,
    ):

        self._additional_metrics = additional_metrics or {}
        self._aim_info_path = aim_info_path
        self._aim_info_aggregate_metrics = aim_info_aggregate_metrics
        self._aim_metadata = aim_metadata or {}

        self._stop_after_seconds = stop_after_seconds
        self._time_started: typing.Optional[datetime.datetime] = None

        self._optimization_step_started: typing.Optional[datetime.datetime] = None

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

        if self._stop_after_seconds < 0.0:
            return

        running_for = (datetime.datetime.now() - self._time_started).total_seconds()

        if running_for >= self._stop_after_seconds:
            print(
                "Triggering experiment to stop after running for",
                running_for,
                f"seconds due to stop_after_seconds={self._stop_after_seconds}",
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

                for name, context, values in run_metrics:
                    if self._aim_info_aggregate_metrics:
                        try:
                            len_values = 0
                            _sum = 0
                            avg = None
                            _min = None
                            _max = None

                            for x in values:
                                if x is None:
                                    continue
                                len_values += 1
                                _sum += x

                                if _min is None or _min > x:
                                    _min = x
                                if _max is None or _max < x:
                                    _max = x

                            if len_values > 0:
                                avg = _sum / len_values

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
                        },
                        f,
                    )
        finally:
            super().on_train_end(args=args, state=state, control=control, **kwargs)
            if CustomAimCallback.the_experiment:
                CustomAimCallback.the_experiment = None


@dataclasses.dataclass
class CustomArgs:
    aim_metadata_path: typing.Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "Path to JSON file containing metadata that sft_trainer.py will store in AIM"
        },
    )

    aim_info_path: typing.Optional[str] = dataclasses.field(
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

    aim_db: typing.Optional[str] = dataclasses.field(
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
        with open(custom_args.aim_metadata_path, "r") as f:
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
