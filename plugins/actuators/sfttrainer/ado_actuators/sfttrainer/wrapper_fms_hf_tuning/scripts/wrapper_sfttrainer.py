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
import datetime as dt
import json
import os
import sys
import typing
from typing import Any

import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.callbacks.metrics_tracker as metrics_tracker
import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.constants as constants
import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.tuning_versions as tuning_versions
import torch
import torch.distributed
from transformers import TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import TrainerCallback


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
        import pynvml

        pynvml.nvmlInit()
    except Exception as e:
        print(
            f"Unable to initialize pynvml when mapping cuda uuid to gpu indices due to {e} - "
            f"will skip mapping the uuids"
        )
        return {}

    gpu_device_count = pynvml.nvmlDeviceGetCount()

    ret = {
        str(pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))): i
        for i in range(gpu_device_count)
    }

    pynvml.nvmlShutdown()

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


class CustomCallBack(TrainerCallback):
    training_steps = 0

    def __init__(
        self,
        system_tracking_interval: float = 10,
        aggregate_metrics: bool = True,
        output_info_path: str | None = None,
        stop_after_seconds: float = -1.0,
        auto_stop_method: constants.AutoStopMethod | None = None,
    ) -> None:

        self._output_info_path = output_info_path

        self._stop_after_seconds = stop_after_seconds
        self._time_started: dt.datetime | None = None

        self._optimization_step_started: dt.datetime | None = None

        # VV: When auto_stop_method = WARMUP_60S_STABLE_120S_OR_10_STEPS, this ivar will (eventually) be the
        # index of the first optimization step that follows the WARMUP phase.
        self._warmup_steps: int | None = None
        self._optimization_step_durations = []
        self._auto_stop_method = auto_stop_method
        self._system_tracking_interval = system_tracking_interval
        self._aggregate_metrics = aggregate_metrics
        self._logged_metrics: dict[tuple[str, str | None], list[float | int]] = {}

        self.resource_tracker = metrics_tracker.ResourceTracker(
            period=self._system_tracking_interval
        )

        self.resource_tracker.begin_track()

        super().__init__()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().on_log(args=args, state=state, control=control, logs=logs, **kwargs)

        if not state.is_world_process_zero or not logs:
            return

        prefix_set = ("train_", "eval_", "test_")
        for log_name, log_value in logs.items():
            if not isinstance(log_value, (int, float)):
                continue

            subset = None
            for prefix in prefix_set:
                if log_name.startswith(prefix):
                    log_name = log_name[len(prefix) :]
                    subset = prefix[:-1]
                    break

            key = (log_name, subset)
            if key not in self._logged_metrics:
                self._logged_metrics[key] = []
            self._logged_metrics[key].append(log_value)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().on_train_begin(args=args, state=state, control=control, **kwargs)

        self._time_started = dt.datetime.now()

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().on_step_begin(args=args, state=state, control=control, **kwargs)
        self._optimization_step_started = dt.datetime.now()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().on_step_end(args=args, state=state, control=control, **kwargs)

        CustomCallBack.training_steps += 1

        now = dt.datetime.now()
        elapsed = (now - self._time_started).total_seconds()
        step_duration = (now - self._optimization_step_started).total_seconds()

        if state.is_local_process_zero and 0.0 <= self._stop_after_seconds <= elapsed:
            print(
                "Triggering experiment to stop after running for",
                elapsed,
                f"seconds due to stop_after_seconds={self._stop_after_seconds}",
            )
            control.should_training_stop = True
            self.resource_tracker.end_track()

        if state.is_local_process_zero:
            if (
                self._auto_stop_method
                == constants.AutoStopMethod.WARMUP_60S_STABLE_120S_OR_10_STEPS
            ):
                # VV: In multi-node runs, each local-rank 0 process dumps a JSON file with metrics it collects.
                # The auto_stop_method ignores system metrics gathered during the warmup phase. To achieve this,
                # each local-rank 0 process must know the warmup duration. Instead of synchronizing across workers,
                # we just have all local-rank 0 processes compute it independently.
                # To this end, local-rank 0 processes track the duration of optimization steps.
                self._optimization_step_durations.append(step_duration)

                # VV: all processes with local rank 0 need to compute their respective warmup phase in order to
                # be able to discard the system metrics that they collect during warmup
                gathered_enough, post_warmup_step_idx = has_gathered_enough_samples(
                    duration_of_optimization_steps=self._optimization_step_durations,
                    warmup_seconds=60,
                    meaningful_samples_seconds=120,
                    meaningful_samples_amount=10,
                )

                if gathered_enough:
                    self._warmup_steps = post_warmup_step_idx
                    print(
                        "Triggering experiment to stop after running for",
                        elapsed,
                        f"seconds due to auto_stop_method={self._auto_stop_method}.",
                        "Warmup steps were",
                        self._warmup_steps,
                    )
                    control.should_training_stop = True
            elif self._auto_stop_method is not None:
                raise NotImplementedError(
                    "Unknown auto_stop_method", self._auto_stop_method
                )

        sys.stderr.flush()
        sys.stdout.flush()

        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            # VV: Stop all workers if all local-rank 0 workers agree to stop
            should_stop = torch.tensor(
                [
                    # VV: We use MIN reduction to stop training when all local-rank 0 workers agree.
                    # Non local-rank-0 workers also participate in this reduction, but their value is
                    # irrelevant - and it is always False.
                    # Here, we set these values to True so that they don't influence the reduce operation at all.
                    (not state.is_local_process_zero) or control.should_training_stop
                ],
                dtype=torch.uint8,
            ).cuda()

            torch.distributed.all_reduce(should_stop, op=torch.distributed.ReduceOp.MIN)
            control.should_training_stop = should_stop.item() == 1

            if control.should_training_stop:
                print("All local-rank 0 workers agree to stop early")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        try:
            if self._output_info_path and state.is_local_process_zero:
                format_time = "%d%m%y-%H%M%S"
                train_time_stop = dt.datetime.now().strftime(format_time)

                cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

                cuda_visible_devices = get_cuda_device_indices(cuda_visible_devices)

                metrics = []
                _skip_first_system_metrics, warmup_seconds = None, None
                keep_after_time: dt.datetime | None = None

                if self._warmup_steps:
                    warmup_seconds = sum(
                        self._optimization_step_durations[: self._warmup_steps]
                    )

                    keep_after_time = self._time_started + dt.timedelta(
                        seconds=warmup_seconds
                    )

                run_metrics = [
                    (m.name, m.context, m.values)
                    for m in self.resource_tracker.to_resource_tracker_metrics(
                        keep_after_time=keep_after_time
                    )
                ]

                for (name, subset), values in self._logged_metrics.items():
                    # VV: This is copying in the train_ metrics which we need to
                    # extrapolate the training duration of the measurement
                    context = {"subset": subset} if subset is not None else {}
                    run_metrics.append((name, context, values))

                for name, context, values in run_metrics:
                    if self._aggregate_metrics:
                        try:
                            values = list(values)

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

                if self._time_started is not None:
                    train_time_start = self._time_started.strftime(format_time)
                else:
                    train_time_start = None

                with open(self._output_info_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "metrics": metrics,
                            "hostname": os.environ.get("HOSTNAME"),
                            "train_time_start": train_time_start,
                            "train_time_stop": train_time_stop,
                            "cuda_visible_devices": cuda_visible_devices,
                            "world_rank": os.environ.get("RANK", "0"),
                            "world_size": os.environ.get("WORLD_SIZE", "1"),
                            "training_steps": CustomCallBack.training_steps,
                            "warmup_steps": self._warmup_steps,
                            "warmup_seconds": warmup_seconds,
                        },
                        f,
                    )
        finally:
            super().on_train_end(args=args, state=state, control=control, **kwargs)
            self.resource_tracker.end_track()


@dataclasses.dataclass
class CustomArgs:
    metrics_info_path: str | None = dataclasses.field(
        default=None,
        metadata={
            "help": "The path to a JSON file that sft_trainer.py will use to store the metrics that the resource tracker and trainer callbacks capture. "
            "If unset, the script will not produce the file"
        },
    )

    metrics_info_aggregate_metrics: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "Whether to store the mean values of the metrics that are captured in the metrics_info_path file"
        },
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

    auto_stop_method: constants.AutoStopMethod | None = dataclasses.field(
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


def main() -> None:
    """Utility method that invokes the main() method of sft_trainer.py, catches GPU OOM exceptions and logs them
    to the --metrics_info_path JSON file as well as STDERR"""
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

    custom_args = typing.cast("CustomArgs", custom_args)

    if custom_args.fms_hf_tuning_version is None:
        raise ValueError("must set --fms_hf_tuning_version")

    import torch.cuda
    import tuning.sft_trainer

    def report_error(exception: Exception, warning: str, exception_type: str) -> None:
        print(warning, file=sys.stderr)
        # Standard
        import traceback

        print(traceback.format_exc(), file=sys.stderr)
        import os

        # VV: 'accelerate' injects this env-var
        if os.environ.get("RANK", ""):
            rank = os.environ["RANK"]
            path = f"{custom_args.metrics_info_path}_{rank}"
        else:
            path = custom_args.metrics_info_path
            rank = "?"

        report = {
            "error": exception_type,
            "exception": str(exception),
            "training_steps": CustomCallBack.training_steps,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                report,
                f,
            )

        print(
            f"Worker {rank} dumped {report} in {path}",
            file=sys.stderr,
        )

    def report_oom(exception: Exception) -> None:
        return report_error(
            exception,
            warning="SFTTRAINER_EXCEPTION: OUT_OF_MEMORY",
            exception_type="OutOfGPUMemoryError",
        )

    def report_nccl_error(exception: Exception) -> None:
        return report_error(
            exception,
            warning="SFTTRAINER_EXCEPTION: NCCL_ERROR",
            exception_type="NCCLError",
        )

    callbacks: list[CustomCallBack] = []
    try:
        print("Worker started", file=sys.stderr)

        if not custom_args.metrics_info_path:
            raise ValueError("must set --metrics_info_path")

        job_config = tuning.sft_trainer.get_json_config()

        callbacks = [
            CustomCallBack(
                output_info_path=custom_args.metrics_info_path,
                aggregate_metrics=custom_args.metrics_info_aggregate_metrics,
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
        if callbacks:
            print("Stopping the resource tracker", file=sys.stderr)
            rt = callbacks[0].resource_tracker
            rt.end_track()
            total_requests = rt.metrics
            print(
                f"Resource tracker stopped after collecting {len(total_requests)} total measurements",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
