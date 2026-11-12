# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import dataclasses
import datetime as dt
import logging
import os
import typing
from threading import Event, Thread

if typing.TYPE_CHECKING:
    import psutil

log = logging.getLogger(__name__)


def was_gpu_in_use(
    gpu_cuda_id: int, cuda_visible_devices: list[int], num_gpus: int
) -> bool:
    # VV: We used that GPU if cuda_visible_devices is empty BUT we did ask
    # FSDP to use at least 1 GPU. OR the gpu cuda id is in the cuda_visible_devices array (i.e. pytorch knew
    # exactly which of the GPUs available on the system it was meant to use)
    return (not cuda_visible_devices and num_gpus > 0) or (
        gpu_cuda_id in cuda_visible_devices
    )


@dataclasses.dataclass
class ResourceTrackerMetric:
    name: str
    context: dict[str, typing.Any]
    values: list[float | int]


@dataclasses.dataclass
class AggregatedValues:
    avg: float | None = dataclasses.field(default=-1.0)
    min: float | None = dataclasses.field(default=-1.0)
    max: float | None = dataclasses.field(default=-1.0)


def aggregate_values(
    values: list[float] | dict[str, float | None],
) -> float | AggregatedValues:
    if isinstance(values, list):
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

    return AggregatedValues(**values)


@dataclasses.dataclass
class GPUMetrics:
    gpu: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues, metadata={"help": "GPU utilization percent"}
    )
    gpu_memory_percent: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues,
        metadata={"help": "GPU memory utilization percent"},
    )
    gpu_power_watts: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues,
        metadata={"help": "GPU power consumption in Watts."},
    )
    gpu_power_percent: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues,
        metadata={"help": "GPU power consumption (percentage)"},
    )

    @classmethod
    def from_metrics_info_dict(cls, info: dict[str, typing.Any]) -> "GPUMetrics":
        return GPUMetrics(
            gpu=aggregate_values(info["__system__gpu"]["values"]),
            gpu_memory_percent=aggregate_values(
                info["__system__gpu_memory_percent"]["values"]
            ),
            gpu_power_watts=aggregate_values(
                info["__system__gpu_power_watts"]["values"]
            ),
            gpu_power_percent=aggregate_values(
                info["__system__gpu_power_percent"]["values"]
            ),
        )


@dataclasses.dataclass
class SystemMetrics:
    cpu: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues,
        metadata={"help": "CPU utilization percent (can be over 1.0)"},
    )
    memory_percent: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues, metadata={"help": "Whole system memory usage"}
    )

    @classmethod
    def from_metrics_info_dict(cls, info: dict[str, typing.Any]) -> "SystemMetrics":
        return SystemMetrics(
            cpu=aggregate_values(info["__system__cpu"]["values"]),
            memory_percent=aggregate_values(info["__system__memory_percent"]["values"]),
        )


@dataclasses.dataclass
class TrainingMetrics:
    train_runtime: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues
    )
    train_samples_per_second: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues
    )
    train_steps_per_second: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues
    )
    train_tokens_per_second: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues
    )
    dataset_tokens: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues
    )

    @classmethod
    def from_metrics_info_list(
        cls, training_metrics: dict[str, dict[str, typing.Any]]
    ) -> "TrainingMetrics":
        if "runtime" not in training_metrics:
            return TrainingMetrics()

        return TrainingMetrics(
            train_runtime=aggregate_values(training_metrics["runtime"]["values"]),
            train_samples_per_second=aggregate_values(
                training_metrics["samples_per_second"]["values"]
            ),
            train_steps_per_second=aggregate_values(
                training_metrics["steps_per_second"]["values"]
            ),
            train_tokens_per_second=aggregate_values(
                training_metrics["tokens_per_second"]["values"]
            ),
            dataset_tokens=aggregate_values(
                training_metrics.get("dataset_tokens", {}).get("values", [])
            ),
        )


@dataclasses.dataclass
class ModelMetrics:
    model_load_time: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues
    )


@dataclasses.dataclass
class Metrics:
    gpus: list[GPUMetrics]
    system: SystemMetrics
    training: TrainingMetrics
    model: ModelMetrics
    training_steps: int = dataclasses.field(
        metadata={"help": "The number of training steps excluding the warmup steps"},
    )
    train_time_start: dt.datetime | None = None
    train_time_stop: dt.datetime | None = None
    hostname_gpus: dict[str, list[int]] = dataclasses.field(
        default_factory=dict,
        metadata={
            "help": "Key=value pairs where the key is a hostname and the value an "
            "array of GPU indices that the host used"
        },
    )
    warmup_steps: int | None = None
    warmup_seconds: float | None = None

    def to_scalar_observations(
        self,
        distributed_backend: typing.Literal["FSDP", "DDP"] | None,
        world_size: int,
    ) -> dict[str, float]:
        scalar_observations = {}
        import dataclasses

        def aggregate_metrics(
            vals: list[float],
        ) -> tuple[float, float, float]:
            """Utility method that returns min, avg, max of a list of floats
            If the array is empty then the method assumes that it's equal to [0.0]
            """
            vals = vals or [0.0]
            return min(vals), sum(vals) / len(vals), max(vals)

        scalar_observations.update(
            {
                "cpu_compute_utilization": self.system.cpu.avg,
                "cpu_memory_utilization": self.system.memory_percent.avg,
            }
        )

        for label, values in (
            ("gpu_compute_utilization", [gpu.gpu.avg for gpu in self.gpus]),
            (
                "gpu_memory_utilization",
                [gpu.gpu_memory_percent.avg for gpu in self.gpus],
            ),
            ("gpu_power_watts", [gpu.gpu_power_watts.avg for gpu in self.gpus]),
            ("gpu_power_percent", [gpu.gpu_power_percent.avg for gpu in self.gpus]),
        ):
            v_min, v_avg, v_max = aggregate_metrics(values)
            scalar_observations[f"{label}_min"] = v_min
            scalar_observations[f"{label}_avg"] = v_avg
            scalar_observations[f"{label}_max"] = v_max

        if self.gpus:
            scalar_observations["gpu_memory_utilization_peak"] = max(
                [gpu.gpu_memory_percent.max for gpu in self.gpus],
            )

        scalar_observations.update(
            {
                k: v["avg"]
                for k, v in dataclasses.asdict(self.training).items()
                if k != "dataset_tokens"
            }
        )

        if world_size > 1 and distributed_backend in ["FSDP", "DDP"]:
            # VV: FSDP and DDP report the train_tokens_per_second per device
            scalar_observations["train_tokens_per_second"] *= world_size

        if self.training.dataset_tokens.avg is not None:
            scalar_observations["dataset_tokens_per_second"] = (
                self.training.dataset_tokens.avg / self.training.train_runtime.avg
            )
            scalar_observations["dataset_tokens_per_second_per_gpu"] = (
                scalar_observations["dataset_tokens_per_second"] / world_size
            )
        else:
            scalar_observations["dataset_tokens_per_second"] = -1.0
            scalar_observations["dataset_tokens_per_second_per_gpu"] = -1.0

        scalar_observations["train_tokens_per_gpu_per_second"] = (
            scalar_observations["train_tokens_per_second"] / world_size
        )
        scalar_observations["model_load_time"] = self.model.model_load_time.avg

        if self.hostname_gpus:
            scalar_observations["hostname_gpus"] = self.hostname_gpus

        if self.train_time_stop is not None:
            scalar_observations["train_time_stop"] = self.train_time_stop

        if self.train_time_start is not None:
            # VV: This includes the warmup phase
            scalar_observations["train_time_start"] = self.train_time_start

        scalar_observations["training_steps"] = self.training_steps

        return scalar_observations

    @classmethod
    def from_metrics_info_dict(
        cls, metrics_info: dict[str, typing.Any], num_gpus: int
    ) -> "Metrics":
        json_metrics = metrics_info["metrics"]

        training_metrics = {
            x["name"]: x
            for x in json_metrics
            if x["name"]
            in (
                "samples_per_second",
                "steps_per_second",
                "tokens_per_second",
                "runtime",
                "dataset_tokens",
            )
        }

        training_metrics = TrainingMetrics.from_metrics_info_list(training_metrics)

        gpus = {}

        for m in json_metrics:
            if m["name"].startswith("__system__gpu"):
                gpu_cuda_id = m["context"]["gpu"]

                if gpu_cuda_id not in gpus:
                    gpus[gpu_cuda_id] = {
                        # VV: init all metrics just in case the tracker couldn't extract any of them
                        x: {"values": [], "context": {"gpu": gpu_cuda_id}, "name": x}
                        for x in [
                            "__system__gpu",
                            "__system__gpu_power_watts",
                            "__system__gpu_memory_percent",
                            "__system__gpu_power_percent",
                        ]
                    }

                gpus[gpu_cuda_id][m["name"]] = m

        gpu_metrics = []
        for gpu_cuda_id, gpu_info in gpus.items():
            if was_gpu_in_use(
                gpu_cuda_id, metrics_info["cuda_visible_devices"], num_gpus
            ):
                gpu_metrics.append(GPUMetrics.from_metrics_info_dict(gpu_info))

        system_metrics = {
            # VV: init all metrics just in case the tracker couldn't extract any of them
            x: {"values": [], "context": {}, "name": x}
            for x in [
                "__system__cpu",
                "__system__memory_percent",
            ]
        }

        for m in json_metrics:
            if m["name"] in system_metrics:
                system_metrics[m["name"]] = m

        system_metrics = SystemMetrics.from_metrics_info_dict(system_metrics)

        model_load_time = [-1.0]

        for m in json_metrics:
            if m["name"] == "model_load_time":
                model_load_time = m["values"]

        train_time_start = metrics_info.get("train_time_start")
        train_time_stop = metrics_info.get("train_time_stop")
        format_time = "%d%m%y-%H%M%S"

        if train_time_start:
            train_time_start = dt.datetime.strptime(train_time_start, format_time)
        if train_time_stop:
            train_time_stop = dt.datetime.strptime(train_time_stop, format_time)

        # VV: training_steps does not include the warmup_steps
        training_steps = metrics_info["training_steps"] - (
            metrics_info.get("warmup_steps") or 0
        )

        warmup_seconds = metrics_info.get("warmup_seconds") or 0

        if warmup_seconds > 0:
            if (
                training_metrics.train_runtime.avg is None
                or training_metrics.train_runtime.avg < 0
            ):
                raise ValueError(
                    f"warmup_seconds={warmup_seconds} is set but training_metrics.train_runtime "
                    f"was not measured ({training_metrics.train_runtime})"
                )
            # VV: Just like training_steps, train_runtime does not include the warmup_seconds
            training_metrics.train_runtime.avg -= warmup_seconds
            training_metrics.train_runtime.min -= warmup_seconds
            training_metrics.train_runtime.max -= warmup_seconds

        return Metrics(
            gpus=gpu_metrics,
            system=system_metrics,
            training=training_metrics,
            model=ModelMetrics(model_load_time=aggregate_values(model_load_time)),
            hostname_gpus={
                metrics_info["hostname"]: metrics_info["cuda_visible_devices"]
            },
            train_time_start=train_time_start,
            train_time_stop=train_time_stop,
            training_steps=training_steps,
            warmup_steps=metrics_info.get("warmup_steps"),
            warmup_seconds=metrics_info.get("warmup_seconds"),
        )

    def filter_unused_gpus(self) -> None:
        """Uses CUDA_VISIBLE_DEVICES to filter out metrics of GPUs that the task did not use

        Updates the metrics in place
        """
        # VV: CUDA_VISIBLE_DEVICES can be a comma separated list of GPU indices that this process
        # has access to. If this is the case, we can use this information to filter the field `gpus` of metrics
        try:
            cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            cuda_devices = [int(x) for x in cuda_devices.split(",")]
            total_gpus = len(self.gpus)

            if len(cuda_devices) and all(x < total_gpus for x in cuda_devices):
                self.gpus = [self.gpus[idx] for idx in sorted(cuda_devices)]
        except (KeyError, ValueError):
            pass


@dataclasses.dataclass
class SnapshotGPUMetrics:
    gpu: float | None = None
    gpu_memory_percent: float | None = None
    gpu_power_watts: float | None = None
    gpu_power_percent: float | None = None

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


@dataclasses.dataclass
class SnapshotSystemMetrics:
    cpu: float | None = None
    memory_percent: float | None = None

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


def extract_gpu_metrics(
    gpu_index: int,
    handle: object,
) -> SnapshotGPUMetrics:
    """Extracts as many metrics as possible for a GPU given its index and handle."""
    import pynvml

    ret: dict[str, float] = {}

    try:
        ret["gpu"] = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception as e:
        log.warning("Failed to get GPU utilization for GPU %s: %s", gpu_index, e)

    try:
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        ret["gpu_memory_percent"] = float(gpu_mem.used * 100.0 / gpu_mem.total)
    except Exception as e:
        log.warning("Failed to get GPU memory for GPU %s: %s", gpu_index, e)

    try:
        power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        power_cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        ret["gpu_power_watts"] = float(power_w)
        ret["gpu_power_percent"] = float(power_w * 100.0 / power_cap)
    except Exception as e:
        log.warning("Failed to get GPU power for GPU %s: %s", gpu_index, e)

    return SnapshotGPUMetrics(**ret)


def extract_system_metrics(process: "psutil.Process") -> SnapshotSystemMetrics:
    """Extracts as many system metrics as possible for a process and the node.

    Args:
        process:
            The process whose CPU utilization is measured. CPU utilization is
            computed since the previous call with the same object, so callers
            must reuse it across calls.
    """
    import psutil

    ret: dict[str, float] = {}

    try:
        ret["cpu"] = process.cpu_percent(None)
    except Exception as e:
        log.warning("Failed to get CPU utilization: %s", e)

    try:
        mem = psutil.virtual_memory()
        ret["memory_percent"] = mem.used * 100.0 / mem.total
    except Exception as e:
        log.warning("Failed to get system memory usage: %s", e)

    return SnapshotSystemMetrics(**ret)


@dataclasses.dataclass
class ResourceSnapshot:
    timestamp: dt.datetime
    system: SnapshotSystemMetrics
    gpus: list[SnapshotGPUMetrics]


class ResourceTracker:
    def __init__(self, period: float = 10.0) -> None:
        """Takes a snapshot of system metrics every @period seconds

        Args:
            period:
                Seconds between successive snapshots
        """
        import psutil

        self._log = logging.getLogger("ResourceTracker")

        # VV: psutil computes cpu_percent() since the previous call on the same Process object,
        # so we keep one around. The first call has no reference point and reports 0.0, therefore
        # we use it to prime the object and drop its cpu value from the initial snapshot.
        self._process = psutil.Process()
        initial_snapshot = self._collect_snapshot()
        initial_snapshot.system.cpu = None
        self.metrics: list[ResourceSnapshot] = [initial_snapshot]

        # VV: end_track() puts a message here asking the thread to stop
        self._end_requested = Event()
        self._stopped = Event()

        self._thread: Thread | None = None
        self._stopped.set()
        self._period = period

    def _collect_snapshot(self) -> ResourceSnapshot:
        system = extract_system_metrics(self._process)

        gpus: list[SnapshotGPUMetrics] = []
        try:
            import pynvml

            pynvml.nvmlInit()
            num_gpus = pynvml.nvmlDeviceGetCount()
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpus.append(extract_gpu_metrics(i, handle))
            pynvml.nvmlShutdown()
        except Exception as e:
            self._log.debug("NVML metrics collection skipped: %s", e)

        return ResourceSnapshot(
            timestamp=dt.datetime.now(),
            system=system,
            gpus=gpus,
        )

    def track_stats(self) -> None:
        snapshot = self._collect_snapshot()
        self._log.debug("Resources Snapshot: %s", snapshot)
        self.metrics.append(snapshot)

    def _kernel(self: "ResourceTracker") -> None:
        """Utility method which takes a metric snapshot every period seconds and can be cancelled via end_track()"""

        while self._end_requested.wait(timeout=self._period) is False:
            self.track_stats()

        # VV: ensure at least 1 measurement but instead of taking it at the start of the measurements
        # track the metrics right after the method was asked to shutdown. This way we don't
        # capture the state of machine "before" the training started
        self.track_stats()

        self._stopped.set()

    def begin_track(self) -> None:
        self._stopped.clear()

        self._thread = Thread(target=self._kernel)
        self._thread.start()

    def end_track(self) -> None:
        if self._thread is not None:
            self._log.debug("Stopping stat tracker thread")
            self._end_requested.set()
            self._thread.join()

            # VV: Just some sanity checking
            if not self._stopped.is_set():
                raise ValueError("_stopped was not set as expected")

            self._thread = None

    def to_resource_tracker_metrics(
        self, keep_after_time: dt.datetime | None
    ) -> list[ResourceTrackerMetric]:
        """Converts stored resource snapshots into metric timeseries.

        The resource tracker records snapshots of the system at each step. This
        method splits those snapshots into individual named timeseries, one per
        metric, using the following naming and context conventions:

        - System-level metrics are named ``system__<metric>``, with an empty context.
        - Per-GPU metrics are named ``system__<metric>``, with a context of
          ``{"gpu": <gpu_index>}``.

        Returns:
            A list of :class:`ResourceTrackerMetric` objects, each representing a single
            timeseries for one system or GPU metric.
        """
        # VV: The resource tracker is storing snapshots of the system,
        # here we are splitting the snapshots into timeseries of individual metrics

        # VV: Format is: {$gpuID: {$metric: [value1, value2,...] }}
        gpu_measurements: dict[int, dict[str, list[float | int]]] = {}

        # VV: Format is: {$metric: [value1, value2,...]}
        system_measurements: dict[str, list[float | int]] = {}

        for m in self.metrics:
            if keep_after_time is not None and m.timestamp < keep_after_time:
                continue

            for index, gpu in enumerate(m.gpus):
                if index not in gpu_measurements:
                    gpu_measurements[index] = {}

                for name, value in gpu.to_dict().items():
                    if name not in gpu_measurements[index]:
                        gpu_measurements[index][name] = []

                    gpu_measurements[index][name].append(value)
            for name, value in m.system.to_dict().items():
                if name not in system_measurements:
                    system_measurements[name] = []
                system_measurements[name].append(value)
        # VV: Metrics are named like so __system__$metric
        # For GPU metrics, we also insert a context with the key: value pair: "gpu: $gpuIndex"
        metrics = []

        for name, values in system_measurements.items():
            metrics.append(
                ResourceTrackerMetric(
                    name=f"__system__{name}", context={}, values=values
                )
            )

        for gpu_index, measurements in gpu_measurements.items():
            for name, values in measurements.items():
                metrics.append(
                    ResourceTrackerMetric(
                        name=f"__system__{name}",
                        context={"gpu": gpu_index},
                        values=values,
                    )
                )

        return metrics
