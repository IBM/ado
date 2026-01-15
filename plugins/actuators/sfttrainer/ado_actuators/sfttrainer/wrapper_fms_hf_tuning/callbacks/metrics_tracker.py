# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import dataclasses
import datetime
import logging
import os
import typing
from threading import Event, Thread

import aim.ext.resource.stat
import psutil

if typing.TYPE_CHECKING:
    from tuning.sft_trainer import EnhancedTrainOutput


def was_gpu_in_use(
    gpu_cuda_id: int, cuda_visible_devices: list[int], num_gpus: int
) -> bool:
    # VV: We used that GPU if cuda_visible_devices is empty BUT we did ask
    # FSDP to use at least 1 GPU. OR the gpu cuda id is in the cuda_visible_devices array (i.e. AIM/pytorch knew
    # exactly which of the GPUs available on the system it was meant to use)
    return (not cuda_visible_devices and num_gpus > 0) or (
        gpu_cuda_id in cuda_visible_devices
    )


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
    gpu_temp: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues, metadata={"help": "GPU temperature"}
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
    def from_aim_info_dict(cls, info: dict[str, typing.Any]) -> "GPUMetrics":
        return GPUMetrics(
            gpu=aggregate_values(info["__system__gpu"]["values"]),
            gpu_memory_percent=aggregate_values(
                info["__system__gpu_memory_percent"]["values"]
            ),
            gpu_temp=aggregate_values(info["__system__gpu_temp"]["values"]),
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
    p_memory_percent: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues,
        metadata={"help": "Memory occupied by process"},
    )
    disk_percent: AggregatedValues = dataclasses.field(
        default_factory=AggregatedValues, metadata={"help": "Disk usage"}
    )

    @classmethod
    def from_aim_info_dict(cls, info: dict[str, typing.Any]) -> "SystemMetrics":
        return SystemMetrics(
            cpu=aggregate_values(info["__system__cpu"]["values"]),
            memory_percent=aggregate_values(info["__system__memory_percent"]["values"]),
            p_memory_percent=aggregate_values(
                info["__system__p_memory_percent"]["values"]
            ),
            disk_percent=aggregate_values(info["__system__disk_percent"]["values"]),
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
    def from_aim_info_list(
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
    aim_run_hash: str | None = None
    train_time_start: datetime.datetime | None = None
    train_time_stop: datetime.datetime | None = None
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

        if self.aim_run_hash:
            scalar_observations["aim_run_hash"] = self.aim_run_hash

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
    def from_aim_info_dict(
        cls, aim_info: dict[str, typing.Any], num_gpus: int
    ) -> "Metrics":
        json_metrics = aim_info["metrics"]

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

        training_metrics = TrainingMetrics.from_aim_info_list(training_metrics)

        gpus = {}

        for m in json_metrics:
            if m["name"].startswith("__system__gpu"):
                gpu_cuda_id = m["context"]["gpu"]

                if gpu_cuda_id not in gpus:
                    gpus[gpu_cuda_id] = {
                        # VV: init all metrics just in case aim couldn't extract any of them
                        x: {"values": [], "context": {"gpu": gpu_cuda_id}, "name": x}
                        for x in [
                            "__system__gpu",
                            "__system__gpu_temp",
                            "__system__gpu_power_watts",
                            "__system__gpu_memory_percent",
                            "__system__gpu_power_percent",
                        ]
                    }

                gpus[gpu_cuda_id][m["name"]] = m

        gpu_metrics = []
        for gpu_cuda_id, gpu_info in gpus.items():
            if was_gpu_in_use(gpu_cuda_id, aim_info["cuda_visible_devices"], num_gpus):
                gpu_metrics.append(GPUMetrics.from_aim_info_dict(gpu_info))

        system_metrics = {
            # VV: init all metrics just in case aim couldn't extract any of them
            x: {"values": [], "context": {}, "name": x}
            for x in [
                "__system__cpu",
                "__system__disk_percent",
                "__system__memory_percent",
                "__system__p_memory_percent",
            ]
        }

        for m in json_metrics:
            if m["name"] in system_metrics:
                system_metrics[m["name"]] = m

        system_metrics = SystemMetrics.from_aim_info_dict(system_metrics)

        model_load_time = [-1.0]

        for m in json_metrics:
            if m["name"] == "model_load_time":
                model_load_time = m["values"]

        train_time_start = aim_info.get("train_time_start")
        train_time_stop = aim_info.get("train_time_stop")
        format_time = "%d%m%y-%H%M%S"

        if train_time_start:
            train_time_start = datetime.datetime.strptime(train_time_start, format_time)
        if train_time_stop:
            train_time_stop = datetime.datetime.strptime(train_time_stop, format_time)

        # VV: training_steps does not include the warmup_steps
        training_steps = aim_info["training_steps"] - (
            aim_info.get("warmup_steps") or 0
        )

        warmup_seconds = aim_info.get("warmup_seconds") or 0

        if warmup_seconds > 0:
            # VV: Just like training_steps, train_runtime does not include the warmup_seconds
            training_metrics.train_runtime.avg -= warmup_seconds
            training_metrics.train_runtime.min -= warmup_seconds
            training_metrics.train_runtime.max -= warmup_seconds

        return Metrics(
            gpus=gpu_metrics,
            system=system_metrics,
            training=training_metrics,
            model=ModelMetrics(model_load_time=aggregate_values(model_load_time)),
            aim_run_hash=aim_info.get("run_hash"),
            hostname_gpus={aim_info["hostname"]: aim_info["cuda_visible_devices"]},
            train_time_start=train_time_start,
            train_time_stop=train_time_stop,
            training_steps=training_steps,
            warmup_steps=aim_info.get("warmup_steps"),
            warmup_seconds=aim_info.get("warmup_seconds"),
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


def round10e5(val: float) -> float:
    return round(val * 10e5) / 10e5


class ResourceTracker:
    AGG_MODE_AVG = "average"
    AGG_MODE_MIN = "min"
    AGG_MODE_MAX = "max"
    AGG_MODE_DIFF = "diff"
    AGG_NOTHING = "nothing"
    AGG_DEFAULT = AGG_NOTHING

    @classmethod
    def aggregate(cls, items: list, mode: str) -> float | list:
        """
        Aggregates array of numbers by a given 'mode'
        """
        if mode == cls.AGG_MODE_MAX:
            return max(items)
        if mode == cls.AGG_MODE_MIN:
            return min(items)
        if mode == cls.AGG_MODE_AVG:
            return round10e5(sum(items) / len(items))
        if mode == cls.AGG_MODE_DIFF:
            return round10e5(max(items) - min(items))
        if mode == cls.AGG_NOTHING:
            return items
        raise ValueError(f"unknown aggregation mode: '{mode}'")

    def __init__(self, period: float = 30.0) -> None:
        """Takes a snapshot of system metrics every @period seconds

        Args:
            period:
                Seconds between successive snapshots
        """
        self._log = logging.getLogger("ResourceTracker")
        self._process = psutil.Process()
        self._stat = aim.ext.resource.stat.Stat(self._process)
        self._fix_stat(self._stat.stat_item)

        self.metrics: list[aim.ext.resource.stat.StatDict] = [self._stat.stat_item]

        # VV: end_track() puts a message here asking the thread to stop
        self._end_requested = Event()
        self._stopped = Event()

        self._thread: Thread | None = None
        self._stopped.set()
        self._period = period

    def track_stats(self) -> None:
        system, gpus = self._stat.get_stats()
        stat = aim.ext.resource.stat.StatDict(system, gpus)

        try:
            self._fix_stat(stat)
        finally:
            self._log.debug(f"Resources Snapshot {stat.to_dict()}")

        self.metrics.append(stat)

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

    @classmethod
    def _fix_stat(cls, stat: aim.ext.resource.stat.StatDict) -> None:
        # VV: There's a bug in AIM which causes some GPU fields to be a tuple of 1 item instead of a float
        for g in stat.gpus:
            for name in g:
                if isinstance(g[name], tuple):
                    if len(g[name]) != 1:
                        raise ValueError(f"Unexpected GPU stat {name} in GPU {g}")
                    g[name] = g[name][0]

    @classmethod
    def aggregate_items(
        cls,
        items: list[aim.ext.resource.stat.StatDict],
        agg_mode: str = AGG_NOTHING,
    ) -> aim.ext.resource.stat.StatDict:
        """
        Aggregates array of `StatDict` items by a given `mode`
        """
        aggregated_stat = aim.ext.resource.stat.StatDict()

        # Return empty item if items array is empty
        if not items or len(items) == 0:
            return aggregated_stat

        gpu_stats = []
        for s in items:
            # Collect system stats
            for k in s.system:
                aggregated_stat.system.setdefault(k, [])
                aggregated_stat.system[k].append(s.system[k])

            # Collect GPU device stats
            for stat_item_gpu_idx in range(len(s.gpus)):
                stat_item_gpu_stat = s.gpus[stat_item_gpu_idx]
                if len(gpu_stats) == stat_item_gpu_idx:
                    gpu_stats.append({})
                for gpu_stat_key in stat_item_gpu_stat:
                    gpu_stat = stat_item_gpu_stat[gpu_stat_key]
                    gpu_stats[stat_item_gpu_idx].setdefault(gpu_stat_key, [])
                    gpu_stats[stat_item_gpu_idx][gpu_stat_key].append(gpu_stat)

        # Aggregate system stats
        for k in aggregated_stat.system:
            aggregated_stat.system[k] = cls.aggregate(
                aggregated_stat.system[k], agg_mode
            )

        # Aggregate GPU device stats
        for g in range(len(gpu_stats)):
            for k in gpu_stats[g]:
                gpu_stats[g][k] = cls.aggregate(gpu_stats[g][k], agg_mode)

        aggregated_stat.gpus = gpu_stats

        return aggregated_stat

    def to_metrics(self, train_output: "EnhancedTrainOutput | None") -> Metrics:
        self._log.debug("Waiting for stat tracker thread to finish")
        self._stopped.wait()

        aggregate = self.aggregate_items(self.metrics, agg_mode=self.AGG_NOTHING)
        gpus = [
            GPUMetrics(**{k: aggregate_values(v) for k, v in g.items()})
            for g in aggregate.gpus
        ]

        system = SystemMetrics(
            **{k: aggregate_values(v) for k, v in aggregate.system.items()}
        )
        train_metrics = {}
        model_load_time = -1

        if train_output is not None:
            model_load_time = train_output.model_load_time
            train_metrics.update(
                {
                    k.name: train_output.train_output.metrics[k.name]
                    for k in dataclasses.fields(TrainingMetrics)
                    if k.name in train_output.train_output.metrics
                }
            )
        training = TrainingMetrics(**train_metrics)

        model = ModelMetrics(model_load_time=aggregate_values([model_load_time]))
        return Metrics(gpus=gpus, system=system, training=training, model=model)

    @classmethod
    def filter_unused_gpus(cls, metrics: Metrics) -> None:
        """Uses CUDA_VISIBLE_DEVICES to filter out metrics of GPUs that the task did not use

        Args:
            metrics:
                The metrics to update in place
        """
        metrics.filter_unused_gpus()
