# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import datetime as dt
import time

import pytest
from ado_actuators.sfttrainer.wrapper_fms_hf_tuning.callbacks.metrics_tracker import (
    Metrics,
    ResourceSnapshot,
    ResourceTracker,
    SnapshotGPUMetrics,
    SnapshotSystemMetrics,
    TrainingMetrics,
)


def test_resource_tracker_basic() -> None:
    # VV: The actuator does not run this code, the job on the Ray Worker that
    # is wrapping fms-hf-tuning does
    pytest.importorskip("psutil", reason="psutil is not installed")

    tracker = ResourceTracker(period=0.1)
    tracker.begin_track()
    time.sleep(0.25)
    tracker.end_track()

    assert len(tracker.metrics) >= 2
    resource_metrics = tracker.to_resource_tracker_metrics(keep_after_time=None)
    assert len(resource_metrics) >= 2  # cpu, memory_percent
    metric_names = [m.name for m in resource_metrics]
    assert "__system__cpu" in metric_names
    assert "__system__memory_percent" in metric_names


def test_metrics_from_metrics_info_dict() -> None:
    sample_info = {
        "metrics": [
            {
                "name": "__system__cpu",
                "values": {"avg": 2.5, "min": 1.0, "max": 4.0},
                "context": {},
            },
            {
                "name": "__system__memory_percent",
                "values": {"avg": 50.0, "min": 45.0, "max": 55.0},
                "context": {},
            },
            {
                "name": "__system__gpu",
                "values": {"avg": 85.0, "min": 80.0, "max": 90.0},
                "context": {"gpu": 0},
            },
            {
                "name": "__system__gpu_memory_percent",
                "values": {"avg": 70.0, "min": 65.0, "max": 75.0},
                "context": {"gpu": 0},
            },
            {
                "name": "__system__gpu_power_watts",
                "values": {"avg": 250.0, "min": 200.0, "max": 300.0},
                "context": {"gpu": 0},
            },
            {
                "name": "__system__gpu_power_percent",
                "values": {"avg": 62.5, "min": 50.0, "max": 75.0},
                "context": {"gpu": 0},
            },
        ],
        "hostname": "test-host",
        "train_time_start": "041126-120000",
        "train_time_stop": "041126-120010",
        "cuda_visible_devices": [0],
        "world_rank": "0",
        "world_size": "1",
        "training_steps": 10,
        "warmup_steps": None,
        "warmup_seconds": None,
    }

    parsed = Metrics.from_metrics_info_dict(metrics_info=sample_info, num_gpus=1)
    observations = parsed.to_scalar_observations(distributed_backend=None, world_size=1)

    assert observations["cpu_compute_utilization"] == 2.5
    assert observations["gpu_compute_utilization_avg"] == 85.0
    assert observations["gpu_memory_utilization_avg"] == 70.0
    assert observations["gpu_power_watts_avg"] == 250.0
    assert observations["training_steps"] == 10


def test_to_resource_tracker_metrics_keep_after_time() -> None:
    # VV: The actuator does not run this code, the job on the Ray Worker that
    # is wrapping fms-hf-tuning does
    pytest.importorskip("psutil", reason="psutil is not installed")
    tracker = ResourceTracker(period=0.1)
    start = dt.datetime(2026, 1, 1, 12, 0, 0)
    tracker.metrics = [
        ResourceSnapshot(
            timestamp=start + dt.timedelta(seconds=seconds),
            system=SnapshotSystemMetrics(cpu=cpu, memory_percent=cpu * 10),
            gpus=[SnapshotGPUMetrics(gpu=cpu * 100)],
        )
        for seconds, cpu in enumerate([1.0, 2.0, 3.0, 4.0])
    ]

    def values_by_name(
        keep_after_time: dt.datetime | None,
    ) -> dict[str, list[float | int]]:
        return {
            m.name: m.values
            for m in tracker.to_resource_tracker_metrics(keep_after_time)
        }

    assert values_by_name(None)["__system__cpu"] == [1.0, 2.0, 3.0, 4.0]

    # Snapshots before keep_after_time are dropped, the one at keep_after_time is kept
    filtered = values_by_name(start + dt.timedelta(seconds=2))
    assert filtered["__system__cpu"] == [3.0, 4.0]
    assert filtered["__system__memory_percent"] == [30.0, 40.0]
    assert filtered["__system__gpu"] == [300.0, 400.0]

    assert values_by_name(start + dt.timedelta(seconds=5)) == {}


def _metrics_info(
    warmup_seconds: float | None, runtime_values: list[float | None] | None
) -> dict:
    metrics = [
        {"name": name, "values": [1.0], "context": {}}
        for name in ("samples_per_second", "steps_per_second", "tokens_per_second")
    ]
    if runtime_values is not None:
        metrics.append({"name": "runtime", "values": runtime_values, "context": {}})

    return {
        "metrics": metrics,
        "hostname": "test-host",
        "train_time_start": "041126-120000",
        "train_time_stop": "041126-120010",
        "cuda_visible_devices": [],
        "training_steps": 10,
        "warmup_steps": None,
        "warmup_seconds": warmup_seconds,
    }


def test_metrics_from_metrics_info_dict_subtracts_warmup_seconds() -> None:
    parsed = Metrics.from_metrics_info_dict(
        metrics_info=_metrics_info(warmup_seconds=10, runtime_values=[100.0, 110.0]),
        num_gpus=0,
    )

    assert parsed.warmup_seconds == 10
    assert parsed.training.train_runtime.avg == 95.0
    assert parsed.training.train_runtime.min == 90.0
    assert parsed.training.train_runtime.max == 100.0


@pytest.mark.parametrize("warmup_seconds", [None, 0])
def test_metrics_from_metrics_info_dict_without_warmup_keeps_runtime(
    warmup_seconds: float | None,
) -> None:
    parsed = Metrics.from_metrics_info_dict(
        metrics_info=_metrics_info(warmup_seconds, runtime_values=[100.0, 110.0]),
        num_gpus=0,
    )

    assert parsed.training.train_runtime.avg == 105.0
    assert parsed.training.train_runtime.min == 100.0
    assert parsed.training.train_runtime.max == 110.0


def test_metrics_from_metrics_info_dict_warmup_seconds_without_runtime_keeps_default() -> (
    None
):
    # Only the global zero rank records the runtime, other local-rank 0 workers
    # report warmup_seconds without one
    parsed = Metrics.from_metrics_info_dict(
        metrics_info=_metrics_info(warmup_seconds=10, runtime_values=None),
        num_gpus=0,
    )

    assert parsed.warmup_seconds == 10
    assert parsed.training.train_runtime == TrainingMetrics().train_runtime


def test_metrics_from_metrics_info_dict_warmup_seconds_exceeding_runtime_raises() -> (
    None
):
    with pytest.raises(ValueError, match="warmup_seconds=10"):
        Metrics.from_metrics_info_dict(
            metrics_info=_metrics_info(warmup_seconds=10, runtime_values=[5.0]),
            num_gpus=0,
        )


def test_metrics_from_metrics_info_dict_warmup_seconds_with_empty_runtime_keeps_none() -> (
    None
):
    # Runtime metric without a value: avg is None, so there is nothing to compare or subtract
    parsed = Metrics.from_metrics_info_dict(
        metrics_info=_metrics_info(warmup_seconds=10, runtime_values=[None]),
        num_gpus=0,
    )

    assert parsed.training.train_runtime.avg is None
    assert parsed.training.train_runtime.min is None
    assert parsed.training.train_runtime.max is None
