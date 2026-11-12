# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import time

from ado_actuators.sfttrainer.wrapper_fms_hf_tuning.callbacks.metrics_tracker import (
    Metrics,
    ResourceTracker,
)


def test_resource_tracker_basic() -> None:
    tracker = ResourceTracker(period=0.1)
    tracker.begin_track()
    time.sleep(0.25)
    tracker.end_track()

    assert len(tracker.metrics) >= 1
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
