# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for LaunchSupervisor with a real Ray cluster.

Requires session ``initialize_ray`` (plain ``ray.init()`` with no custom resources).
The unschedulable-task scenario uses ``UNSCHEDULABLE_RESOURCE``; tests fail if the
session fixture registers that key with non-zero capacity.
"""

from __future__ import annotations

import time

import pytest
import ray
from ray.util.queue import Empty as RayQueueEmpty

from orchestrator.modules.actuators.measurement_launch import (
    LaunchSupervisor,
    LaunchSupervisorConfig,
    RayTaskState,
    _default_task_state_lookup,
)
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from tests.actuators.test_measurement_launch import _sample_request

# Must not be advertised on the test cluster (see module docstring).
UNSCHEDULABLE_RESOURCE = "ado_launch_supervisor_unschedulable"


@ray.remote
def sleep_forever() -> None:
    """Long-running task that stays RUNNING."""
    time.sleep(3600)


@ray.remote
def fail_fast() -> None:
    """Task that fails immediately when executed."""
    raise RuntimeError("integration failure")


@ray.remote
def return_immediately() -> int:
    """Task that completes successfully."""
    return 1


@ray.remote(resources={UNSCHEDULABLE_RESOURCE: 1})
def never_scheduled() -> None:
    """Task that cannot schedule on the default pytest Ray cluster."""
    time.sleep(3600)


@pytest.fixture
def measurement_queue() -> MeasurementQueue:
    """Fresh measurement queue for a supervisor integration test."""
    return MeasurementQueue()


@pytest.fixture
def supervisor_config() -> LaunchSupervisorConfig:
    """Short timeouts for wall-clock integration tests."""
    return LaunchSupervisorConfig(
        launchSchedulingGraceSeconds=0.5,
        launchTimeoutSeconds=1.0,
        launchSupervisorPollIntervalSeconds=0.1,
    )


@pytest.fixture
def supervisor_config_long_launch_timeout() -> LaunchSupervisorConfig:
    """Grace/launch timeouts for tasks that stay pending before RUNNING."""
    return LaunchSupervisorConfig(
        launchSchedulingGraceSeconds=0.5,
        launchTimeoutSeconds=60.0,
        launchSupervisorPollIntervalSeconds=0.1,
    )


def drain_queue(queue: MeasurementQueue, timeout: float) -> MeasurementRequest | None:
    """Return the next queued request or None if the queue is empty within ``timeout``."""
    try:
        return queue.get(timeout=timeout)
    except RayQueueEmpty:
        return None


@pytest.mark.timeout(30)
def test_supervisor_launch_timeout_emits_invalid(
    measurement_queue: MeasurementQueue,
    supervisor_config: LaunchSupervisorConfig,
) -> None:
    """Unschedulable custom-resource task triggers launch-timeout invalid result."""
    supervisor = LaunchSupervisor(measurement_queue, supervisor_config)
    supervisor.start()
    try:
        request = _sample_request("timeout1")
        ref = never_scheduled.remote()
        supervisor.register(request, ref)
        result = drain_queue(measurement_queue, timeout=5.0)
        assert result is not None
        assert result.status == MeasurementRequestStateEnum.FAILED
        assert result.measurements is not None
        assert "did not start" in result.measurements[0].reason  # type: ignore[union-attr]
    finally:
        supervisor.stop()


@pytest.mark.timeout(30)
def test_supervisor_running_task_not_timed_out(
    measurement_queue: MeasurementQueue,
    supervisor_config_long_launch_timeout: LaunchSupervisorConfig,
) -> None:
    """Pending/RUNNING tasks are not subject to launch timeout before it elapses."""
    supervisor = LaunchSupervisor(
        measurement_queue,
        supervisor_config_long_launch_timeout,
    )
    supervisor.start()
    try:
        request = _sample_request("running1")
        ref = sleep_forever.remote()
        supervisor.register(request, ref)
        time.sleep(0.5)
        result = drain_queue(measurement_queue, timeout=1.0)
        assert result is None
    finally:
        supervisor.stop()
        ray.cancel(ref, force=True, recursive=True)


@pytest.mark.timeout(30)
def test_supervisor_failed_task_emits_invalid(
    measurement_queue: MeasurementQueue,
    supervisor_config: LaunchSupervisorConfig,
) -> None:
    """FAILED Ray task state triggers invalid after scheduling grace."""
    supervisor = LaunchSupervisor(measurement_queue, supervisor_config)
    supervisor.start()
    try:
        request = _sample_request("fail1")
        ref = fail_fast.remote()
        submitted_at = time.monotonic()
        supervisor.register(request, ref)
        deadline = submitted_at + 8.0
        result: MeasurementRequest | None = None
        while time.monotonic() < deadline:
            result = drain_queue(measurement_queue, timeout=0.2)
            if result is not None:
                break
            ready, _ = ray.wait([ref], timeout=0)
            if ready:
                pytest.skip(
                    "fail_fast completed before FAILED supervision path was exercised"
                )
            elapsed = time.monotonic() - submitted_at
            if (
                _default_task_state_lookup(ref) == RayTaskState.FAILED
                and elapsed >= supervisor_config.launchSchedulingGraceSeconds
            ):
                time.sleep(supervisor_config.launchSupervisorPollIntervalSeconds * 3)
                result = drain_queue(measurement_queue, timeout=2.0)
                if result is not None:
                    break
            time.sleep(0.05)
        assert result is not None, (
            "supervisor should queue invalid when Ray reports FAILED before the "
            "executor ref is ready"
        )
        assert result.status == MeasurementRequestStateEnum.FAILED
        assert result.measurements is not None
        assert "failed before completion" in result.measurements[0].reason  # type: ignore[union-attr]
    finally:
        supervisor.stop()
        ray.cancel(ref, force=True, recursive=True)


@pytest.mark.timeout(30)
def test_supervisor_completed_task_no_supervisor_put(
    measurement_queue: MeasurementQueue,
    supervisor_config: LaunchSupervisorConfig,
) -> None:
    """When the executor completes, the supervisor unregisters without queueing."""
    supervisor = LaunchSupervisor(measurement_queue, supervisor_config)
    supervisor.start()
    try:
        request = _sample_request("done1")
        ref = return_immediately.remote()
        supervisor.register(request, ref)
        ray.get(ref)
        time.sleep(0.5)
        result = drain_queue(measurement_queue, timeout=1.0)
        assert result is None
    finally:
        supervisor.stop()


@pytest.mark.timeout(30)
def test_default_task_state_lookup_unschedulable_task_returns_other() -> None:
    """Custom-resource task that cannot schedule maps to OTHER via State API."""
    ref = never_scheduled.remote()
    try:
        time.sleep(0.3)
        state = _default_task_state_lookup(ref)
        assert state == RayTaskState.OTHER
    finally:
        ray.cancel(ref, force=True, recursive=True)
