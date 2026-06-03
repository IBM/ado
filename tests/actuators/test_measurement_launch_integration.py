# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for LaunchSupervisor with a real Ray cluster.

Requires session ``initialize_ray`` (plain ``ray.init()`` with no custom resources).
The unschedulable-task scenario uses ``UNSCHEDULABLE_RESOURCE``; tests fail if the
session fixture registers that key with non-zero capacity.

Run this module serially (not under pytest-xdist) to avoid Ray init timeouts.

The FAILED-state supervision branch is not covered here: Ray on CI typically
reports task completion before the State API exposes ``FAILED``.
"""

from __future__ import annotations

import time

import pytest
import ray
from ray.util.queue import Empty as RayQueueEmpty

from orchestrator.modules.actuators.measurement_launch import (
    ExperimentExecutorSupervisor,
    ExperimentExecutorSupervisorConfig,
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
def return_immediately() -> int:
    """Task that completes successfully."""
    return 1


@ray.remote
def raise_before_return() -> None:
    """Task that fails with an uncaught exception (ref becomes ready)."""
    raise RuntimeError("executor boom")


@ray.remote(resources={UNSCHEDULABLE_RESOURCE: 1})
def never_scheduled() -> None:
    """Task that cannot schedule on the default pytest Ray cluster."""
    time.sleep(3600)


@pytest.fixture
def measurement_queue() -> MeasurementQueue:
    """Fresh measurement queue for a supervisor integration test."""
    return MeasurementQueue()


@pytest.fixture
def supervisor_config() -> ExperimentExecutorSupervisorConfig:
    """Short timeouts for wall-clock integration tests."""
    return ExperimentExecutorSupervisorConfig(
        taskFailedGraceSeconds=0.5,
        taskRunningTimeoutSeconds=1.0,
        supervisorPollIntervalSeconds=0.1,
    )


@pytest.fixture
def supervisor_config_long_launch_timeout() -> ExperimentExecutorSupervisorConfig:
    """Grace/launch timeouts for tasks that stay pending before RUNNING."""
    return ExperimentExecutorSupervisorConfig(
        taskFailedGraceSeconds=0.5,
        taskRunningTimeoutSeconds=60.0,
        supervisorPollIntervalSeconds=0.1,
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
    supervisor_config: ExperimentExecutorSupervisorConfig,
) -> None:
    """Unschedulable custom-resource task triggers launch-timeout invalid result."""
    supervisor = ExperimentExecutorSupervisor(measurement_queue, supervisor_config)
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
    supervisor_config_long_launch_timeout: ExperimentExecutorSupervisorConfig,
) -> None:
    """Pending/RUNNING tasks are not subject to launch timeout before it elapses."""
    supervisor = ExperimentExecutorSupervisor(
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
def test_supervisor_completed_task_no_supervisor_put(
    measurement_queue: MeasurementQueue,
    supervisor_config: ExperimentExecutorSupervisorConfig,
) -> None:
    """When the executor completes, the supervisor unregisters without queueing."""
    supervisor = ExperimentExecutorSupervisor(measurement_queue, supervisor_config)
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


@pytest.mark.timeout(30)
def test_supervisor_executor_exception_emits_invalid(
    measurement_queue: MeasurementQueue,
    supervisor_config: ExperimentExecutorSupervisorConfig,
) -> None:
    """Uncaught executor exception surfaces as invalid on the measurement queue."""
    supervisor = ExperimentExecutorSupervisor(measurement_queue, supervisor_config)
    supervisor.start()
    try:
        request = _sample_request("exc1")
        ref = raise_before_return.remote()
        supervisor.register(request, ref)
        result = drain_queue(measurement_queue, timeout=5.0)
        assert result is not None
        assert result.requestid == "exc1"
        assert result.status == MeasurementRequestStateEnum.FAILED
        assert result.measurements is not None
        assert "Executor task raised" in result.measurements[0].reason  # type: ignore[union-attr]
        assert "executor boom" in result.measurements[0].reason  # type: ignore[union-attr]
    finally:
        supervisor.stop()


@pytest.mark.timeout(30)
def test_mark_completed_prevents_duplicate_launch_failure(
    measurement_queue: MeasurementQueue,
) -> None:
    """mark_completed prevents launch-timeout invalid when a result was already queued."""
    config = ExperimentExecutorSupervisorConfig(
        taskFailedGraceSeconds=0.2,
        taskRunningTimeoutSeconds=0.5,
        supervisorPollIntervalSeconds=0.05,
    )
    supervisor = ExperimentExecutorSupervisor(measurement_queue, config)
    supervisor.start()
    stuck_ref = never_scheduled.remote()
    try:
        request = _sample_request("mc1")
        request.status = MeasurementRequestStateEnum.SUCCESS
        supervisor.register(request, stuck_ref)
        measurement_queue.put(request, block=False)
        supervisor.mark_completed(request.requestid)
        result = drain_queue(measurement_queue, timeout=5.0)
        assert result is not None
        assert result.requestid == "mc1"
        assert result.status == MeasurementRequestStateEnum.SUCCESS
        time.sleep(
            config.taskRunningTimeoutSeconds + config.supervisorPollIntervalSeconds * 3
        )
        duplicate = drain_queue(measurement_queue, timeout=0.5)
        assert duplicate is None
    finally:
        supervisor.stop()
        ray.cancel(stuck_ref, force=True, recursive=True)
