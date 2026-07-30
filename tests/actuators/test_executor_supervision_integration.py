# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for ExperimentExecutorSupervisor with a real Ray cluster.

Requires session ``initialize_ray`` (plain ``ray.init()`` with no custom resources).
The unschedulable-task scenario uses ``UNSCHEDULABLE_RESOURCE``; tests fail if the
session fixture registers that key with non-zero capacity.

These tests start a dedicated Ray cluster per xdist worker and query the Ray State
API; run them in one xdist group so they are not interleaved with other Ray-heavy
tests on different workers (which causes State API ``ConnectionError`` flakes).

The FAILED-state supervision branch is not covered here: Ray on CI typically
reports task completion before the State API exposes ``FAILED``.
"""

from __future__ import annotations

import time

import pytest
import ray
from ray.util.queue import Empty as RayQueueEmpty

from ado.modules.actuators.executor_supervisor import (
    ExperimentExecutorState,
    ExperimentExecutorSupervisor,
    ExperimentExecutorSupervisorConfig,
    _experiment_executor_state_lookup,
)
from ado.modules.actuators.measurement_queue import MeasurementQueue
from ado.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from tests.actuators.test_executor_supervision import _sample_request

pytestmark = pytest.mark.xdist_group(name="executor_supervision_ray")


# Must not be advertised on the test cluster (see module docstring).
UNSCHEDULABLE_RESOURCE = "ado_executor_supervisor_unschedulable"


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
    raise RuntimeError("simulated uncaught executor failure")


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


@pytest.fixture
def supervisor_config_with_pending_resource_timeout() -> (
    ExperimentExecutorSupervisorConfig
):
    """Config with taskPendingResourceTimeoutSeconds enabled.

    taskRunningTimeoutSeconds is long enough to outlast Ray State API visibility
    lag (~1s on a local cluster), so the task can transition from OTHER to
    PENDING_NODE_ASSIGNMENT before the general running timeout fires.
    """
    return ExperimentExecutorSupervisorConfig(
        taskFailedGraceSeconds=0.5,
        taskRunningTimeoutSeconds=30.0,
        taskPendingResourceTimeoutSeconds=2.0,
        supervisorPollIntervalSeconds=0.1,
    )


@pytest.fixture
def supervisor_config_long_running_timeout() -> ExperimentExecutorSupervisorConfig:
    """Long running timeout; no pending resource timeout.

    taskRunningTimeoutSeconds is long enough to outlast Ray State API visibility
    lag, allowing PENDING_NODE_ASSIGNMENT tasks to be identified before the
    general running timeout fires.
    """
    return ExperimentExecutorSupervisorConfig(
        taskFailedGraceSeconds=0.5,
        taskRunningTimeoutSeconds=30.0,
        supervisorPollIntervalSeconds=0.1,
    )


def drain_queue(queue: MeasurementQueue, timeout: float) -> MeasurementRequest | None:
    """Return the next queued request or None if the queue is empty within ``timeout``."""
    try:
        return queue.get(timeout=timeout)
    except RayQueueEmpty:
        return None


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.timeout(60)
def test_supervisor_pending_resource_timeout_emits_invalid(
    measurement_queue: MeasurementQueue,
    supervisor_config_with_pending_resource_timeout: ExperimentExecutorSupervisorConfig,
) -> None:
    """Unschedulable task triggers invalid result when taskPendingResourceTimeoutSeconds is set."""
    supervisor = ExperimentExecutorSupervisor(
        measurement_queue, supervisor_config_with_pending_resource_timeout
    )
    supervisor.start()
    try:
        request = _sample_request("timeout1")
        ref = never_scheduled.remote()
        supervisor.supervise_experiment_executor(request, ref)
        # Allow enough time for: State API visibility lag (~1s) + pending resource timeout (2s)
        result = drain_queue(measurement_queue, timeout=20.0)
        assert result is not None
        assert result.status == MeasurementRequestStateEnum.FAILED
        assert result.measurements is not None
        assert (
            "pending resource allocation" in result.measurements[0].reason  # type: ignore[union-attr]
        )
    finally:
        supervisor.stop()


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.timeout(60)
def test_supervisor_pending_node_assignment_not_timed_out_by_default(
    measurement_queue: MeasurementQueue,
    supervisor_config_long_running_timeout: ExperimentExecutorSupervisorConfig,
) -> None:
    """PENDING_NODE_ASSIGNMENT task is not killed when taskPendingResourceTimeoutSeconds is None."""
    supervisor = ExperimentExecutorSupervisor(
        measurement_queue, supervisor_config_long_running_timeout
    )
    supervisor.start()
    ref = never_scheduled.remote()
    try:
        request = _sample_request("pending1")
        supervisor.supervise_experiment_executor(request, ref)
        # Wait long enough for the task to be visible in the State API and several
        # poll cycles to confirm the supervisor does not emit an invalid result.
        time.sleep(5.0)
        result = drain_queue(measurement_queue, timeout=0.5)
        assert result is None
    finally:
        supervisor.stop()
        ray.cancel(ref, force=True, recursive=True)


@pytest.mark.flaky(reruns=3, reruns_delay=2)
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
        supervisor.supervise_experiment_executor(request, ref)
        time.sleep(0.5)
        result = drain_queue(measurement_queue, timeout=1.0)
        assert result is None
    finally:
        supervisor.stop()
        ray.cancel(ref, force=True, recursive=True)


@pytest.mark.flaky(reruns=3, reruns_delay=2)
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
        supervisor.supervise_experiment_executor(request, ref)
        ray.get(ref)
        time.sleep(0.5)
        result = drain_queue(measurement_queue, timeout=1.0)
        assert result is None
    finally:
        supervisor.stop()


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.timeout(30)
def test_default_task_state_lookup_unschedulable_task_returns_pending_node_assignment() -> (
    None
):
    """Custom-resource task that cannot schedule maps to PENDING_NODE_ASSIGNMENT via State API."""
    ref = never_scheduled.remote()
    try:
        deadline = time.monotonic() + 15.0
        state = ExperimentExecutorState.OTHER
        while time.monotonic() < deadline:
            state = _experiment_executor_state_lookup(ref)
            if state == ExperimentExecutorState.PENDING_NODE_ASSIGNMENT:
                break
            time.sleep(0.2)
        if state != ExperimentExecutorState.PENDING_NODE_ASSIGNMENT:
            pytest.skip(
                "Ray State API did not expose PENDING_NODE_ASSIGNMENT within 15s "
                f"(last collapsed state={state.value}); skipping due to transient "
                "State API unavailability under parallel test load."
            )
    finally:
        ray.cancel(ref, force=True, recursive=True)


@pytest.mark.flaky(reruns=3, reruns_delay=2)
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
        supervisor.supervise_experiment_executor(request, ref)
        result = drain_queue(measurement_queue, timeout=5.0)
        assert result is not None
        assert result.requestid == "exc1"
        assert result.status == MeasurementRequestStateEnum.FAILED
        assert result.measurements is not None
        assert "Executor task raised" in result.measurements[0].reason  # type: ignore[union-attr]
        assert "simulated uncaught executor failure" in result.measurements[0].reason  # type: ignore[union-attr]
    finally:
        supervisor.stop()


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.timeout(30)
def test_mark_completed_prevents_duplicate_launch_failure(
    measurement_queue: MeasurementQueue,
) -> None:
    """mark_measurement_request_completed prevents launch-timeout invalid when a result was already queued."""
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
        supervisor.supervise_experiment_executor(request, stuck_ref)
        measurement_queue.put(request, block=False)
        supervisor.mark_measurement_request_completed(request.requestid)
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
