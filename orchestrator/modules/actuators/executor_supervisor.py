# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Supervision of Ray measurement executor tasks that fail to start or run."""

from __future__ import annotations

import enum
import logging
import threading
import time
import typing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

import pydantic
import ray
from pydantic import ConfigDict

from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.schema.result import InvalidMeasurementResult
from orchestrator.utilities.support import compute_measurement_status

if TYPE_CHECKING:
    from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
    from orchestrator.schema.request import MeasurementRequest


_SUPERVISOR_RAY_STATE_NAMES = frozenset(
    {
        "RUNNING",
        "RUNNING_IN_RAY_GET",
        "RUNNING_IN_RAY_WAIT",
        "FAILED",
        "PENDING_NODE_ASSIGNMENT",
        "PENDING_OBJ_STORE_MEM_AVAIL",
    }
)


def _ray_api_task_state_names() -> frozenset[str]:
    """Return task state strings declared on Ray's ``TaskState`` schema."""
    from ray.util.state.common import TaskState as RayTaskStateRecord

    annotation = RayTaskStateRecord.__annotations__["state"]
    return frozenset(typing.get_args(annotation))


def _verify_supervisor_ray_states_supported() -> None:
    """Fail fast if Ray's State API no longer exposes RUNNING or FAILED."""
    api_states = _ray_api_task_state_names()
    missing = _SUPERVISOR_RAY_STATE_NAMES - api_states
    if missing:
        raise RuntimeError(
            "Ray State API task states no longer include "
            f"{sorted(missing)} (required by LaunchSupervisor). "
            f"Available states: {sorted(api_states)}. "
            "Update orchestrator.modules.actuators.measurement_launch."
        )


_verify_supervisor_ray_states_supported()


class RayTaskState(str, enum.Enum):
    """Collapsed task state used by launch supervision.

    Ray's State API exposes many lifecycle states; the supervisor collapses them
    into five buckets: running, failed, resource-wait pending, and everything
    else (other transient pending, finished, unknown, or lookup failure).
    """

    RUNNING = "RUNNING"
    FAILED = "FAILED"
    PENDING_NODE_ASSIGNMENT = "PENDING_NODE_ASSIGNMENT"
    PENDING_OBJ_STORE_MEM_AVAIL = "PENDING_OBJ_STORE_MEM_AVAIL"
    OTHER = "OTHER"

    @classmethod
    def from_ray_state(
        cls,
        raw: str | None,
        logger: logging.Logger | None = None,
    ) -> RayTaskState:
        """Map a Ray State API state string to a supervisor ``RayTaskState``.

        Args:
            raw: ``TaskState.state`` from ``ray.util.state.list_tasks``, or None.
            logger: Logger for unmapped values; defaults to module logger.

        Returns:
            ``RUNNING`` (including ``RUNNING_IN_RAY_GET``/``RUNNING_IN_RAY_WAIT``),
            ``FAILED``, ``PENDING_NODE_ASSIGNMENT``, ``PENDING_OBJ_STORE_MEM_AVAIL``,
            or ``OTHER`` (all other states and unavailable lookups).
        """
        if raw in (cls.RUNNING.value, "RUNNING_IN_RAY_GET", "RUNNING_IN_RAY_WAIT"):
            return cls.RUNNING
        if raw == cls.FAILED.value:
            return cls.FAILED
        if raw == cls.PENDING_NODE_ASSIGNMENT.value:
            return cls.PENDING_NODE_ASSIGNMENT
        if raw == cls.PENDING_OBJ_STORE_MEM_AVAIL.value:
            return cls.PENDING_OBJ_STORE_MEM_AVAIL
        log = logger or logging.getLogger(__name__)
        if raw is None:
            log.debug("Ray task state unavailable; treating as %s", cls.OTHER.value)
        else:
            log.debug(
                "Ray task state %r collapsed to %s for launch supervision",
                raw,
                cls.OTHER.value,
            )
        return cls.OTHER


_RESOURCE_WAIT_STATES: frozenset[RayTaskState] = frozenset(
    {RayTaskState.PENDING_NODE_ASSIGNMENT, RayTaskState.PENDING_OBJ_STORE_MEM_AVAIL}
)


class ExperimentExecutorSupervisorConfig(pydantic.BaseModel):
    """Configuration for Ray experiment executor supervision."""

    model_config = ConfigDict(extra="forbid")

    taskFailedGraceSeconds: Annotated[
        float,
        pydantic.Field(
            gt=0,
            description=(
                "Grace period after Ray State API reports FAILED before emitting "
                "an InvalidMeasurementResult for the MeasurementRequest an executor was processing."
            ),
        ),
    ] = 600.0

    taskRunningTimeoutSeconds: Annotated[
        float,
        pydantic.Field(
            gt=0,
            description=(
                "Timeout for an experiment executor task to reach RUNNING state after being started "
                "(scheduling/runtime_env failure)."
            ),
        ),
    ] = 900.0

    supervisorPollIntervalSeconds: Annotated[
        float,
        pydantic.Field(
            gt=0,
            description="Interval between supervisor polls of in-flight executor tasks.",
        ),
    ] = 5.0

    taskPendingResourceTimeoutSeconds: Annotated[
        float | None,
        pydantic.Field(
            gt=0,
            description=(
                "Timeout for a task stuck in PENDING_NODE_ASSIGNMENT or "
                "PENDING_OBJ_STORE_MEM_AVAIL before emitting an InvalidMeasurementResult. "
                "None (default) disables this guard, which is safe for fixed or shared "
                "clusters where resource contention is expected. Set a value on "
                "autoscaling clusters where the scheduler should eventually provision "
                "sufficient resources."
            ),
        ),
    ] = None


class ExperimentExecutorSupervisorParameters(GenericActuatorParameters):
    """Actuator configuration parameters for experiment executor supervsions

    Inherit in an actuator class to add these parameters if you
    want to use ExperimentExecutorSupervisor"""

    model_config = ConfigDict(extra="allow")

    taskFailedGraceSeconds: Annotated[
        float,
        pydantic.Field(gt=0),
    ] = 600.0

    taskRunningTimeoutSeconds: Annotated[
        float,
        pydantic.Field(gt=0),
    ] = 900.0

    supervisorPollIntervalSeconds: Annotated[
        float,
        pydantic.Field(gt=0),
    ] = 5.0

    taskPendingResourceTimeoutSeconds: Annotated[
        float | None,
        pydantic.Field(gt=0),
    ] = None

    def to_supervisor_config(self) -> ExperimentExecutorSupervisorConfig:
        """Build a supervisor config from actuator parameters."""
        return ExperimentExecutorSupervisorConfig(
            taskFailedGraceSeconds=self.taskFailedGraceSeconds,
            taskRunningTimeoutSeconds=self.taskRunningTimeoutSeconds,
            supervisorPollIntervalSeconds=self.supervisorPollIntervalSeconds,
            taskPendingResourceTimeoutSeconds=self.taskPendingResourceTimeoutSeconds,
        )


@dataclass
class _MonitoredExecutor:
    """An in-flight executor Ray task registered with the supervisor."""

    request: MeasurementRequest
    executor_ref: ray.ObjectRef
    submitted_at: float
    seen_running: bool = False
    """True once Ray State API has reported RUNNING for this task."""


@dataclass
class _SupervisorState:
    """Mutable supervisor state guarded by a lock."""

    monitored_executors: dict[str, _MonitoredExecutor] = field(default_factory=dict)
    completed_request_ids: set[str] = field(default_factory=set)


def _default_task_state_lookup(executor_ref: ray.ObjectRef) -> RayTaskState:
    """Return collapsed supervisor state for an executor ref.

    Uses ``ray.util.state.list_tasks``.  Returns ``RUNNING``, ``FAILED``,
    ``PENDING_NODE_ASSIGNMENT``, ``PENDING_OBJ_STORE_MEM_AVAIL``, or ``OTHER``
    (lookup failure, missing task, or any other Ray state).
    """
    try:
        task_id = executor_ref.task_id().hex()
    except (AttributeError, RuntimeError, ValueError):
        return RayTaskState.OTHER

    try:
        from ray.util.state import list_tasks

        tasks = list_tasks(
            filters=[("task_id", "=", task_id)],
            limit=1,
            raise_on_missing_output=False,
        )
    except Exception:
        return RayTaskState.OTHER

    if not tasks:
        return RayTaskState.OTHER

    raw_state = getattr(tasks[0], "state", None) or tasks[0].get("state")
    if isinstance(raw_state, RayTaskState):
        return raw_state
    if isinstance(raw_state, str):
        return RayTaskState.from_ray_state(raw_state)
    return RayTaskState.OTHER


def notify_executor_supervisor_completed(
    notifier: object | None,
    requestid: str,
) -> None:
    """Notify executor supervisor that an executor queue a measurement result.

    ``notifier`` is usually the hosting actuator (in-process or as a Ray actor
    handle). It must expose ``mark_launch_completed``;

    Args:
        notifier: Actuator or supervisor to notify, or None to skip.
        requestid: MeasurementRequest identifier that now has a queued result.
    """
    if notifier is None:
        return
    method = getattr(notifier, "mark_launch_completed", None)
    if method is None:
        method = getattr(notifier, "mark_completed", None)
    if method is None:
        return
    remote_call = getattr(method, "remote", None)
    if remote_call is not None:
        remote_call(requestid)
    else:
        method(requestid)


def add_invalid_measurement_results(
    request: MeasurementRequest,
    reason: str,
) -> MeasurementRequest:
    """Attach per-entity InvalidMeasurementResult values when task fails."""
    measurements = [
        InvalidMeasurementResult(
            entityIdentifier=entity.identifier,
            experimentReference=request.experimentReference,
            reason=reason,
        )
        for entity in request.entities
    ]
    request.measurements = measurements
    request.status = compute_measurement_status(measurements)
    return request


class ExperimentExecutorSupervisor:
    """Monitors Ray executor ObjectRefs and emits Invalid results on unexpected failures.

    Intended for reuse by any actuator that fire-and-forgets Ray measurement tasks.
    """

    def __init__(
        self,
        queue: MeasurementQueue,
        config: ExperimentExecutorSupervisorConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the supervisor.

        Args:
            queue: Measurement queue for launch-failure invalid results.
            config: Executor supervision timeouts and poll interval.
            logger: Optional logger; defaults to module logger.
        """
        self._queue = queue
        self._config = config
        self._log = logger or logging.getLogger(__name__)
        self._state = _SupervisorState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background supervision loop."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ExperimentExecutorSupervisor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the supervision loop to stop."""
        self._stop.set()

    def register(
        self,
        request: MeasurementRequest,
        executor_ref: ray.ObjectRef,
    ) -> None:
        """Register an executor task for supervision."""
        with self._lock:
            if request.requestid in self._state.completed_request_ids:
                return
            self._state.monitored_executors[request.requestid] = _MonitoredExecutor(
                request=request,
                executor_ref=executor_ref,
                submitted_at=time.monotonic(),
            )

    def mark_completed(self, requestid: str) -> None:
        """Record that a requestid has a queued result.

        This is to avoid sending duplicate results for a request in the case
        an external problem causes the task to FAIL with no associated exception
        e.g. raylet failure, node failure, some issue with ray ref retrieval.

        Called from the executor path via :func:`notify_executor_supervisor_completed`
        as soon as the measurement queue receives a result, before the Ray executor
        ref becomes ready.
        """
        with self._lock:
            self._state.completed_request_ids.add(requestid)
            self._state.monitored_executors.pop(requestid, None)

    def _run_loop(self) -> None:
        """Poll pending executor tasks until stopped."""
        while not self._stop.is_set():
            self._poll_once()
            time.sleep(self._config.supervisorPollIntervalSeconds)

    def _poll_once(self) -> None:
        """Run a single supervision pass over pending launches."""
        with self._lock:
            pending_snapshot = list(self._state.monitored_executors.values())

        for pending in pending_snapshot:
            self._check_pending(pending)

    def _check_pending(self, pending: _MonitoredExecutor) -> None:
        """Evaluate one pending executor task."""
        requestid = pending.request.requestid
        with self._lock:
            if requestid in self._state.completed_request_ids:
                self._state.monitored_executors.pop(requestid, None)
                return

        ready_refs, _ = ray.wait([pending.executor_ref], timeout=0)
        if ready_refs:
            self._handle_ready(pending)
            return

        elapsed = time.monotonic() - pending.submitted_at
        task_state = _default_task_state_lookup(pending.executor_ref)
        if task_state == RayTaskState.RUNNING:
            with self._lock:
                if requestid in self._state.monitored_executors:
                    self._state.monitored_executors[requestid].seen_running = True
            return

        if (
            task_state == RayTaskState.FAILED
            and elapsed >= self._config.taskFailedGraceSeconds
        ):
            self._emit_launch_failure(
                pending,
                reason=(
                    "Measurement task failed before completion "
                    f"(Ray state={task_state.value})"
                ),
            )
            return

        if task_state in _RESOURCE_WAIT_STATES:
            resource_timeout = self._config.taskPendingResourceTimeoutSeconds
            if (
                resource_timeout is not None
                and not pending.seen_running
                and elapsed >= resource_timeout
            ):
                self._emit_launch_failure(
                    pending,
                    reason=(
                        "Measurement task pending resource allocation for "
                        f"{int(resource_timeout)}s "
                        f"(Ray state={task_state.value})"
                    ),
                )
            return

        if (
            elapsed >= self._config.taskRunningTimeoutSeconds
            and not pending.seen_running
        ):
            self._emit_launch_failure(
                pending,
                reason=(
                    "Measurement task did not start within "
                    f"{int(self._config.taskRunningTimeoutSeconds)}s "
                    "(scheduling/runtime_env)"
                ),
            )

    def _handle_ready(self, pending: _MonitoredExecutor) -> None:
        """Executor finished; queue invalid on ``ray.get`` failure and unregister."""
        try:
            ray.get(pending.executor_ref)
        except Exception as error:
            self._record_executor_failure(
                pending,
                reason=f"Executor task raised: {error}",
            )
        else:
            with self._lock:
                self._state.completed_request_ids.add(pending.request.requestid)
                self._state.monitored_executors.pop(pending.request.requestid, None)

    def _record_executor_failure(
        self, pending: _MonitoredExecutor, reason: str
    ) -> None:
        """Queue an invalid measurement unless a result was already marked completed."""
        requestid = pending.request.requestid
        with self._lock:
            if requestid in self._state.completed_request_ids:
                self._log.warning(
                    "Executor failure for request %s after result queued; ignoring: %s",
                    requestid,
                    reason,
                )
                self._state.monitored_executors.pop(requestid, None)
                return
            self._state.completed_request_ids.add(requestid)

        failed_request = add_invalid_measurement_results(
            pending.request.model_copy(deep=True),
            reason=reason,
        )
        self._queue.put(failed_request, block=False)
        self._log.warning(
            "Launch supervision failure for request %s (index=%s): %s",
            requestid,
            pending.request.requestIndex,
            reason,
        )
        with self._lock:
            self._state.monitored_executors.pop(requestid, None)

    def _emit_launch_failure(self, pending: _MonitoredExecutor, reason: str) -> None:
        """Queue an invalid measurement for a launch/scheduling failure."""
        ready_refs, _ = ray.wait([pending.executor_ref], timeout=0)
        if ready_refs:
            self._handle_ready(pending)
            return

        self._record_executor_failure(pending, reason=reason)
        self._cancel_executor(pending.executor_ref)

    def _cancel_executor(self, executor_ref: ray.ObjectRef) -> None:
        """Best-effort cancellation of a stuck executor task."""
        try:
            ray.cancel(executor_ref, force=True, recursive=True)
        except (TypeError, ValueError, RuntimeError) as error:
            self._log.debug("Could not cancel executor ref: %s", error)
