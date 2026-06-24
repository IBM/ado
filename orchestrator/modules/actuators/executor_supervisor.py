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
            f"{sorted(missing)} (required by ExperimentExecutorSupervisor). "
            f"Available states: {sorted(api_states)}. "
            "Update orchestrator.modules.actuators.executor_supervisor."
        )


_verify_supervisor_ray_states_supported()


class ExperimentExecutorState(str, enum.Enum):
    """Defines state used by experiment executor supervisor

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
    ) -> ExperimentExecutorState:
        """Map a Ray State API state string to a supervisor ``ExperimentExecutorState``.

        Args:
            raw: ``TaskState.state`` from ``ray.util.state.list_tasks``, or None.
            logger: Logger for unmapped values; defaults to module logger.

        Returns:
            ``RUNNING`` (including ``RUNNING_IN_RAY_GET``/``RUNNING_IN_RAY_WAIT``),
            ``FAILED``, ``PENDING_NODE_ASSIGNMENT``, ``PENDING_OBJ_STORE_MEM_AVAIL``,
            or ``OTHER`` (all other states and unavailable lookups).
        """
        match raw:
            case cls.RUNNING.value | "RUNNING_IN_RAY_GET" | "RUNNING_IN_RAY_WAIT":
                return cls.RUNNING
            case cls.FAILED.value:
                return cls.FAILED
            case cls.PENDING_NODE_ASSIGNMENT.value:
                return cls.PENDING_NODE_ASSIGNMENT
            case cls.PENDING_OBJ_STORE_MEM_AVAIL.value:
                return cls.PENDING_OBJ_STORE_MEM_AVAIL
            case None:
                log = logger or logging.getLogger(__name__)
                log.debug("Ray task state unavailable; treating as %s", cls.OTHER.value)
                return cls.OTHER
            case _:
                return cls.OTHER


_RESOURCE_WAIT_STATES: frozenset[ExperimentExecutorState] = frozenset(
    {
        ExperimentExecutorState.PENDING_NODE_ASSIGNMENT,
        ExperimentExecutorState.PENDING_OBJ_STORE_MEM_AVAIL,
    }
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
class _SupervisedExperimentExecutor:
    """An in-flight executor Ray task registered with the supervisor."""

    request: MeasurementRequest
    executor_ref: ray.ObjectRef
    submitted_at: float
    seen_running: bool = False
    """True once Ray State API has reported RUNNING for this task."""


@dataclass
class _SupervisorState:
    """Mutable supervisor state guarded by a lock."""

    supervised_experiment_executors: dict[str, _SupervisedExperimentExecutor] = field(
        default_factory=dict
    )
    completed_request_ids: set[str] = field(default_factory=set)


def _experiment_executor_state_lookup(
    executor_ref: ray.ObjectRef,
) -> ExperimentExecutorState:
    """Return collapsed supervisor state for an executor ref.

    Uses ``ray.util.state.list_tasks``.  Returns ``RUNNING``, ``FAILED``,
    ``PENDING_NODE_ASSIGNMENT``, ``PENDING_OBJ_STORE_MEM_AVAIL``, or ``OTHER``
    (lookup failure, missing task, or any other Ray state).
    """
    try:
        task_id = executor_ref.task_id().hex()
    except (AttributeError, RuntimeError, ValueError):
        return ExperimentExecutorState.OTHER

    from ray.util.state import list_tasks

    last_error: Exception | None = None
    tasks: list[object] = []
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            tasks = list_tasks(
                filters=[("task_id", "=", task_id)],
                limit=1,
                raise_on_missing_output=False,
            )
            last_error = None
            break
        except Exception as error:
            last_error = error
            if attempt < max_attempts - 1:
                delay = 0.15 * (attempt + 1)
                if type(error).__name__ == "ServerUnavailable":
                    delay = max(delay, 0.5)
                time.sleep(delay)

    if last_error is not None:
        return ExperimentExecutorState.OTHER

    if not tasks:
        return ExperimentExecutorState.OTHER

    raw_state = getattr(tasks[0], "state", None) or tasks[0].get("state")
    if isinstance(raw_state, ExperimentExecutorState):
        return raw_state
    if isinstance(raw_state, str):
        return ExperimentExecutorState.from_ray_state(raw_state)
    return ExperimentExecutorState.OTHER


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
            target=self._experiment_executor_supervision_loop,
            name="ExperimentExecutorSupervisor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the supervision loop to stop."""
        self._stop.set()

    def supervise_experiment_executor(
        self,
        request: MeasurementRequest,
        executor_ref: ray.ObjectRef,
    ) -> None:
        """Register an executor task for supervision."""
        with self._lock:
            if request.requestid in self._state.completed_request_ids:
                return
            self._state.supervised_experiment_executors[request.requestid] = (
                _SupervisedExperimentExecutor(
                    request=request,
                    executor_ref=executor_ref,
                    submitted_at=time.monotonic(),
                )
            )

    def mark_measurement_request_completed(self, requestid: str) -> None:
        """Record that MeasurementRequest requestid has queued a result.

        This is to avoid sending duplicate results for a request in the case
        an external problem causes the task to FAIL with no associated exception
        e.g. raylet failure, node failure, some issue with ray ref retrieval.
        """
        with self._lock:
            self._state.completed_request_ids.add(requestid)
            self._state.supervised_experiment_executors.pop(requestid, None)

    def _experiment_executor_supervision_loop(self) -> None:
        """Poll pending executor tasks until stopped."""

        while not self._stop.is_set():
            with self._lock:
                experiment_executors = list(
                    self._state.supervised_experiment_executors.values()
                )

            for experiment_executor in experiment_executors:
                self._check_experiment_executor(experiment_executor)

            time.sleep(self._config.supervisorPollIntervalSeconds)

    def _check_experiment_executor(
        self, experiment_executor: _SupervisedExperimentExecutor
    ) -> None:
        """Evaluate an experiment executor."""

        # Check if we've been notified that it completed
        requestid = experiment_executor.request.requestid
        with self._lock:
            if requestid in self._state.completed_request_ids:
                self._state.supervised_experiment_executors.pop(requestid, None)
                return

        # Check if it is done directly
        # If it is this will handle if it exited with an exception
        if self._check_and_handle_experiment_executor_completed(experiment_executor):
            return

        # Check if its running
        # If it is mark it so we don't have to check for launch timeout
        elapsed = time.monotonic() - experiment_executor.submitted_at
        executor_state = _experiment_executor_state_lookup(
            experiment_executor.executor_ref
        )
        if executor_state == ExperimentExecutorState.RUNNING:
            with self._lock:
                if requestid in self._state.supervised_experiment_executors:
                    self._state.supervised_experiment_executors[
                        requestid
                    ].seen_running = True
            return

        # Check if its Failed - if we are here this should usually mean it failed to launch
        # It can also mean it Failed during execution but didn't return a ref/exception
        # due to some ray infrastructure issue.
        # This case is gated by taskFailedGraceSeconds
        # i.e. we give Failed task this long to return a ref and be handled by
        # _check_and_handle_experiment_executor_completed
        if executor_state == ExperimentExecutorState.FAILED:
            if elapsed >= self._config.taskFailedGraceSeconds:
                self._handle_experiment_executor_launch_failure(
                    experiment_executor,
                    reason=(
                        "Measurement task failed before completion "
                        f"(Ray state={executor_state.value})"
                    ),
                )
            return

        # The following two checks are relevant if pending timeout is set
        if executor_state in _RESOURCE_WAIT_STATES:
            resource_timeout = self._config.taskPendingResourceTimeoutSeconds
            if resource_timeout is not None and elapsed >= resource_timeout:
                self._handle_experiment_executor_launch_failure(
                    experiment_executor,
                    reason=(
                        "Measurement task pending resource allocation for "
                        f"{int(resource_timeout)}s "
                        f"(Ray state={executor_state.value})"
                    ),
                )
            return

        if executor_state == ExperimentExecutorState.OTHER:
            resource_timeout = self._config.taskPendingResourceTimeoutSeconds
            if resource_timeout is not None and elapsed >= resource_timeout:
                self._handle_experiment_executor_launch_failure(
                    experiment_executor,
                    reason=(
                        "Measurement task pending resource allocation for "
                        f"{int(resource_timeout)}s "
                        "(Ray state unavailable or pending scheduling)"
                    ),
                )
            return

        if experiment_executor.seen_running:
            return

        # This is for all other non-running non-pending states
        if elapsed >= self._config.taskRunningTimeoutSeconds:
            with self._lock:
                if requestid in self._state.completed_request_ids:
                    self._state.supervised_experiment_executors.pop(requestid, None)
                    return
            self._handle_experiment_executor_launch_failure(
                experiment_executor,
                reason=(
                    "Measurement task did not start within "
                    f"{int(self._config.taskRunningTimeoutSeconds)}s "
                    "(scheduling/runtime_env)"
                ),
            )

    def _check_and_handle_experiment_executor_completed(
        self, experiment_executor: _SupervisedExperimentExecutor
    ) -> bool:
        """Check if executor finished; if so handle"""

        completed = False
        ready_refs, _ = ray.wait([experiment_executor.executor_ref], timeout=0)
        if ready_refs:
            completed = True
            try:
                ray.get(experiment_executor.executor_ref)
            except Exception as error:
                self._record_experiment_executor_failure(
                    experiment_executor,
                    reason=f"Executor task raised: {error}",
                )
            else:
                self.mark_measurement_request_completed(
                    experiment_executor.request.requestid
                )

        return completed

    def _record_experiment_executor_failure(
        self, experiment_executor: _SupervisedExperimentExecutor, reason: str
    ) -> None:
        """Set an InvalidMeasurement result for the executor unless it was already marked completed."""

        requestid = experiment_executor.request.requestid
        with self._lock:
            if requestid in self._state.completed_request_ids:
                self._log.warning(
                    "Executor failure for request %s after result queued; ignoring: %s",
                    requestid,
                    reason,
                )
                self._state.supervised_experiment_executors.pop(requestid, None)
                return
            self._state.completed_request_ids.add(requestid)

        failed_request = add_invalid_measurement_results(
            experiment_executor.request.model_copy(deep=True),
            reason=reason,
        )
        self._queue.put(failed_request, block=False)
        self._log.warning(
            "Launch failure for request %s (index=%s): %s",
            requestid,
            experiment_executor.request.requestIndex,
            reason,
        )
        with self._lock:
            self._state.supervised_experiment_executors.pop(requestid, None)

    def _handle_experiment_executor_launch_failure(
        self, experiment_executor: _SupervisedExperimentExecutor, reason: str
    ) -> None:
        """Handle when experiment executor fails to launch within timeout"""

        # Check if it did complete - there can be race between ray API and actually putting result
        if self._check_and_handle_experiment_executor_completed(experiment_executor):
            return

        requestid = experiment_executor.request.requestid

        # Check if it notified it put a result and then something happened
        with self._lock:
            if requestid in self._state.completed_request_ids:
                self._state.supervised_experiment_executors.pop(requestid, None)
                return

        self._record_experiment_executor_failure(experiment_executor, reason=reason)
        self._request_experiment_executor_cancellation(experiment_executor.executor_ref)

    def _request_experiment_executor_cancellation(
        self, executor_ref: ray.ObjectRef
    ) -> None:
        """Best-effort cancellation of a stuck executor task."""
        try:
            ray.cancel(executor_ref, force=True, recursive=True)
        except (TypeError, ValueError, RuntimeError) as error:
            self._log.debug("Could not cancel executor ref: %s", error)
