# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Supervision of Ray measurement executor tasks that fail to start or run."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

import pydantic
import ray
from pydantic import ConfigDict

from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.schema.result import InvalidMeasurementResult
from orchestrator.utilities.support import compute_measurement_status

if TYPE_CHECKING:
    from collections.abc import Callable

    from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
    from orchestrator.schema.request import MeasurementRequest


class LaunchSupervisorConfig(pydantic.BaseModel):
    """Infrastructure timeouts for Ray executor launch supervision."""

    model_config = ConfigDict(extra="forbid")

    launchSchedulingGraceSeconds: Annotated[
        float,
        pydantic.Field(
            gt=0,
            description=(
                "Grace period while a Ray task is pending (cluster scaling, runtime env)."
            ),
        ),
    ] = 600.0

    launchTimeoutSeconds: Annotated[
        float,
        pydantic.Field(
            gt=0,
            description=(
                "Hard cap for an executor task that never reaches RUNNING "
                "(scheduling/runtime_env failure)."
            ),
        ),
    ] = 900.0

    launchSupervisorPollIntervalSeconds: Annotated[
        float,
        pydantic.Field(
            gt=0,
            description="Interval between supervisor polls of in-flight executor tasks.",
        ),
    ] = 5.0


class LaunchSupervisorParameters(GenericActuatorParameters):
    """Actuator configuration parameters for launch supervision.

    Inherit in an actuator class to add these parameters if you
    want to use LaunchSupervisor"""

    model_config = ConfigDict(extra="allow")

    launchSchedulingGraceSeconds: Annotated[
        float,
        pydantic.Field(gt=0),
    ] = 600.0

    launchTimeoutSeconds: Annotated[
        float,
        pydantic.Field(gt=0),
    ] = 900.0

    launchSupervisorPollIntervalSeconds: Annotated[
        float,
        pydantic.Field(gt=0),
    ] = 5.0

    def to_supervisor_config(self) -> LaunchSupervisorConfig:
        """Build a supervisor config from actuator parameters."""
        return LaunchSupervisorConfig(
            launchSchedulingGraceSeconds=self.launchSchedulingGraceSeconds,
            launchTimeoutSeconds=self.launchTimeoutSeconds,
            launchSupervisorPollIntervalSeconds=self.launchSupervisorPollIntervalSeconds,
        )


@dataclass
class _PendingLaunch:
    """An in-flight executor Ray task registered with the supervisor."""

    request: MeasurementRequest
    executor_ref: ray.ObjectRef
    submitted_at: float
    seen_running: bool = False


@dataclass
class _SupervisorState:
    """Mutable supervisor state guarded by a lock."""

    pending: dict[str, _PendingLaunch] = field(default_factory=dict)
    completed_request_ids: set[str] = field(default_factory=set)


def _default_task_state_lookup(executor_ref: ray.ObjectRef) -> str | None:
    """Return Ray task state for an executor ObjectRef, or None if unavailable."""
    try:
        task_id = executor_ref.task_id()
    except (AttributeError, RuntimeError, ValueError):
        return None

    try:
        from ray.util.state import list_tasks

        tasks = list_tasks(
            filters=[("task_id", "=", task_id)],
            limit=1,
            raise_on_missing_output=False,
        )
    except Exception:
        return None

    if not tasks:
        return None
    return getattr(tasks[0], "state", None) or tasks[0].get("state")


def build_launch_failure_measurements(
    request: MeasurementRequest,
    reason: str,
) -> MeasurementRequest:
    """Attach per-entity InvalidMeasurementResult values for a launch failure."""
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


class LaunchSupervisor:
    """Monitors Ray executor ObjectRefs and emits Invalid results on launch failure.

    Intended for reuse by any actuator that fire-and-forgets Ray measurement tasks.
    """

    def __init__(
        self,
        queue: MeasurementQueue,
        config: LaunchSupervisorConfig,
        logger: logging.Logger | None = None,
        *,
        task_state_lookup: Callable[[ray.ObjectRef], str | None] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Initialize the supervisor.

        Args:
            queue: Measurement queue for launch-failure invalid results.
            config: Launch supervision timeouts and poll interval.
            logger: Optional logger; defaults to module logger.
            task_state_lookup: Injectable Ray task state lookup (for tests).
            monotonic: Injectable monotonic clock (for tests).
            sleep: Injectable sleep (for tests).
        """
        self._queue = queue
        self._config = config
        self._log = logger or logging.getLogger(__name__)
        self._task_state_lookup = task_state_lookup or _default_task_state_lookup
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
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
            name="LaunchSupervisor",
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
        """Register an executor task for launch supervision."""
        with self._lock:
            if request.requestid in self._state.completed_request_ids:
                return
            self._state.pending[request.requestid] = _PendingLaunch(
                request=request,
                executor_ref=executor_ref,
                submitted_at=self._monotonic(),
            )

    def mark_completed(self, requestid: str) -> None:
        """Record that a requestid has a result (avoids duplicate queue puts)."""
        with self._lock:
            self._state.completed_request_ids.add(requestid)
            self._state.pending.pop(requestid, None)

    def _run_loop(self) -> None:
        """Poll pending executor tasks until stopped."""
        while not self._stop.is_set():
            self._poll_once()
            self._sleep(self._config.launchSupervisorPollIntervalSeconds)

    def _poll_once(self) -> None:
        """Run a single supervision pass over pending launches."""
        with self._lock:
            pending_snapshot = list(self._state.pending.values())

        for pending in pending_snapshot:
            self._check_pending(pending)

    def _check_pending(self, pending: _PendingLaunch) -> None:
        """Evaluate one pending executor task."""
        requestid = pending.request.requestid
        with self._lock:
            if requestid in self._state.completed_request_ids:
                self._state.pending.pop(requestid, None)
                return

        ready_refs, _ = ray.wait([pending.executor_ref], timeout=0)
        if ready_refs:
            self._handle_ready(pending)
            return

        elapsed = self._monotonic() - pending.submitted_at
        task_state = self._task_state_lookup(pending.executor_ref)
        if task_state == "RUNNING":
            with self._lock:
                if requestid in self._state.pending:
                    self._state.pending[requestid].seen_running = True
            return

        if (
            task_state == "FAILED"
            and elapsed >= self._config.launchSchedulingGraceSeconds
        ):
            self._emit_launch_failure(
                pending,
                reason=(
                    "Measurement task failed before completion "
                    f"(Ray state={task_state})"
                ),
            )
            return

        if elapsed >= self._config.launchTimeoutSeconds and task_state != "RUNNING":
            self._emit_launch_failure(
                pending,
                reason=(
                    "Measurement task did not start within "
                    f"{int(self._config.launchTimeoutSeconds)}s "
                    "(scheduling/runtime_env)"
                ),
            )

    def _handle_ready(self, pending: _PendingLaunch) -> None:
        """Executor finished; surface errors and unregister."""
        requestid = pending.request.requestid
        try:
            ray.get(pending.executor_ref)
        except Exception as error:
            self._log.warning(
                "Executor task for request %s raised: %s",
                requestid,
                error,
            )
        finally:
            with self._lock:
                self._state.completed_request_ids.add(requestid)
                self._state.pending.pop(requestid, None)

    def _emit_launch_failure(self, pending: _PendingLaunch, reason: str) -> None:
        """Queue an invalid measurement for a launch/scheduling failure."""
        ready_refs, _ = ray.wait([pending.executor_ref], timeout=0)
        if ready_refs:
            self._handle_ready(pending)
            return

        requestid = pending.request.requestid
        with self._lock:
            if requestid in self._state.completed_request_ids:
                self._state.pending.pop(requestid, None)
                return
            self._state.completed_request_ids.add(requestid)

        failed_request = build_launch_failure_measurements(
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
        self._cancel_executor(pending.executor_ref)
        with self._lock:
            self._state.pending.pop(requestid, None)

    def _cancel_executor(self, executor_ref: ray.ObjectRef) -> None:
        """Best-effort cancellation of a stuck executor task."""
        try:
            ray.cancel(executor_ref, force=True, recursive=True)
        except (TypeError, ValueError, RuntimeError) as error:
            self._log.debug("Could not cancel executor ref: %s", error)
