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


_SUPERVISOR_RAY_STATE_NAMES = frozenset({"RUNNING", "FAILED"})


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

    Ray's State API exposes many lifecycle states; the supervisor only needs to
    distinguish running, failed, and everything else (pending, finished, unknown,
    or lookup failure).
    """

    RUNNING = "RUNNING"
    FAILED = "FAILED"
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
            logger: Logger for non-running/non-failed values; defaults to module logger.

        Returns:
            ``RUNNING``, ``FAILED``, or ``OTHER`` (includes unavailable lookup).
        """
        if raw == cls.RUNNING.value:
            return cls.RUNNING
        if raw == cls.FAILED.value:
            return cls.FAILED
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
    """True once Ray State API has reported RUNNING for this task."""


@dataclass
class _SupervisorState:
    """Mutable supervisor state guarded by a lock."""

    pending: dict[str, _PendingLaunch] = field(default_factory=dict)
    completed_request_ids: set[str] = field(default_factory=set)


def _default_task_state_lookup(executor_ref: ray.ObjectRef) -> RayTaskState:
    """Return collapsed supervisor state (``RUNNING`` / ``FAILED`` / ``OTHER``).

    Uses ``ray.util.state.list_tasks``. Any non-``RUNNING``/``FAILED`` Ray value,
    lookup failure, or missing task maps to ``OTHER``.
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


def notify_launch_supervisor_completed(
    notifier: object | None,
    requestid: str,
) -> None:
    """Notify launch supervision that a measurement result was queued.

    ``notifier`` is usually the hosting actuator (in-process or as a Ray actor
    handle). It must expose ``mark_launch_completed``; ``LaunchSupervisor`` may
    be passed directly with ``mark_completed``.

    Args:
        notifier: Actuator or supervisor to notify, or None to skip.
        requestid: Measurement request identifier that now has a queued result.
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
    ) -> None:
        """Initialize the supervisor.

        Args:
            queue: Measurement queue for launch-failure invalid results.
            config: Launch supervision timeouts and poll interval.
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
                submitted_at=time.monotonic(),
            )

    def mark_completed(self, requestid: str) -> None:
        """Record that a requestid has a queued result (avoids duplicate invalid puts).

        Called from the executor path via :func:`notify_launch_supervisor_completed`
        as soon as the measurement queue receives a result, before the Ray executor
        ref becomes ready.
        """
        with self._lock:
            self._state.completed_request_ids.add(requestid)
            self._state.pending.pop(requestid, None)

    def _run_loop(self) -> None:
        """Poll pending executor tasks until stopped."""
        while not self._stop.is_set():
            self._poll_once()
            time.sleep(self._config.launchSupervisorPollIntervalSeconds)

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

        elapsed = time.monotonic() - pending.submitted_at
        task_state = _default_task_state_lookup(pending.executor_ref)
        if task_state == RayTaskState.RUNNING:
            with self._lock:
                if requestid in self._state.pending:
                    self._state.pending[requestid].seen_running = True
            return

        if (
            task_state == RayTaskState.FAILED
            and elapsed >= self._config.launchSchedulingGraceSeconds
        ):
            self._emit_launch_failure(
                pending,
                reason=(
                    "Measurement task failed before completion "
                    f"(Ray state={task_state.value})"
                ),
            )
            return

        if elapsed >= self._config.launchTimeoutSeconds and not pending.seen_running:
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
