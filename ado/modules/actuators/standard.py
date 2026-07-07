# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""StandardActuator — a flexible base class for actuators.

Provides both synchronous (execute) and asynchronous (submit) execution paths
without requiring callers to manage Ray actors or MeasurementQueues.

Subclasses customise experiment logic through one of two hooks:

* _experiment_implementations() — simple path: return a mapping from experiment
  identifier to a callable ``fn(**kwargs) -> dict[str, Any]``.  The caller
  supplies constitutive property values as kwargs; the callee returns a dict
  whose keys are observed property identifiers.

* _get_request_executor() — custom path: override to return any zero-argument
  callable that runs the batch and returns a completed MeasurementRequest.
  Use functools.partial or a closure to capture any actuator state required.
"""

import functools
import logging
import uuid
from collections.abc import Callable
from typing import Any

import pydantic
import ray
from ray.actor import ActorHandle

from ado.modules.actuators.base import ActuatorBase
from ado.modules.actuators.executor_supervisor import (
    ExperimentExecutorSupervisor,
    ExperimentExecutorSupervisorParameters,
)
from ado.modules.actuators.measurement_queue import MeasurementQueue, NullQueue
from ado.schema.entity import Entity
from ado.schema.experiment import Experiment, ParameterizedExperiment
from ado.schema.reference import ExperimentReference
from ado.schema.request import MeasurementRequest
from ado.schema.result import MeasurementResult  # noqa: TC001
from ado.utilities.support import (
    compute_measurement_status,
    create_measurement_result,
    observed_property_values_from_dict,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helper functions — pure functions with no dependency on self.
# ---------------------------------------------------------------------------


def _execute_experiments_serial(
    request: MeasurementRequest,
    *,
    fn: Callable[..., dict[str, Any]],
    experiment: Experiment | ParameterizedExperiment,
) -> MeasurementRequest:
    """Run fn for each entity serially in the current process.

    For each entity, extract its constitutive property values, call fn(**values),
    convert the returned dict to ObservedPropertyValues, and build a
    MeasurementResult.  Exceptions per entity produce an InvalidMeasurementResult.

    Args:
        request: The request describing the experiment batch.
        fn: Callable ``(**kwargs) -> dict[str, Any]``.
        experiment: The  Experiment or ParameterizedExperiment being executed.

    Returns:
        The completed MeasurementRequest.
    """

    results: list[MeasurementResult] = []
    for entity in request.entities:
        try:
            input_values = experiment.propertyValuesFromEntity(entity)
            result_dict = fn(**input_values)
            values = observed_property_values_from_dict(result_dict, experiment)
            results.append(
                create_measurement_result(
                    entity.identifier, values, request.experimentReference
                )
            )
        except Exception as exc:  # noqa: BLE001, PERF203
            logger.warning(
                "Experiment %s failed for entity %s: %s",
                experiment.identifier,
                entity.identifier,
                exc,
            )
            results.append(
                create_measurement_result(
                    entity.identifier,
                    [],
                    request.experimentReference,
                    error=str(exc),
                )
            )
    request.measurements = results
    request.status = compute_measurement_status(results)
    return request


@ray.remote
def _ray_fn_runner(
    fn: Callable[..., dict[str, Any]], kwargs: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Run fn(**kwargs) as a Ray task.

    ray.remote(fn).remote(kwargs) would work but raises immediately on failure,
    aborting the entire batch.  This wrapper returns (result, error) tuples so
    _execute_experiments_parallel can handle per-entity failures gracefully.

    Returns:
        ``(result_dict, None)`` on success or ``(None, error_string)`` on failure.
    """
    try:
        return fn(**kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _execute_experiments_parallel(
    request: MeasurementRequest,
    *,
    fn: Callable[..., dict[str, Any]],
    experiment: Experiment | ParameterizedExperiment,
) -> MeasurementRequest:
    """Run fn for each entity in parallel using nested Ray tasks.

    Dispatches one Ray task per entity, waits for all to complete, then builds
    MeasurementResults and completes the request.

    Args:
        request: The request describing the experiment batch.
        fn: Callable ``(**kwargs) -> dict[str, Any]`` — must be picklable.
        experiment: The  Experiment or Parameterized experiment being executed.

    Returns:
        The completed MeasurementRequest.
    """

    futures = [
        _ray_fn_runner.remote(fn, experiment.propertyValuesFromEntity(entity))
        for entity in request.entities
    ]
    ray_results: list[tuple[dict[str, Any] | None, str | None]] = ray.get(futures)

    results: list[MeasurementResult] = []
    for entity, (result_dict, error) in zip(request.entities, ray_results, strict=True):
        if error is not None:
            logger.warning(
                "Experiment %s failed for entity %s: %s",
                experiment.identifier,
                entity.identifier,
                error,
            )
            results.append(
                create_measurement_result(
                    entity.identifier,
                    [],
                    request.experimentReference,
                    error=error,
                )
            )
        else:
            values = observed_property_values_from_dict(result_dict, experiment)
            results.append(
                create_measurement_result(
                    entity.identifier, values, request.experimentReference
                )
            )
    request.measurements = results
    request.status = compute_measurement_status(results)
    return request


@ray.remote
def _run_execute_fn(
    execute_fn: Callable[[], MeasurementRequest],
) -> MeasurementRequest:
    """Ray worker: call execute_fn() and return the completed MeasurementRequest.

    Dispatched only via Ray (``.remote()``); wraps arbitrary ``execute_fn`` such as
    ``functools.partial`` targets that cannot themselves be decorated with
    ``@ray.remote``.

    Args:
        execute_fn: Zero-argument callable returning a completed MeasurementRequest.

    Returns:
        The completed MeasurementRequest.
    """
    return execute_fn()


@ray.remote
def _enqueue_completed(
    execute_fn: Callable[[], MeasurementRequest],
    queue: MeasurementQueue | NullQueue,
    actuator_actor: ActorHandle["StandardActuator"] | None,
) -> None:
    """Ray worker: run execute_fn() and put the completed request on queue.

    This function is dispatched as a Ray remote task by StandardActuator.submit
    so that submission is non-blocking.

    Args:
        execute_fn: Zero-argument callable returning a completed MeasurementRequest.
        queue: The MeasurementQueue to which the result is written.
    """
    try:
        result = execute_fn()
        queue.put(result, block=False)
    except Exception:  # noqa: BLE001
        logger.exception(
            "_enqueue_completed: execute_fn or queue.put raised an exception"
        )
        raise
    else:
        if actuator_actor:
            actuator_actor.mark_measurement_request_completed.remote(result.requestid)


# ---------------------------------------------------------------------------
# StandardActuator
# ---------------------------------------------------------------------------


class StandardActuatorParameters(ExperimentExecutorSupervisorParameters):
    """Configuration parameters for the CustomExperiments actuator."""

    model_config = pydantic.ConfigDict(extra="forbid")


class StandardActuator(ActuatorBase):
    """Flexible actuator base class supporting synchronous and asynchronous execution.

    Provides two public execution methods:

    * ``execute()`` — runs experiments in the current process (or optionally via
      Ray) and returns the completed ``MeasurementRequest`` directly.  No Ray
      actors or ``MeasurementQueue`` are required.

    * ``submit()`` — dispatches a Ray task and returns immediately, placing the
      completed ``MeasurementRequest`` on the ``MeasurementQueue`` when done.
      Mirrors the ``ActuatorBase.submit`` contract.

    Subclasses must still implement ``catalog()``.  Experiment logic is
    provided through one of two hooks (see module docstring).
    """

    parameters_class = StandardActuatorParameters

    def __init__(
        self,
        queue: MeasurementQueue | NullQueue | None = None,
        params: dict | StandardActuatorParameters | None = None,
    ) -> None:
        """Initialise the actuator.

        Args:
            queue: MeasurementQueue for submit()-based async operation.
                   If None, a NullQueue is used — suitable for execute()-only use.
            params: Actuator configuration parameters.
        """

        # This does not convert params from dict or None
        super().__init__(
            queue=queue if queue is not None else NullQueue(), params=params
        )

        if self._parameters:
            self._parameters = self.parameters_class.model_validate(
                params, from_attributes=True
            )
        else:
            self._parameters = self.parameters_class()

        self._launch_supervisor = ExperimentExecutorSupervisor(
            queue=self._stateUpdateQueue,
            config=self._parameters.to_supervisor_config(),
            logger=self.log,
        )
        self._launch_supervisor.start()

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _experiment_implementations(self) -> dict[str, Callable[..., dict[str, Any]]]:
        """Return a mapping from experiment identifier to experiment callable.

        The keys are the identifiers of the Experiment instance in the catalog()

        The parameters of the callable for an experiment identifier
        must be the same as the constitutive property names of the
        related Experiment instance.
        The callable must return a dict mapping target property
        identifiers of the Experiment to their measured values.

        Returns:
            Dict mapping experiment identifier → callable.
        """
        return {}

    def _get_request_executor(
        self,
        request: MeasurementRequest,
        use_ray: bool = False,
    ) -> Callable[[], MeasurementRequest]:
        """Build a zero-argument callable that executes the request.

        The default implementation resolves the experiment callable via
        ``_experiment_implementations``, selects the serial or parallel executor
        based on ``use_ray``, and binds all arguments with ``functools.partial``.

        Override to implement custom execution strategies.  The returned callable:

        * Must accept no arguments and return a completed ``MeasurementRequest``
          (measurements and status set).
        * Must be picklable when ``use_ray=True`` or called from ``submit()``,
          because it is serialised and sent to a Ray task.  ``functools.partial``
          over a module-level function is the recommended pattern.

        Side effects performed *inside this method* (e.g. resource allocation,
        counter increments) run on the calling side — i.e. on the Ray actor for
        ``submit`` and in the current process for ``execute`` — before any Ray
        dispatch takes place.

        Args:
            request: The MeasurementRequest describing the experiment batch.
            use_ray: When True the default implementation selects the parallel
                     executor for nested Ray entity processing.

        Returns:
            A zero-argument callable returning a completed MeasurementRequest.

        Raises:
            KeyError: If no implementation exists for the requested experiment.
            UnknownExperimentError: If the experiment is not found in the catalog.
            DeprecatedExperimentError: If the experiment is deprecated.
        """
        fn, experiment = self._resolve_fn_and_experiment(request)
        executor = (
            _execute_experiments_parallel if use_ray else _execute_experiments_serial
        )
        return functools.partial(executor, request, fn=fn, experiment=experiment)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_fn_and_experiment(
        self,
        request: MeasurementRequest,
    ) -> tuple[Callable[..., dict[str, Any]], Experiment | ParameterizedExperiment]:
        """Look up the experiment object and its implementation callable.

        Args:
            request: The MeasurementRequest whose ``experimentReference`` is resolved.

        Returns:
            Tuple of ``(fn, experiment)`` where experiment is the resolved
            experiment based on the catalog (Experiment or ParameterizedExperiment)

        Raises:
            UnknownExperimentError: If the experiment is not found in the catalog.
            DeprecatedExperimentError: If the experiment is deprecated.
            KeyError: If no implementation exists for the experiment identifier.
        """
        experiment: Experiment | None = (
            type(self)
            .catalog()
            .experimentForReference(request.experimentReference, resolve=True)
        )

        implementations = self._experiment_implementations()
        experiment_id = experiment.identifier
        if experiment_id not in implementations:
            raise KeyError(
                f"No implementation for experiment {experiment_id!r}. "
                f"Available: {list(implementations)}"
            )

        return implementations[experiment_id], experiment

    def mark_measurement_request_completed(self, requestid: str) -> None:
        """Record that a measurement result was queued."""
        self._launch_supervisor.mark_measurement_request_completed(requestid)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
        use_ray: bool = False,
    ) -> MeasurementRequest:
        """Run the experiment and return the completed MeasurementRequest.

        No Ray actor or MeasurementQueue is required when ``use_ray=False``.

        Args:
            entities: Entities to measure.
            experimentReference: Reference to the experiment to run.
            requesterid: ID of the requesting operation.
            requestIndex: Index of this request within the operation.
            use_ray: When True, dispatch entity processing to nested Ray tasks.

        Returns:
            A completed MeasurementRequest with measurements and status set.
        """
        request = MeasurementRequest(
            operation_id=requesterid,
            requestIndex=requestIndex,
            experimentReference=experimentReference,
            entities=entities,
            requestid=str(uuid.uuid4())[:6],
        )
        execute_fn = self._get_request_executor(request, use_ray=use_ray)
        if not use_ray:
            return execute_fn()
        # When use_ray=True we always wrap execution in an outer Ray task. For the
        # default path this adds one hop before per-entity parallel tasks; that hop is
        # required so overrides of _get_request_executor may return an arbitrary
        # picklable zero-arg callable (including functools.partial), which Ray cannot
        # decorate directly at definition time.
        return ray.get(_run_execute_fn.remote(execute_fn))

    def submit(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
    ) -> list[str]:
        """Submit the experiment for asynchronous execution via Ray.

        Dispatches a Ray task that runs the experiment and puts the completed
        MeasurementRequest on the MeasurementQueue.  Returns immediately.

        Args:
            entities: Entities to measure.
            experimentReference: Reference to the experiment to run.
            requesterid: ID of the requesting operation.
            requestIndex: Index of this request within the operation.

        Returns:
            A list containing the single request ID.
        """
        request = MeasurementRequest(
            operation_id=requesterid,
            requestIndex=requestIndex,
            experimentReference=experimentReference,
            entities=entities,
            requestid=str(uuid.uuid4())[:6],
        )

        try:
            actuator_actor = ray.get_runtime_context().current_actor
        except Exception:
            actuator_actor = None

        execute_fn = self._get_request_executor(request, use_ray=True)
        executor_ref = _enqueue_completed.remote(
            execute_fn, self._stateUpdateQueue, actuator_actor
        )
        self._launch_supervisor.supervise_experiment_executor(request, executor_ref)
        return [request.requestid]
