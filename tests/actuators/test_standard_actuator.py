# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for StandardActuator and its module-level helper functions."""

import functools
import sys
from collections.abc import Callable
from typing import Any

import cloudpickle
import pytest

from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.modules.actuators.catalog import ExperimentCatalog
from orchestrator.modules.actuators.errors import (
    DeprecatedExperimentError,
    UnknownExperimentError,
)
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue, NullQueue
from orchestrator.modules.actuators.standard import (
    StandardActuator,
)
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.point import SpacePoint
from orchestrator.schema.property import (
    AbstractPropertyDescriptor,
    ConstitutiveProperty,
)
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from orchestrator.schema.result import ValidMeasurementResult
from orchestrator.utilities.support import (
    compute_measurement_status,
    create_measurement_result,
    dict_to_measurements,
)

# ---------------------------------------------------------------------------
# Module-level constants shared by concrete test actuators
# ---------------------------------------------------------------------------

_ACTUATOR_ID = "test_standard"
_EXPERIMENT_ID = "double_x"

_TEST_EXPERIMENT = Experiment(
    actuatorIdentifier=_ACTUATOR_ID,
    identifier=_EXPERIMENT_ID,
    requiredProperties=(
        ConstitutiveProperty(
            identifier="x",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
                domainRange=[0.0, 100.0],
            ),
        ),
    ),
    targetProperties=[AbstractPropertyDescriptor(identifier="result")],
)
_TEST_CATALOG = ExperimentCatalog(experiments={_EXPERIMENT_ID: _TEST_EXPERIMENT})
_TEST_REF = ExperimentReference(
    actuatorIdentifier=_ACTUATOR_ID,
    experimentIdentifier=_EXPERIMENT_ID,
)

_DEPRECATED_EXPERIMENT_ID = "deprecated_double"
_DEPRECATED_EXPERIMENT = Experiment(
    actuatorIdentifier=_ACTUATOR_ID,
    identifier=_DEPRECATED_EXPERIMENT_ID,
    deprecated=True,
    requiredProperties=_TEST_EXPERIMENT.requiredProperties,
    targetProperties=_TEST_EXPERIMENT.targetProperties,
)
_DEPRECATED_CATALOG = ExperimentCatalog(
    experiments={_DEPRECATED_EXPERIMENT_ID: _DEPRECATED_EXPERIMENT}
)
_DEPRECATED_REF = ExperimentReference(
    actuatorIdentifier=_ACTUATOR_ID,
    experimentIdentifier=_DEPRECATED_EXPERIMENT_ID,
)


# ---------------------------------------------------------------------------
# Module-level experiment functions (must be picklable for Ray)
# ---------------------------------------------------------------------------


def _double_x(x: float) -> dict[str, Any]:
    """Doubles the input x — used as the experiment function in tests."""
    return {"result": x * 2.0}


def _custom_execute(request: MeasurementRequest) -> MeasurementRequest:
    """Picklable module-level function used by _CustomActuator._get_request_executor.

    Returns a fixed result of 99.0 regardless of entity property values.
    """
    results = [
        create_measurement_result(
            entity.identifier,
            dict_to_measurements({"result": 99.0}, _TEST_EXPERIMENT),
            request.experimentReference,
        )
        for entity in request.entities
    ]
    request.measurements = results
    request.status = compute_measurement_status(results)
    return request


# ---------------------------------------------------------------------------
# Concrete StandardActuator subclasses for testing
# ---------------------------------------------------------------------------


class _DoubleXActuator(StandardActuator):
    """Simple-path actuator: uses _experiment_implementations."""

    identifier = _ACTUATOR_ID

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> ExperimentCatalog:
        """Return the test catalog."""
        return _TEST_CATALOG

    def _experiment_implementations(self) -> dict[str, Callable[..., dict[str, Any]]]:
        """Return the double_x experiment implementation."""
        return {_EXPERIMENT_ID: _double_x}


class _DeprecatedExperimentActuator(StandardActuator):
    """Catalog contains only a deprecated experiment."""

    identifier = _ACTUATOR_ID

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> ExperimentCatalog:
        """Return the catalog with a deprecated experiment."""
        return _DEPRECATED_CATALOG

    def _experiment_implementations(self) -> dict[str, Callable[..., dict[str, Any]]]:
        """Unused — resolution raises DeprecatedExperimentError first."""
        return {_DEPRECATED_EXPERIMENT_ID: _double_x}


class _EmptyImplementationsActuator(StandardActuator):
    """Catalog lists double_x but no Python implementation is registered."""

    identifier = _ACTUATOR_ID

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> ExperimentCatalog:
        """Return the test catalog."""
        return _TEST_CATALOG

    def _experiment_implementations(self) -> dict[str, Callable[..., dict[str, Any]]]:
        """Deliberately omit double_x."""
        return {}


class _CustomActuator(StandardActuator):
    """Custom-path actuator: overrides _get_request_executor to track calls."""

    identifier = _ACTUATOR_ID

    def __init__(
        self,
        queue: MeasurementQueue | NullQueue | None = None,
        params: dict | None = None,
    ) -> None:
        """Initialise with a call counter for side-effect tracking."""
        super().__init__(queue=queue, params=params)
        self.build_fn_calls = 0

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> ExperimentCatalog:
        """Return the test catalog."""
        return _TEST_CATALOG

    def _get_request_executor(
        self,
        request: MeasurementRequest,
        use_ray: bool = False,
    ) -> Callable[[], MeasurementRequest]:
        """Increment call counter (side effect) and return a picklable callable."""
        self.build_fn_calls += 1
        return functools.partial(_custom_execute, request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity(x: float = 3.0) -> Any:  # noqa: ANN401
    """Create a test entity with constitutive property x."""
    return SpacePoint.model_validate({"entity": {"x": x}}).to_entity()


# Register this module for pickle-by-value so that test-defined functions
# (_double_x, _custom_execute, etc.) are serialised by bytecode when passed
# to Ray remote tasks.  Without this, cloudpickle would try to import
# tests.actuators.test_standard_actuator on the Ray worker, which fails
# because the tests directory is not installed.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


# ---------------------------------------------------------------------------
# Tests — simple path via _experiment_implementations
# ---------------------------------------------------------------------------


def test_execute_returns_measurement_request() -> None:
    """execute() returns a completed MeasurementRequest with status SUCCESS."""
    actuator = _DoubleXActuator()
    result = actuator.execute(
        entities=[_entity(3.0)],
        experimentReference=_TEST_REF,
        requesterid="test-op",
        requestIndex=0,
    )

    assert isinstance(result, MeasurementRequest)
    assert result.status == MeasurementRequestStateEnum.SUCCESS
    assert result.measurements is not None
    assert len(result.measurements) == 1
    assert isinstance(result.measurements[0], ValidMeasurementResult)


def test_execute_produces_correct_measured_value() -> None:
    """execute() passes constitutive property values to fn and maps results correctly."""
    actuator = _DoubleXActuator()
    result = actuator.execute(
        entities=[_entity(5.0)],
        experimentReference=_TEST_REF,
        requesterid="test-op",
        requestIndex=0,
    )

    assert result.status == MeasurementRequestStateEnum.SUCCESS
    observed = result.measurements[0].measurements
    assert len(observed) == 1
    assert observed[0].property.targetProperty.identifier == "result"
    assert observed[0].value == pytest.approx(10.0)


def test_execute_multi_entity() -> None:
    """execute() handles a batch of entities, producing one result per entity."""
    actuator = _DoubleXActuator()
    entities = [_entity(float(i)) for i in range(3)]
    result = actuator.execute(
        entities=entities,
        experimentReference=_TEST_REF,
        requesterid="test-op",
        requestIndex=0,
    )

    assert result.status == MeasurementRequestStateEnum.SUCCESS
    assert len(result.measurements) == 3
    ids_in_result = {m.entityIdentifier for m in result.measurements}
    assert ids_in_result == {e.identifier for e in entities}


def test_execute_no_queue_required() -> None:
    """execute() works when no MeasurementQueue is provided (NullQueue is used)."""
    actuator = _DoubleXActuator()  # no queue argument → NullQueue
    result = actuator.execute(
        entities=[_entity()],
        experimentReference=_TEST_REF,
        requesterid="test-op",
        requestIndex=0,
    )

    assert result.status == MeasurementRequestStateEnum.SUCCESS


def test_execute_with_use_ray() -> None:
    """execute(use_ray=True) produces the correct result via parallel Ray tasks."""
    actuator = _DoubleXActuator()
    result = actuator.execute(
        entities=[_entity(7.0)],
        experimentReference=_TEST_REF,
        requesterid="test-op",
        requestIndex=0,
        use_ray=True,
    )

    assert result.status == MeasurementRequestStateEnum.SUCCESS
    observed = result.measurements[0].measurements
    assert observed[0].value == pytest.approx(14.0)


def test_execute_unknown_experiment_raises() -> None:
    """execute() raises UnknownExperimentError when the reference is absent from the catalog."""
    actuator = _DoubleXActuator()
    missing_ref = ExperimentReference(
        actuatorIdentifier=_ACTUATOR_ID,
        experimentIdentifier="nonexistent_experiment",
    )
    with pytest.raises(UnknownExperimentError, match="No experiment matching"):
        actuator.execute(
            entities=[_entity()],
            experimentReference=missing_ref,
            requesterid="test-op",
            requestIndex=0,
        )


def test_execute_deprecated_experiment_raises_deprecated_error() -> None:
    """execute() raises DeprecatedExperimentError when the experiment is deprecated."""
    actuator = _DeprecatedExperimentActuator()
    with pytest.raises(DeprecatedExperimentError, match="deprecated"):
        actuator.execute(
            entities=[_entity()],
            experimentReference=_DEPRECATED_REF,
            requesterid="test-op",
            requestIndex=0,
        )


def test_execute_missing_implementation_raises_key_error() -> None:
    """execute() raises KeyError when the catalog lists an experiment without code."""
    actuator = _EmptyImplementationsActuator()
    with pytest.raises(KeyError, match="No implementation"):
        actuator.execute(
            entities=[_entity()],
            experimentReference=_TEST_REF,
            requesterid="test-op",
            requestIndex=0,
        )


def test_submit_dispatches_to_queue() -> None:
    """submit() returns the request ID and eventually puts the result on the queue."""
    queue = MeasurementQueue()
    actuator = _DoubleXActuator(queue=queue)
    entity = _entity(4.0)

    request_ids = actuator.submit(
        entities=[entity],
        experimentReference=_TEST_REF,
        requesterid="test-op",
        requestIndex=0,
    )

    assert len(request_ids) == 1

    # Block until the async Ray task completes and the result arrives on the queue
    result: MeasurementRequest = queue.get(timeout=30)

    assert isinstance(result, MeasurementRequest)
    assert result.requestid == request_ids[0]
    assert result.status == MeasurementRequestStateEnum.SUCCESS
    assert len(result.measurements) == 1


# ---------------------------------------------------------------------------
# Tests — custom path via _get_request_executor
# ---------------------------------------------------------------------------


def test_custom_build_execute_fn_is_called() -> None:
    """_get_request_executor is invoked once per execute() call."""
    actuator = _CustomActuator()

    assert actuator.build_fn_calls == 0
    actuator.execute(
        entities=[_entity()],
        experimentReference=_TEST_REF,
        requesterid="test-op",
        requestIndex=0,
    )
    assert actuator.build_fn_calls == 1


def test_custom_build_fn_side_effect_before_dispatch() -> None:
    """_get_request_executor side effects run synchronously before the Ray task is launched."""
    queue = MeasurementQueue()
    actuator = _CustomActuator(queue=queue)

    actuator.submit(
        entities=[_entity()],
        experimentReference=_TEST_REF,
        requesterid="test-op",
        requestIndex=0,
    )

    # Counter must already be incremented before submit() returns
    assert actuator.build_fn_calls == 1

    # Drain the queue so the background Ray task does not linger
    queue.get(timeout=30)


def test_custom_execute_returns_correct_value() -> None:
    """Custom _get_request_executor result (99.0) is reflected in the MeasurementRequest."""
    actuator = _CustomActuator()
    result = actuator.execute(
        entities=[_entity()],
        experimentReference=_TEST_REF,
        requesterid="test-op",
        requestIndex=0,
    )

    assert result.status == MeasurementRequestStateEnum.SUCCESS
    observed = result.measurements[0].measurements
    assert observed[0].value == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Tests — NullQueue
# ---------------------------------------------------------------------------


def test_null_queue_put_is_silent() -> None:
    """NullQueue.put() accepts items without error and discards them."""
    q = NullQueue()
    q.put("anything", block=True, timeout=None)
    q.put_nowait("anything_else")


def test_null_queue_ray_namespace_returns_none() -> None:
    """NullQueue.ray_namespace() returns None."""
    assert NullQueue().ray_namespace() is None


# ---------------------------------------------------------------------------
# Lifecycle test — MeasurementRequest serialisation round-trip
# ---------------------------------------------------------------------------


def test_measurement_request_lifecycle() -> None:
    """MeasurementRequest can be serialised and recreated from its dump."""
    actuator = _DoubleXActuator()
    original = actuator.execute(
        entities=[_entity(2.5)],
        experimentReference=_TEST_REF,
        requesterid="test-op",
        requestIndex=7,
    )

    dumped = original.model_dump()
    restored = MeasurementRequest.model_validate(dumped)

    assert restored.requestid == original.requestid
    assert restored.status == original.status
    assert restored.operation_id == original.operation_id
    assert restored.requestIndex == original.requestIndex
    assert len(restored.measurements) == len(original.measurements)
    assert (
        restored.measurements[0].entityIdentifier
        == original.measurements[0].entityIdentifier
    )
    # Verify observed value survives the round-trip
    orig_val = original.measurements[0].measurements[0].value
    rest_val = restored.measurements[0].measurements[0].value
    assert rest_val == pytest.approx(orig_val)
