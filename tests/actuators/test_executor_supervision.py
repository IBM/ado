# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Unit tests for measurement executor supervision helpers (no Ray cluster)."""

from __future__ import annotations

from orchestrator.modules.actuators.executor_supervisor import (
    ExperimentExecutorSupervisorParameters,
    RayTaskState,
    _ray_api_task_state_names,
    add_invalid_measurement_results,
)
from orchestrator.schema.entity import Entity
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from orchestrator.schema.result import InvalidMeasurementResult


def _sample_request(requestid: str = "abc123") -> MeasurementRequest:
    entity = Entity(
        identifier="ent-1",
        constitutive_property_values=(),
        generatorid="test",
    )
    return MeasurementRequest(
        operation_id="op-1",
        requestIndex=0,
        experimentReference=ExperimentReference(
            actuatorIdentifier="custom_experiments",
            experimentIdentifier="exp",
        ),
        entities=[entity],
        requestid=requestid,
    )


def test_build_launch_failure_measurements() -> None:
    """Launch failure builds invalid results for every entity."""
    request = _sample_request()
    failed = add_invalid_measurement_results(request, reason="timeout")
    assert failed.status == MeasurementRequestStateEnum.FAILED
    assert failed.measurements is not None
    assert len(failed.measurements) == 1
    assert isinstance(failed.measurements[0], InvalidMeasurementResult)
    assert failed.measurements[0].reason == "timeout"


def test_ray_api_includes_supervisor_states() -> None:
    """Ray State API literal must still include RUNNING and FAILED."""
    api_states = _ray_api_task_state_names()
    assert "RUNNING" in api_states
    assert "FAILED" in api_states


def test_ray_task_state_from_ray_state_collapses() -> None:
    """Ray state strings collapse to RUNNING, FAILED, or OTHER."""
    assert RayTaskState.from_ray_state("RUNNING") == RayTaskState.RUNNING
    assert RayTaskState.from_ray_state("FAILED") == RayTaskState.FAILED
    assert RayTaskState.from_ray_state(None) == RayTaskState.OTHER
    assert RayTaskState.from_ray_state("PENDING_NODE_ASSIGNMENT") == RayTaskState.OTHER


def test_launch_supervisor_parameters_to_config() -> None:
    """Actuator parameters map to supervisor config."""
    params = ExperimentExecutorSupervisorParameters(
        taskFailedGraceSeconds=120.0,
        taskRunningTimeoutSeconds=300.0,
        supervisorPollIntervalSeconds=2.0,
    )
    config = params.to_supervisor_config()
    assert config.taskFailedGraceSeconds == 120.0
    assert config.taskRunningTimeoutSeconds == 300.0
    assert config.supervisorPollIntervalSeconds == 2.0
