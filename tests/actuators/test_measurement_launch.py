# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for measurement launch supervision."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
import ray

from orchestrator.modules.actuators import measurement_launch
from orchestrator.modules.actuators.measurement_launch import (
    LaunchSupervisor,
    LaunchSupervisorConfig,
    LaunchSupervisorParameters,
    RayTaskState,
    build_launch_failure_measurements,
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
    failed = build_launch_failure_measurements(request, reason="timeout")
    assert failed.status == MeasurementRequestStateEnum.FAILED
    assert failed.measurements is not None
    assert len(failed.measurements) == 1
    assert isinstance(failed.measurements[0], InvalidMeasurementResult)
    assert failed.measurements[0].reason == "timeout"


def test_ray_api_includes_supervisor_states() -> None:
    """Ray State API literal must still include RUNNING and FAILED."""
    api_states = measurement_launch._ray_api_task_state_names()
    assert "RUNNING" in api_states
    assert "FAILED" in api_states


def test_verify_supervisor_ray_states_supported_rejects_missing_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verification fails if Ray removes RUNNING from its task state schema."""
    monkeypatch.setattr(
        measurement_launch,
        "_ray_api_task_state_names",
        lambda: frozenset({"FAILED", "PENDING_ARGS_AVAIL"}),
    )
    with pytest.raises(RuntimeError, match="RUNNING"):
        measurement_launch._verify_supervisor_ray_states_supported()


def test_ray_task_state_from_ray_state_collapses() -> None:
    """Ray state strings collapse to RUNNING, FAILED, or OTHER."""
    assert RayTaskState.from_ray_state("RUNNING") == RayTaskState.RUNNING
    assert RayTaskState.from_ray_state("FAILED") == RayTaskState.FAILED
    assert RayTaskState.from_ray_state(None) == RayTaskState.OTHER
    assert RayTaskState.from_ray_state("PENDING_NODE_ASSIGNMENT") == RayTaskState.OTHER


def test_supervisor_failed_state_after_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """FAILED Ray task state triggers invalid after scheduling grace."""
    queued: list[MeasurementRequest] = []

    class MockQueue:
        def put(self, item: MeasurementRequest, block: bool = False) -> None:
            queued.append(item)

    config = LaunchSupervisorConfig(
        launchSchedulingGraceSeconds=0.5,
        launchTimeoutSeconds=900.0,
        launchSupervisorPollIntervalSeconds=0.1,
    )
    monkeypatch.setattr(
        measurement_launch,
        "_default_task_state_lookup",
        lambda _ref: RayTaskState.FAILED,
    )
    monkeypatch.setattr(ray, "wait", MagicMock(return_value=([], [])))
    monkeypatch.setattr(ray, "cancel", MagicMock())

    supervisor = LaunchSupervisor(
        queue=MockQueue(),  # type: ignore[arg-type]
        config=config,
    )
    pending_ref = MagicMock(spec=ray.ObjectRef)
    supervisor.start()
    try:
        supervisor.register(_sample_request("fail1"), pending_ref)
        time.sleep(0.8)
        assert len(queued) == 1
        assert "failed before completion" in queued[0].measurements[0].reason  # type: ignore[index, union-attr]
    finally:
        supervisor.stop()


def test_launch_supervisor_parameters_to_config() -> None:
    """Actuator parameters map to supervisor config."""
    params = LaunchSupervisorParameters(
        launchSchedulingGraceSeconds=120.0,
        launchTimeoutSeconds=300.0,
        launchSupervisorPollIntervalSeconds=2.0,
    )
    config = params.to_supervisor_config()
    assert config.launchSchedulingGraceSeconds == 120.0
    assert config.launchTimeoutSeconds == 300.0
    assert config.launchSupervisorPollIntervalSeconds == 2.0
