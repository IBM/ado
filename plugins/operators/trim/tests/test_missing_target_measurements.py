# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest
from pydantic import ValidationError
from trim.samplers.no_priors_parameters import (
    MissingTargetMeasurements,
    MissingTargetMode,
    NoPriorsParameters,
)
from trim.trim_pydantic import TrimParameters

# ---------------------------------------------------------------------------
# MissingTargetMeasurements — construction and validation
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Default mode is RaiseError with no budget or defaultValue."""
    m = MissingTargetMeasurements()
    assert m.mode == MissingTargetMode.RaiseError
    assert m.budget is None
    assert m.defaultValue is None
    assert m.skip_entities == []
    assert m.no_target_variable_entities == []


def test_budget_zero_rejected() -> None:
    """budget=0 must raise a ValidationError."""
    with pytest.raises(ValidationError, match="budget must be > 0"):
        MissingTargetMeasurements(budget=0)


def test_budget_negative_rejected() -> None:
    """budget=-1 must raise a ValidationError."""
    with pytest.raises(ValidationError, match="budget must be > 0"):
        MissingTargetMeasurements(budget=-1)


def test_budget_positive_accepted() -> None:
    """budget=1 is valid."""
    m = MissingTargetMeasurements(mode=MissingTargetMode.Skip, budget=1)
    assert m.budget == 1


def test_inject_without_default_value_rejected() -> None:
    """mode=InjectDefaultValue without defaultValue must raise ValidationError."""
    with pytest.raises(ValidationError, match="defaultValue must be set"):
        MissingTargetMeasurements(mode=MissingTargetMode.InjectDefaultValue)


def test_inject_with_default_value_accepted() -> None:
    """mode=InjectDefaultValue with defaultValue=0.0 is valid."""
    m = MissingTargetMeasurements(
        mode=MissingTargetMode.InjectDefaultValue, defaultValue=0.0
    )
    assert m.mode == MissingTargetMode.InjectDefaultValue
    assert m.defaultValue == 0.0


def test_skip_mode_no_default_required() -> None:
    """mode=Skip does not require defaultValue."""
    m = MissingTargetMeasurements(mode=MissingTargetMode.Skip)
    assert m.mode == MissingTargetMode.Skip
    assert m.defaultValue is None


# ---------------------------------------------------------------------------
# MissingTargetMeasurements — round-trip serialisation
# ---------------------------------------------------------------------------


def test_round_trip_raise_error() -> None:
    """RaiseError round-trips through model_dump / model_validate."""
    original = MissingTargetMeasurements()
    dumped = original.model_dump()
    restored = MissingTargetMeasurements.model_validate(dumped)
    assert restored.mode == MissingTargetMode.RaiseError
    assert restored.budget is None
    assert restored.defaultValue is None


def test_round_trip_inject_default_value() -> None:
    """InjectDefaultValue round-trips correctly."""
    original = MissingTargetMeasurements(
        mode=MissingTargetMode.InjectDefaultValue, defaultValue=-99.0, budget=5
    )
    restored = MissingTargetMeasurements.model_validate(original.model_dump())
    assert restored.mode == MissingTargetMode.InjectDefaultValue
    assert restored.defaultValue == -99.0
    assert restored.budget == 5


def test_round_trip_skip() -> None:
    """Skip mode round-trips correctly."""
    original = MissingTargetMeasurements(mode=MissingTargetMode.Skip, budget=3)
    restored = MissingTargetMeasurements.model_validate(original.model_dump())
    assert restored.mode == MissingTargetMode.Skip
    assert restored.budget == 3


# ---------------------------------------------------------------------------
# Runtime-only fields absent from model_dump
# ---------------------------------------------------------------------------


def test_skip_entities_absent_from_dump() -> None:
    """skip_entities is a SkipJsonSchema field and must not appear in model_dump."""
    m = MissingTargetMeasurements()
    dumped = m.model_dump()
    assert "skip_entities" not in dumped


def test_no_target_variable_entities_absent_from_dump() -> None:
    """no_target_variable_entities must not appear in model_dump."""
    m = MissingTargetMeasurements()
    dumped = m.model_dump()
    assert "no_target_variable_entities" not in dumped


# ---------------------------------------------------------------------------
# NoPriorsParameters round-trip
# ---------------------------------------------------------------------------


def test_no_priors_parameters_round_trip() -> None:
    """NoPriorsParameters with missing_target_variables round-trips correctly."""
    original = NoPriorsParameters(
        targetOutput="latency",
        missing_target_variables=MissingTargetMeasurements(
            mode=MissingTargetMode.Skip, budget=2
        ),
    )
    dumped = original.model_dump()
    restored = NoPriorsParameters.model_validate(dumped)
    assert restored.targetOutput == "latency"
    assert restored.missing_target_variables.mode == MissingTargetMode.Skip
    assert restored.missing_target_variables.budget == 2


# ---------------------------------------------------------------------------
# TrimParameters round-trip and propagation validator
# ---------------------------------------------------------------------------


def test_trim_parameters_round_trip() -> None:
    """TrimParameters with missing_target_variables round-trips correctly."""
    original = TrimParameters(
        targetOutput="throughput",
        missing_target_variables=MissingTargetMeasurements(
            mode=MissingTargetMode.InjectDefaultValue, defaultValue=0.0
        ),
    )
    dumped = original.model_dump()
    restored = TrimParameters.model_validate(dumped)
    assert (
        restored.missing_target_variables.mode == MissingTargetMode.InjectDefaultValue
    )
    assert restored.missing_target_variables.defaultValue == 0.0


def test_trim_parameters_propagates_missing_target_variables() -> None:
    """TrimParameters.propagate_missing_target_variables copies the policy into noPriorParameters."""
    params = TrimParameters(
        targetOutput="throughput",
        missing_target_variables=MissingTargetMeasurements(
            mode=MissingTargetMode.Skip, budget=4
        ),
    )
    assert (
        params.noPriorParameters.missing_target_variables.mode == MissingTargetMode.Skip
    )
    assert params.noPriorParameters.missing_target_variables.budget == 4


def test_trim_parameters_no_default_for_unmeasured_properties() -> None:
    """TrimParameters no longer has a defaultForUnmeasuredProperties field."""
    params = TrimParameters(targetOutput="x")
    assert not hasattr(params, "defaultForUnmeasuredProperties")
