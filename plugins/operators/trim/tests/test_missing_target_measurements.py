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


# ---------------------------------------------------------------------------
# NoPriorsParameters round-trip
# ---------------------------------------------------------------------------


def test_no_priors_parameters_round_trip() -> None:
    """NoPriorsParameters round-trips correctly; missingTargetVariables is in the dump."""
    original = NoPriorsParameters(targetOutput="latency")
    dumped = original.model_dump()
    assert "missingTargetVariables" in dumped
    restored = NoPriorsParameters.model_validate(dumped)
    assert restored.targetOutput == "latency"
    # Default policy is RaiseError
    assert restored.missingTargetVariables.mode == MissingTargetMode.RaiseError


def test_no_priors_parameters_missing_target_variables_is_a_kwarg() -> None:
    """missingTargetVariables can be set via the NoPriorsParameters constructor."""
    params = NoPriorsParameters(
        targetOutput="latency",
        missingTargetVariables=MissingTargetMeasurements(mode=MissingTargetMode.Skip),
    )
    assert params.missingTargetVariables.mode == MissingTargetMode.Skip


def test_no_priors_parameters_missing_target_variables_survives_round_trip() -> None:
    """missingTargetVariables set at runtime survives model_dump / model_validate."""
    params = NoPriorsParameters(targetOutput="latency")
    policy = MissingTargetMeasurements(mode=MissingTargetMode.Skip, budget=2)
    params.missingTargetVariables = policy
    assert params.missingTargetVariables.mode == MissingTargetMode.Skip
    assert params.missingTargetVariables.budget == 2
    # The value is now in the dump so it round-trips correctly.
    dumped = params.model_dump()
    assert dumped["missingTargetVariables"]["mode"] == MissingTargetMode.Skip
    restored = NoPriorsParameters.model_validate(dumped)
    assert restored.missingTargetVariables.mode == MissingTargetMode.Skip
    assert restored.missingTargetVariables.budget == 2


# ---------------------------------------------------------------------------
# TrimParameters round-trip and propagation validator
# ---------------------------------------------------------------------------


def test_trim_parameters_round_trip() -> None:
    """TrimParameters with missingTargetVariables round-trips correctly."""
    original = TrimParameters(
        targetOutput="throughput",
        missingTargetVariables=MissingTargetMeasurements(
            mode=MissingTargetMode.InjectDefaultValue, defaultValue=0.0
        ),
    )
    dumped = original.model_dump()
    restored = TrimParameters.model_validate(dumped)
    assert restored.missingTargetVariables.mode == MissingTargetMode.InjectDefaultValue
    assert restored.missingTargetVariables.defaultValue == 0.0


def test_trim_parameters_propagates_missingTargetVariables() -> None:
    """TrimParameters.propagate_missingTargetVariables copies the policy into noPriorParameters."""
    params = TrimParameters(
        targetOutput="throughput",
        missingTargetVariables=MissingTargetMeasurements(
            mode=MissingTargetMode.Skip, budget=4
        ),
    )
    assert (
        params.noPriorParameters.missingTargetVariables.mode == MissingTargetMode.Skip
    )
    assert params.noPriorParameters.missingTargetVariables.budget == 4


def test_trim_parameters_no_default_for_unmeasured_properties() -> None:
    """TrimParameters no longer has a defaultForUnmeasuredProperties field."""
    params = TrimParameters(targetOutput="x")
    assert not hasattr(params, "defaultForUnmeasuredProperties")
