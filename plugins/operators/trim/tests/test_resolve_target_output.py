# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
from trim.operator import validate_targetOutput
from trim.trim_pydantic import TrimParameters

from ado.modules.actuators.custom_experiments import custom_experiment
from ado.schema.measurementspace import MeasurementSpace, MeasurementSpaceConfiguration


# ---------------------------------------------------------------------------
# Register a minimal custom experiment once for this module.
# The decorator adds the experiment to the global custom_experiments catalog.
# ---------------------------------------------------------------------------
@custom_experiment(output_property_identifiers=["foo"])
def foo_experiment(alpha: float) -> dict:
    """Minimal custom experiment for testing _resolve_target_output."""
    return {"foo": alpha * 2.0}


# The Experiment object built by the decorator
_EXP = foo_experiment._experiment

# Build a real MeasurementSpace directly from the experiment's observed properties
# (no actuator registry / catalog lookup required).
_MS = MeasurementSpace(
    configuration=MeasurementSpaceConfiguration(
        observedProperties=_EXP.observedProperties,
        experiments=[_EXP],
    )
)

# Derive the expected identifiers once so tests stay DRY.
_OBSERVED_ID = _MS.observedProperties[0].identifier  # "foo_experiment-foo"
_BARE_ID = _MS.observedProperties[0].targetProperty.identifier  # "foo"


def _make_space(num_exp_refs: int = 1) -> SimpleNamespace:
    """Return a minimal discoverySpace stub with the given number of experimentReferences."""
    return SimpleNamespace(
        measurementSpace=SimpleNamespace(
            experimentReferences=list(_MS.experimentReferences) * num_exp_refs,
            observedProperties=_MS.observedProperties,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resolve_bare_target_identifier() -> None:
    """Bare target property identifier 'foo' is accepted and kept as-is."""
    params = TrimParameters(targetOutput=_BARE_ID)
    resolved = validate_targetOutput(params, _make_space())

    assert resolved.targetOutput == _BARE_ID
    assert resolved.noPriorParameters.targetOutput == _BARE_ID


def test_resolve_fully_qualified_observed_identifier() -> None:
    """Fully-qualified observed property identifier is rewritten to the bare form."""
    params = TrimParameters(targetOutput=_OBSERVED_ID)
    resolved = validate_targetOutput(params, _make_space())

    assert resolved.targetOutput == _BARE_ID
    assert resolved.noPriorParameters.targetOutput == _BARE_ID


def test_unrecognised_target_raises_value_error() -> None:
    """An unrecognised targetOutput name raises ValueError with helpful message."""
    params = TrimParameters(targetOutput="not_foo")
    with pytest.raises(ValueError, match="not_foo"):
        validate_targetOutput(params, _make_space())


def test_multiple_experiments_raises_value_error() -> None:
    """A space with more than 1 experiment raises ValueError."""
    params = TrimParameters(targetOutput=_BARE_ID)
    with pytest.raises(ValueError, match="exactly 1 experiment"):
        validate_targetOutput(params, _make_space(num_exp_refs=2))


def test_zero_experiments_raises_value_error() -> None:
    """A space with 0 experiments raises ValueError."""
    params = TrimParameters(targetOutput=_BARE_ID)
    with pytest.raises(ValueError, match="exactly 1 experiment"):
        validate_targetOutput(params, _make_space(num_exp_refs=0))
