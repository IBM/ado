# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Unit tests for _resolve_target_output in trim/operator.py."""

from types import SimpleNamespace

import pytest
from trim.operator import _resolve_target_output
from trim.trim_pydantic import TrimParameters


def _make_observed_property(experiment_id: str, target_id: str) -> SimpleNamespace:
    """Minimal observed-property stub with the two fields _resolve_target_output uses."""
    return SimpleNamespace(
        identifier=f"{experiment_id}-{target_id}",
        targetProperty=SimpleNamespace(identifier=target_id),
    )


def _make_space(*observed_properties: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        measurementSpace=SimpleNamespace(observedProperties=list(observed_properties))
    )


def _make_params(target_output: str) -> TrimParameters:
    return TrimParameters(targetOutput=target_output)


# ---------------------------------------------------------------------------
# Already fully-qualified
# ---------------------------------------------------------------------------


def test_already_qualified_accepted() -> None:
    """Fully-qualified observed property identifier passes through unchanged."""
    op = _make_observed_property("exp_a", "pressure")
    space = _make_space(op)
    params = _make_params("exp_a-pressure")

    result = _resolve_target_output(params, space)  # type: ignore[arg-type]

    assert result.targetOutput == "exp_a-pressure"
    assert result.noPriorParameters.targetOutput == "exp_a-pressure"


# ---------------------------------------------------------------------------
# Bare name — unambiguous auto-resolve
# ---------------------------------------------------------------------------


def test_bare_name_single_match_resolved() -> None:
    """Bare target property identifier is resolved when exactly one experiment matches."""
    op = _make_observed_property("calculate_pressure_ideal_gas", "pressure")
    space = _make_space(op)
    params = _make_params("pressure")

    result = _resolve_target_output(params, space)  # type: ignore[arg-type]

    assert result.targetOutput == "calculate_pressure_ideal_gas-pressure"
    assert (
        result.noPriorParameters.targetOutput == "calculate_pressure_ideal_gas-pressure"
    )


def test_bare_name_resolution_updates_no_priors_params() -> None:
    """noPriorParameters.targetOutput is kept in sync after resolution."""
    op = _make_observed_property("exp_x", "latency")
    space = _make_space(op)
    params = _make_params("latency")

    result = _resolve_target_output(params, space)  # type: ignore[arg-type]

    assert result.noPriorParameters.targetOutput == result.targetOutput


# ---------------------------------------------------------------------------
# Bare name — not found
# ---------------------------------------------------------------------------


def test_bare_name_not_in_space_raises() -> None:
    """Bare name that matches no observed property raises ValueError."""
    op = _make_observed_property("exp_a", "throughput")
    space = _make_space(op)
    params = _make_params("latency")

    with pytest.raises(ValueError, match="does not match any observed property"):
        _resolve_target_output(params, space)  # type: ignore[arg-type]


def test_error_lists_valid_identifiers() -> None:
    """The ValueError lists the valid observed property identifiers."""
    op = _make_observed_property("exp_a", "throughput")
    space = _make_space(op)
    params = _make_params("latency")

    with pytest.raises(ValueError, match="exp_a-throughput"):
        _resolve_target_output(params, space)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bare name — ambiguous (multiple experiments, same target)
# ---------------------------------------------------------------------------


def test_ambiguous_bare_name_raises() -> None:
    """Bare name matching multiple observed properties raises ValueError."""
    op1 = _make_observed_property("exp_a", "pressure")
    op2 = _make_observed_property("exp_b", "pressure")
    space = _make_space(op1, op2)
    params = _make_params("pressure")

    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_target_output(params, space)  # type: ignore[arg-type]


def test_ambiguous_error_lists_candidates() -> None:
    """The ambiguity ValueError lists both candidate identifiers."""
    op1 = _make_observed_property("exp_a", "pressure")
    op2 = _make_observed_property("exp_b", "pressure")
    space = _make_space(op1, op2)
    params = _make_params("pressure")

    with pytest.raises(ValueError, match="exp_a-pressure"):
        _resolve_target_output(params, space)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fully-qualified identifier that does NOT exist in the space is rejected
# ---------------------------------------------------------------------------


def test_qualified_identifier_not_in_space_raises() -> None:
    """A fully-qualified identifier that isn't in the space raises ValueError."""
    op = _make_observed_property("exp_a", "pressure")
    space = _make_space(op)
    params = _make_params("exp_b-pressure")  # looks qualified but isn't in space

    with pytest.raises(ValueError, match="does not match any observed property"):
        _resolve_target_output(params, space)  # type: ignore[arg-type]
