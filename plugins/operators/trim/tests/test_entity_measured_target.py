# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Unit tests for entity_measured_target in missing_target_utils.

The key invariant: entity_measured_target must check whether the stored entity
has an ObservedPropertyValue whose *targetProperty* identifier matches the
requested target output.  It must NOT look up the target output as a key in
seriesRepresentation, because seriesRepresentation keys observed values by the
*observed property* identifier (format: "{experimentId}-{targetPropertyId}"),
not by the bare target property identifier.
"""

from types import SimpleNamespace

from trim.samplers.missing_target_utils import entity_measured_target

from orchestrator.schema.entity import Entity
from orchestrator.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from orchestrator.schema.property import (
    AbstractPropertyDescriptor,
    ConstitutiveProperty,
)
from orchestrator.schema.property_value import ConstitutivePropertyValue
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.result import ValidMeasurementResult


def _make_entity(pressure_value: float | None) -> Entity:
    """Build a minimal entity whose only constitutive property is 'temperature'.

    If *pressure_value* is not None the entity has an ObservedPropertyValue for
    the 'pressure' target property produced by experiment
    'calculate_pressure_ideal_gas' on actuator 'custom_experiments'.  This
    mirrors the real-world case in the integration test and exercises the
    "{experimentId}-{targetPropertyId}" observed-property identifier format.
    """
    cp = ConstitutiveProperty(
        identifier="temperature",
        propertyDomain={"domainRange": [270, 274], "interval": 2.0},
    )
    cpv = ConstitutivePropertyValue(property=cp, value=270.0)
    entity = Entity(
        identifier="temperature.270.0",
        generatorid="test",
        constitutive_property_values=(cpv,),
    )

    if pressure_value is not None:
        exp_ref = ExperimentReference(
            experimentIdentifier="calculate_pressure_ideal_gas",
            actuatorIdentifier="custom_experiments",
        )
        op = ObservedProperty(
            targetProperty=AbstractPropertyDescriptor(identifier="pressure"),
            experimentReference=exp_ref,
        )
        opv = ObservedPropertyValue(property=op, value=pressure_value)
        entity.add_measurement_result(
            ValidMeasurementResult(
                entityIdentifier=entity.identifier, measurements=[opv]
            )
        )

    return entity


# The fully-qualified observed property identifier for the ideal-gas experiment.
OBSERVED_ID = "calculate_pressure_ideal_gas-pressure"


def _fake_space(entity: Entity) -> SimpleNamespace:
    """Return a minimal discovery-space stub whose entity_for_point returns *entity*."""
    return SimpleNamespace(entity_for_point=lambda _point: entity)


# ---------------------------------------------------------------------------
# hit=True cases
# ---------------------------------------------------------------------------


def test_hit_when_pressure_measured() -> None:
    """Returns hit=True when the entity has a non-null measurement for OBSERVED_ID."""
    entity = _make_entity(pressure_value=101325.0)
    space = _fake_space(entity)

    hit, _ = entity_measured_target(entity, space, OBSERVED_ID)  # type: ignore[arg-type]

    assert hit is True


def test_series_returned_regardless_of_hit() -> None:
    """The series return value is always a pandas Series (used by callers)."""
    import pandas as pd

    entity = _make_entity(pressure_value=42.0)
    space = _fake_space(entity)

    _, series = entity_measured_target(entity, space, OBSERVED_ID)  # type: ignore[arg-type]

    assert isinstance(series, pd.Series)


# ---------------------------------------------------------------------------
# hit=False cases
# ---------------------------------------------------------------------------


def test_no_hit_when_no_measurement() -> None:
    """Returns hit=False when the entity has no observed property values at all."""
    entity = _make_entity(pressure_value=None)
    space = _fake_space(entity)

    hit, _ = entity_measured_target(entity, space, OBSERVED_ID)  # type: ignore[arg-type]

    assert hit is False


def test_no_hit_for_different_target() -> None:
    """Returns hit=False when queried for a different observed property identifier."""
    entity = _make_entity(pressure_value=101325.0)
    space = _fake_space(entity)

    hit, _ = entity_measured_target(entity, space, "exp_other-latency")  # type: ignore[arg-type]

    assert hit is False


# ---------------------------------------------------------------------------
# Regression: must match on observed property identifier, not target property id
# ---------------------------------------------------------------------------


def test_hit_requires_observed_property_identifier() -> None:
    """hit=True only when the fully-qualified observed property identifier is supplied.

    Callers (NoPriorsSampleSelector, TrimSampleSelector) always receive the
    resolved observed property identifier from _resolve_target_output, never the
    bare target property identifier.  This test guards against regressions where
    entity_measured_target incorrectly accepts the bare name.
    """
    entity = _make_entity(pressure_value=99999.0)
    space = _fake_space(entity)

    # Confirm the observed property identifier has the qualified form.
    assert len(entity.observedPropertyValues) == 1
    op_id = entity.observedPropertyValues[0].property.identifier
    assert op_id == OBSERVED_ID

    # Fully-qualified identifier → hit.
    hit_qualified, _ = entity_measured_target(entity, space, OBSERVED_ID)  # type: ignore[arg-type]
    assert hit_qualified is True

    # Bare target property identifier → no hit (callers never pass this).
    hit_bare, _ = entity_measured_target(entity, space, "pressure")  # type: ignore[arg-type]
    assert hit_bare is False
