# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Unit tests for NoPriorsSampleSelector._core_iterator_logic.

Focuses on the quota-counting behaviour under Skip mode, especially the
post-loop HIT path that was previously not incrementing quota_count.
"""

from types import SimpleNamespace

from trim.samplers.no_priors_parameters import (
    MissingTargetMeasurements,
    MissingTargetMode,
    NoPriorsParametersExtended,
)
from trim.samplers.no_priors_sampler import NoPriorsSampleSelector

from ado.schema.entity import Entity
from ado.schema.observed_property import ObservedProperty, ObservedPropertyValue
from ado.schema.property import AbstractPropertyDescriptor, ConstitutiveProperty
from ado.schema.property_value import ConstitutivePropertyValue
from ado.schema.reference import ExperimentReference
from ado.schema.result import ValidMeasurementResult

TARGET_OUTPUT = "exp-target"


def _make_entity(identifier: str, measured: bool) -> Entity:
    """Build a minimal entity; if *measured* is True attach a target measurement."""
    cp = ConstitutiveProperty(
        identifier="x",
        propertyDomain={"domainRange": [0, 100], "interval": 1.0},
    )
    cpv = ConstitutivePropertyValue(property=cp, value=float(identifier))
    entity = Entity(
        identifier=identifier,
        generatorid="test",
        constitutive_property_values=(cpv,),
    )
    if measured:
        exp_ref = ExperimentReference(
            experimentIdentifier="exp",
            actuatorIdentifier="custom_experiments",
        )
        op = ObservedProperty(
            targetProperty=AbstractPropertyDescriptor(identifier="target"),
            experimentReference=exp_ref,
        )
        opv = ObservedPropertyValue(property=op, value=1.0)
        entity.add_measurement_result(
            ValidMeasurementResult(
                entityIdentifier=entity.identifier, measurements=[opv]
            )
        )
    return entity


def _fake_space(entities: list[Entity]) -> SimpleNamespace:
    """Return a minimal space stub that resolves entities by identifier."""
    by_id = {e.identifier: e for e in entities}
    return SimpleNamespace(entity_for_point=lambda point: by_id[point])


def _make_sampler(quota: int, mode: MissingTargetMode) -> NoPriorsSampleSelector:
    params = NoPriorsParametersExtended(
        targetOutput=TARGET_OUTPUT,
        samples=quota,
        missingTargetVariables=MissingTargetMeasurements(mode=mode),
    )
    return NoPriorsSampleSelector(parameters=params)


def _run_iterator(
    sampler: NoPriorsSampleSelector,
    entities: list[Entity],
) -> list[str]:
    """Drive _core_iterator_logic synchronously and return yielded identifiers."""
    space = _fake_space(entities)
    yielded = []
    for batch in sampler._core_iterator_logic(space, entities):
        yielded.extend(e.identifier for e in batch)
    return yielded


# ---------------------------------------------------------------------------
# Regression test: last entity is a HIT that fills quota
# ---------------------------------------------------------------------------


def test_last_entity_hit_fills_quota_skip_mode() -> None:
    """Reproduces the production failure: pool exhausted with quota filled only
    by the very last entity's HIT, which was previously not counted.

    Sequence (12 entities, quota=10, Skip mode):
      idx  0-2:  HIT  (quota_count -> 3)
      idx  3:    MISS (skipped, quota_count stays 3)
      idx  4-9:  HIT  (quota_count -> 9)
      idx 10:    MISS (skipped, quota_count stays 9)
      idx 11:    HIT  (post-loop, quota_count -> 10)

    With the fix the warning must NOT fire and all 12 entities are yielded.
    """
    entities = []
    outcomes = [
        True,
        True,
        True,  # 0-2 HIT
        False,  # 3   MISS
        True,
        True,
        True,
        True,
        True,
        True,  # 4-9 HIT
        False,  # 10  MISS
        True,  # 11  HIT (the post-loop entity)
    ]
    for i, hit in enumerate(outcomes):
        entities.append(_make_entity(str(i), measured=hit))

    sampler = _make_sampler(quota=10, mode=MissingTargetMode.Skip)
    yielded = _run_iterator(sampler, entities)

    # All 12 entities must be yielded (pool drives until quota or exhaustion)
    assert len(yielded) == 12
    # Quota was filled — no shortfall warning should fire (quota_count == quota)
    assert sampler._missing_count == 2


def test_quota_reached_mid_pool_stops_early() -> None:
    """When quota is reached before the pool is exhausted, iteration stops early."""
    # 5 HITs in a row, quota=3 → stops after 4 yields (quota filled on check of 4th)
    entities = [_make_entity(str(i), measured=True) for i in range(10)]
    sampler = _make_sampler(quota=3, mode=MissingTargetMode.Skip)
    yielded = _run_iterator(sampler, entities)

    assert len(yielded) == 4  # yields 0,1,2,3 — checks 3 at top of iter 4 → break


def test_skip_mode_does_not_count_miss_toward_quota() -> None:
    """MISS entities are not counted toward quota under Skip mode."""
    # MISS, HIT, HIT, HIT — quota=3
    outcomes = [False, True, True, True]
    entities = [_make_entity(str(i), measured=hit) for i, hit in enumerate(outcomes)]
    sampler = _make_sampler(quota=3, mode=MissingTargetMode.Skip)
    yielded = _run_iterator(sampler, entities)

    # All 4 yielded: MISS at 0 doesn't count, so 3 HITs needed from remaining 3
    assert len(yielded) == 4
    assert sampler._missing_count == 1


def test_all_miss_pool_exhausted_warning() -> None:
    """If every entity misses the target under Skip mode, pool is exhausted at 0/quota."""
    entities = [_make_entity(str(i), measured=False) for i in range(5)]
    sampler = _make_sampler(quota=3, mode=MissingTargetMode.Skip)
    yielded = _run_iterator(sampler, entities)

    # All 5 yielded, quota never reached
    assert len(yielded) == 5
    assert sampler._missing_count == 5
