# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest

import ado.modules.operators.randomwalk  # noqa: F401 — loads operator plugins
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.resource import OperationProvenanceInfo
from ado.modules.operators._explore_orchestration import (
    _check_and_extract_discovery_space,
    explore_operation_provenance,
)
from ado.modules.operators.collections import explore


class _Space(DiscoverySpace):
    """Discovery space stand-in that skips full construction."""

    def __init__(self) -> None:
        """Skip parent initialization."""


def test_single_discovery_space_returns_the_only_space() -> None:
    """A one-space input dict yields that space, under any parameter name."""
    space = _Space()
    assert _check_and_extract_discovery_space({"measuredSpace": space}) is space


def test_single_discovery_space_rejects_zero_or_many_spaces() -> None:
    """Explore operations must be given exactly one discovery space."""
    space = _Space()
    with pytest.raises(ValueError, match="found 0"):
        _check_and_extract_discovery_space({})
    with pytest.raises(ValueError, match="found 2"):
        _check_and_extract_discovery_space({"a": space, "b": _Space()})


def test_explore_provenance_empty_when_no_spaces() -> None:
    """Explore provenance with no spaces records operator provenance only."""
    metadata = explore.operators["random_walk"]
    provenance = explore_operation_provenance(metadata, [])

    assert isinstance(provenance, OperationProvenanceInfo)
    assert set(provenance.operators) == {metadata.operatorIdentifier}
    assert provenance.experiments == []
    assert provenance.actuators == {}


def test_explore_provenance_from_space_is_deduplicated(
    pfas_space: DiscoverySpace,
) -> None:
    """The same experiment on more than one input space is recorded once."""
    metadata = explore.operators["random_walk"].model_copy(update={"provenance": None})
    provenance = explore_operation_provenance(metadata, [pfas_space, pfas_space])

    assert provenance.operators == {}
    assert len(provenance.experiments) == 1
    assert provenance.experiments[0].actuatorIdentifier == "replay"
    assert (
        provenance.experiments[0].experimentIdentifier
        == "transformer-toxicity-inference-experiment"
    )
    assert set(provenance.actuators) == {"replay"}
    assert provenance.actuators["replay"].distributionName == "ado-core"
