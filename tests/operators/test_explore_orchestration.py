# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import pytest

import ado.modules.operators.randomwalk  # noqa: F401 — loads operator plugins
from ado.core.operation.config import FunctionOperationInfo, GenericOperatorParameters
from ado.core.operation.resource import OperationProvenanceInfo
from ado.modules.operators._explore_orchestration import orchestrate_explore_operation
from ado.modules.operators.collections import explore
from ado.modules.operators.provenance import explore_operation_provenance


def test_orchestrate_explore_rejects_zero_discovery_spaces() -> None:
    """Explore operations require exactly one discovery space; zero raises ValueError."""
    metadata = explore.operators["random_walk"]
    with pytest.raises(ValueError, match="found 0"):
        orchestrate_explore_operation(
            operator_metadata=metadata,
            inputs={},
            parameters=GenericOperatorParameters(),
            operation_info=FunctionOperationInfo(),
        )


def test_orchestrate_explore_rejects_multiple_discovery_spaces(
    make_stub_discovery_space: Callable,
) -> None:
    """Explore operations require exactly one discovery space; two raises ValueError."""
    metadata = explore.operators["random_walk"]
    with pytest.raises(ValueError, match="found 2"):
        orchestrate_explore_operation(
            operator_metadata=metadata,
            inputs={"a": make_stub_discovery_space(), "b": make_stub_discovery_space()},
            parameters=GenericOperatorParameters(),
            operation_info=FunctionOperationInfo(),
        )


def test_explore_provenance_empty_when_no_spaces() -> None:
    """Explore provenance with no spaces records operator provenance only."""
    metadata = explore.operators["random_walk"]
    provenance = explore_operation_provenance(metadata, [])

    assert isinstance(provenance, OperationProvenanceInfo)
    assert set(provenance.operators) == {metadata.operatorIdentifier}
    assert provenance.experiments == []
    assert provenance.actuators == {}
