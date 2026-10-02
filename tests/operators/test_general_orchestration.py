# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import inspect

import pytest

import ado.modules.operators.randomwalk  # noqa: F401 — loads operator plugins
from ado.modules.operators._general_orchestration import _operator_callable_for_harness
from ado.modules.operators._orchestrate_core import operator_provenance_mapping
from ado.modules.operators.collections import characterize, explore


def test_operator_provenance_mapping_uses_package_provenance() -> None:
    """Registered operators record their package under the operator identifier."""
    metadata = explore.operators["random_walk"]
    mapping = operator_provenance_mapping(metadata)

    assert set(mapping) == {metadata.operatorIdentifier}
    assert mapping[metadata.operatorIdentifier].distributionName == "ado-core"


def test_operator_provenance_mapping_empty_when_unset() -> None:
    """Operators with no package provenance contribute an empty mapping."""
    metadata = explore.operators["random_walk"].model_copy(update={"provenance": None})
    assert operator_provenance_mapping(metadata) == {}


@pytest.mark.parametrize(
    "operator_name",
    ["profile"],
)
def test_operator_callable_for_harness_unwraps_decorated_operator(
    operator_name: str,
) -> None:
    """Decorated operators register a wrapper; harness must call the implementation."""
    registered = characterize.operators[operator_name].function
    assert registered is not None
    resolved = _operator_callable_for_harness(registered)
    assert resolved is inspect.unwrap(registered)
