# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import inspect

import pytest

import orchestrator.modules.operators.randomwalk  # noqa: F401 — loads operator plugins
from orchestrator.modules.operators._general_orchestration import (
    _operator_callable_for_harness,
)
from orchestrator.modules.operators.collections import characterize


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
