# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for explore-operation orchestration waiting on the operator."""

import pytest

from ado.core.operation.operation import OperationOutput
from ado.modules.operators._explore_orchestration import _await_explore_operation


def test_await_explore_operation_still_gets_result_when_live_updates_fail() -> None:
    """Live-console failures must not skip collecting the operator result."""
    expected = OperationOutput()
    get_result_calls = {"count": 0}

    def live_updates() -> None:
        raise SystemError(
            "Unable to get the measurement requests for operation op-1. "
            "Error: (pymysql.err.OperationalError) (2013, "
            "'Lost connection to MySQL server during query')"
        )

    def get_result() -> OperationOutput:
        get_result_calls["count"] += 1
        return expected

    result = _await_explore_operation(
        get_result=get_result,
        live_updates=live_updates,
    )

    assert result is expected
    assert get_result_calls["count"] == 1


def test_await_explore_operation_propagates_operator_errors() -> None:
    """Failures from the operator itself must still propagate."""

    def live_updates() -> None:
        return None

    def get_result() -> OperationOutput:
        raise RuntimeError("operator failed")

    with pytest.raises(RuntimeError, match="operator failed"):
        _await_explore_operation(
            get_result=get_result,
            live_updates=live_updates,
        )
