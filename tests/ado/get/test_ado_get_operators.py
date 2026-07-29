# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado


@pytest.mark.parametrize("operator_name", ["ray_tune", "random_walk"])
def test_get_operator(operator_name: str) -> None:
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "operator", operator_name])
    assert result.exit_code == 0
    assert operator_name in result.output
