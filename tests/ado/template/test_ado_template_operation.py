# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado


@pytest.mark.parametrize("operator_name", ["ray_tune", "random_walk"])
def test_template_operation(operator_name: str) -> None:
    runner = CliRunner()
    result = runner.invoke(
        ado, ["template", "operation", f"--operator-name={operator_name}"]
    )
    assert result.exit_code == 0, result.output
    assert operator_name in result.output


def test_template_operation_defaults_to_random_walk() -> None:
    runner = CliRunner()
    result = runner.invoke(ado, ["template", "operation"])
    assert result.exit_code == 0, result.output
    assert "operatorName: random_walk" in result.output
