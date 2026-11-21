# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from ado.cli.core.cli import app


@pytest.mark.timeout(900)
@pytest.mark.parametrize("operator_name", ["pkh", "flora"])
def test_active_learning_ideal_gas_example(tmp_path: Path, operator_name: str) -> None:
    """Each operator measures six gas configurations and predicts the other two."""
    runner = CliRunner()
    operation_file = Path(
        "plugins/operators/active_learning/examples/ideal_gas/operation_pkh.yaml"
    )
    target_override = []
    if operator_name == "flora":
        operation_file = tmp_path / "operation_flora.yaml"
        template = runner.invoke(
            app,
            [
                "template",
                "operation",
                "--operator-name",
                "flora",
                "--output-file",
                str(operation_file),
            ],
        )
        assert template.exit_code == 0, template.output
        target_override = ["--set", "operation.parameters.targetOutput=pressure"]
    result = runner.invoke(
        app,
        [
            "create",
            "operation",
            "-f",
            str(operation_file),
            "--with",
            "space=tests/resources/trim/space_minimal.yaml",
            "--set",
            "operation.parameters.numberEntities=6",
            "--set",
            f"operation.parameters.outputDirectory={tmp_path / 'model'}",
            *target_override,
        ],
    )
    assert result.exit_code == 0, result.output

    directory = tmp_path / "model_finalized"
    metadata = json.loads((directory / "metadata.json").read_text())
    predictions = pd.read_csv(directory / "combined_predictions.csv")
    assert metadata["measuredCount"] == 6
    if operator_name == "pkh":
        assert f"pkhMmdSquared: {metadata['mmdSquared']}" in result.output
    assert len(predictions) == 8
    assert predictions["is_predicted"].sum() == 2
    assert predictions["pressure"].notna().all()
    assert (directory / "model.pkl").is_file()
