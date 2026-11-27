# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from ado.cli.core.cli import app


@pytest.mark.timeout(900)
def test_flora_titration_example(tmp_path: Path) -> None:
    """FLORA measures eight of sixteen titration points and predicts the rest."""
    result = CliRunner().invoke(
        app,
        [
            "create",
            "operation",
            "-f",
            "examples/titration/example_yamls/operation_flora.yaml",
            "--with",
            "space=tests/resources/titration/space_minimal.yaml",
            "--set",
            "operation.parameters.numberEntities=8",
            "--set",
            "operation.parameters.pilotSize=3",
            "--set",
            f"operation.parameters.outputDirectory={tmp_path / 'model'}",
        ],
    )
    assert result.exit_code == 0, result.output

    directory = tmp_path / "model_finalized"
    metadata = json.loads((directory / "metadata.json").read_text())
    predictions = pd.read_csv(directory / "combined_predictions.csv")
    assert metadata["measuredCount"] == 8
    assert len(predictions) == 16
    assert predictions["is_predicted"].sum() == 8
    assert predictions["ph"].notna().all()
    assert (directory / "model.pkl").is_file()
