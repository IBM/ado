# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from ado.cli.core.cli import app


@pytest.mark.timeout(900)
def test_pkh_ideal_gas_example(tmp_path: Path) -> None:
    """PKH measures six of eight gas configurations and predicts the other two."""
    result = CliRunner().invoke(
        app,
        [
            "create",
            "operation",
            "-f",
            "plugins/operators/active_learning/examples/ideal_gas/operation_pkh.yaml",
            "--with",
            "space=tests/resources/trim/space_minimal.yaml",
            "--set",
            "operation.parameters.numberEntities=6",
            "--set",
            f"operation.parameters.outputDirectory={tmp_path / 'model'}",
        ],
    )
    assert result.exit_code == 0, result.output

    directory = tmp_path / "model_finalized"
    metadata = json.loads((directory / "metadata.json").read_text())
    predictions = pd.read_csv(directory / "combined_predictions.csv")
    assert metadata["measuredCount"] == 6
    assert f"pkhMmdSquared: {metadata['mmdSquared']}" in result.output
    assert len(predictions) == 8
    assert predictions["is_predicted"].sum() == 2
    assert predictions["pressure"].notna().all()
    assert (directory / "model.pkl").is_file()
