# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner, Result

from ado.cli.core.cli import app


def _run_example(tmp_path: Path, budget: int) -> Result:
    """Run the PKH example on the eight-point ideal-gas space."""
    return CliRunner().invoke(
        app,
        [
            "create",
            "operation",
            "-f",
            "plugins/operators/active_learning/examples/ideal_gas/operation_pkh.yaml",
            "--with",
            "space=tests/resources/trim/space_minimal.yaml",
            "--set",
            f"operation.parameters.numberEntities={budget}",
            "--set",
            f"operation.parameters.outputDirectory={tmp_path / 'model'}",
        ],
    )


@pytest.mark.timeout(900)
def test_pkh_ideal_gas_example(tmp_path: Path) -> None:
    """PKH measures six of eight gas configurations and predicts the other two."""
    result = _run_example(tmp_path, budget=6)
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


@pytest.mark.timeout(900)
def test_pkh_failed_run_cleans_reserved_directory(tmp_path: Path) -> None:
    """PKH releases its output directory when the measurement run raises."""
    result = _run_example(tmp_path, budget=100)
    assert result.exit_code != 0, result.output
    assert "Requested number of entities to sample (100)" in result.output
    assert not (tmp_path / "model_finalized").exists()
