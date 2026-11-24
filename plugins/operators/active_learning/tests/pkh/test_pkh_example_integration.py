# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import re
import sys
from contextlib import chdir
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner, Result

from ado.cli.core.cli import app
from ado.core.datacontainer.resource import DataContainerResource


def _run_example(tmp_path: Path, budget: int) -> Result:
    """Run the PKH example on the eight-point ideal-gas space."""
    example = Path(
        "plugins/operators/active_learning/examples/ideal_gas/operation_pkh.yaml"
    ).resolve()
    space = Path("tests/resources/trim/space_minimal.yaml").resolve()
    with chdir(tmp_path):
        return CliRunner().invoke(
            app,
            [
                "--override-ado-app-dir",
                str(tmp_path),
                "create",
                "operation",
                "-f",
                str(example),
                "--with",
                f"space={space}",
                "--set",
                f"operation.parameters.numberEntities={budget}",
            ],
        )


@pytest.mark.timeout(900)
def test_pkh_ideal_gas_example(tmp_path: Path) -> None:
    """Retrieve the completed table by PKH operation ID and check the actual measurements."""
    budget = 8 if sys.version_info >= (3, 14) else 6
    result = _run_example(tmp_path, budget=budget)
    assert result.exit_code == 0, result.output
    assert "pkhMmdSquared:" in result.output
    operation_match = re.search(r"operation-pkh@[\w.]+-[a-f0-9]{8}", result.output)
    assert operation_match, result.output
    cli_args = ["--override-ado-app-dir", str(tmp_path)]
    runner = CliRunner()
    related = runner.invoke(
        app,
        [
            *cli_args,
            "show",
            "related",
            "operation",
            operation_match[0],
            "--max-hops",
            "1",
        ],
    )
    assert related.exit_code == 0, related.output
    [container_id] = re.findall(r"datacontainer-[a-f0-9]{8}", related.stdout)
    assert "random_walk@" in related.stdout
    stored = runner.invoke(
        app, [*cli_args, "get", "datacontainer", container_id, "-o", "json"]
    )
    assert stored.exit_code == 0, stored.output
    container = DataContainerResource.model_validate_json(stored.stdout)
    predictions = container.config.tabularData["predictions"].dataframe()
    assert len(predictions) == 8
    assert predictions.entityIdentifier.is_unique
    assert predictions.is_predicted.sum() == 8 - budget
    assert predictions.pressure.notna().all()
    observations = runner.invoke(
        app,
        [
            *cli_args,
            "show",
            "measurements",
            "space",
            "--use-latest",
            "--property",
            "pressure",
            "-o",
            "csv",
        ],
    )
    assert observations.exit_code == 0, observations.output
    measured = pd.read_csv(StringIO(observations.stdout)).set_index("identifier")
    measured_rows = predictions.loc[~predictions.is_predicted].set_index(
        "entityIdentifier"
    )
    assert measured_rows.pressure.to_dict() == pytest.approx(
        measured.pressure.to_dict()
    )
    assert not (tmp_path / "pkh_models_finalized").exists()
    assert not (tmp_path / "AutogluonModels").exists()


@pytest.mark.timeout(900)
def test_pkh_failed_run_has_no_dataset(tmp_path: Path) -> None:
    """An infeasible measurement request leaves the data-container catalog empty."""
    result = _run_example(tmp_path, budget=100)
    assert result.exit_code != 0, result.output
    assert "Requested number of entities to sample (100)" in result.output
    containers = CliRunner().invoke(
        app,
        ["--override-ado-app-dir", str(tmp_path), "get", "datacontainer", "-o", "name"],
    )
    assert containers.exit_code == 0, containers.output
    assert "datacontainer-" not in containers.stdout
    assert not (tmp_path / "pkh_models_finalized").exists()
