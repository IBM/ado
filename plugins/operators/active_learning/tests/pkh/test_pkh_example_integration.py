# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import re
import sys
from contextlib import chdir
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from active_learning.pkh.data import candidate_entities
from active_learning.pkh.operator import pkh
from active_learning.pkh.parameters import PKHParameters
from typer.testing import CliRunner, Result

from ado.cli.core.cli import app
from ado.cli.core.config import AdoConfiguration
from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.schema.observed_property import ObservedPropertyValue
from ado.schema.result import ValidMeasurementResult


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


@pytest.mark.timeout(300)
def test_partial_prior_is_measured_by_the_requested_actuator(tmp_path: Path) -> None:
    """Only the selected actuator's plateau measurement can complete the dataset."""
    result = CliRunner().invoke(
        app,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "create",
            "space",
            "-f",
            "plugins/actuators/example_actuator/yamls/discoveryspace.yaml",
            "--new-sample-store",
            "--set",
            "entitySpace[1].propertyDomain.values=[0.1]",
            "--set",
            "entitySpace[2].propertyDomain.values=[0.1]",
        ],
    )
    assert result.exit_code == 0, result.output
    [space_id] = re.findall(r"Created space with identifier: (\S+)", result.output)
    space = DiscoverySpace.from_stored_configuration(
        project_context=AdoConfiguration.load(
            _override_config_dir=tmp_path
        ).project_context,
        space_identifier=space_id,
    )
    [entity] = candidate_entities(space)
    sibling, target = space.measurementSpace.observedProperties
    other_lab = target.experimentReference.model_copy(
        update={"actuatorIdentifier": "other_lab"}
    )
    foreign_target = target.model_copy(update={"experimentReference": other_lab})
    space.sample_store.addEntities([entity])
    space.sample_store.add_measurement_results(
        [
            ValidMeasurementResult(
                entityIdentifier=entity.identifier,
                measurements=[ObservedPropertyValue(property=prop, value=value)],
            )
            for prop, value in [(sibling, [0.25]), (foreign_target, -1)]
        ],
        skip_relationship_to_request=True,
    )

    parameters = PKHParameters(
        targetOutput="adsorption_plateau_value", numberEntities=1
    )
    _, dataset = pkh(space, parameters=parameters).resources
    table = dataset.config.tabularData["predictions"].dataframe()
    assert not table.is_predicted.item()
    assert 0 <= table.adsorption_plateau_value.item() < 1
