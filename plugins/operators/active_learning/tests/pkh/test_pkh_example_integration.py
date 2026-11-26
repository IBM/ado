# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Run the PKH example and inspect the resources published by the operator."""

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


def _ado_output(tmp_path: Path, *arguments: str) -> str:
    """Read a result from this test's ADO catalog."""
    result = CliRunner().invoke(
        app, ["--override-ado-app-dir", str(tmp_path), *arguments]
    )
    assert result.exit_code == 0, result.output
    return result.stdout


@pytest.mark.timeout(900)
def test_pkh_ideal_gas_example(tmp_path: Path) -> None:
    """PKH links its completed table and labels the rows measured by its walk."""
    budget = 8 if sys.version_info >= (3, 14) else 6
    result = _run_example(tmp_path, budget=budget)
    assert result.exit_code == 0, result.output
    assert "pkhMmdSquared:" in result.output
    [pkh_id] = set(re.findall(r"operation-pkh@[\w.]+-[a-f0-9]{8}", result.output))
    related = _ado_output(
        tmp_path, "show", "related", "operation", pkh_id, "--max-hops", "1"
    )
    [container_id] = re.findall(r"datacontainer-[a-f0-9]{8}", related)
    [walk_id] = re.findall(r"random_walk@[\w.-]+", related)
    stored = _ado_output(tmp_path, "get", "datacontainer", container_id, "-o", "json")
    container = DataContainerResource.model_validate_json(stored)
    predictions = (
        container.config.tabularData["predictions"]
        .dataframe()
        .set_index("entityIdentifier", verify_integrity=True)
    )
    assert len(predictions) == 8
    assert predictions.is_predicted.sum() == 8 - budget
    assert predictions.pressure.notna().all()
    observations = _ado_output(
        tmp_path, "show", "measurements", "operation", walk_id, "-o", "csv"
    )
    measured = pd.read_csv(StringIO(observations)).set_index("identifier")
    measured_rows = predictions.loc[~predictions.is_predicted, "pressure"]
    assert measured_rows.to_dict() == pytest.approx(measured.pressure.to_dict())


@pytest.mark.timeout(300)
def test_partial_prior_is_measured_by_the_requested_actuator(tmp_path: Path) -> None:
    """Measure the target when only a sibling output is available from its actuator."""
    created = CliRunner().invoke(
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
    assert created.exit_code == 0, created.output
    [space_id] = re.findall(r"Created space with identifier: (\S+)", created.output)
    space = DiscoverySpace.from_stored_configuration(
        project_context=AdoConfiguration.load(
            _override_config_dir=tmp_path
        ).project_context,
        space_identifier=space_id,
    )
    [entity] = candidate_entities(space)
    sibling, target = space.measurementSpace.observedProperties
    foreign_target = target.model_copy(
        update={
            "experimentReference": target.experimentReference.model_copy(
                update={"actuatorIdentifier": "other_lab"}
            )
        }
    )
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
        targetOutput=target.targetProperty.identifier, numberEntities=1
    )
    with chdir(tmp_path):
        walk, dataset = pkh(space, parameters=parameters).resources

    [measured_target] = [
        value.value
        for result in space.measurement_results_for_operation(walk.identifier)
        if isinstance(result, ValidMeasurementResult)
        for value in result.measurements
        if value.property == target
    ]
    assert 0 <= measured_target < 1
    table = dataset.config.tabularData["predictions"].dataframe()
    assert not table.is_predicted.item()
    assert table.adsorption_plateau_value.item() == measured_target
