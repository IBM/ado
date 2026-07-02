# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import warnings

from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.modules.actuators.registry import ActuatorRegistry
from tests.schema.test_algorithm_versioning import _make_experiment


def test_describe_versioned_experiment_by_bare_name(
    global_registry: ActuatorRegistry,
) -> None:
    """Unversioned CLI args resolve via experiments_matching_identifier."""
    catalog = global_registry.catalogForActuatorIdentifier("mock")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(
            _make_experiment("describe_cli_test_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "mock"}
            )
        )

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "describe",
            "experiment",
            "mock.describe_cli_test_exp",
        ],
    )
    assert result.exit_code == 0
    assert "describe_cli_test_exp" in result.output


def test_describe_versioned_experiment_with_version_suffix(
    global_registry: ActuatorRegistry,
) -> None:
    """Versioned CLI args resolve via experimentForReference."""
    catalog = global_registry.catalogForActuatorIdentifier("mock")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(
            _make_experiment("describe_cli_test_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "mock"}
            )
        )

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "describe",
            "experiment",
            "mock.describe_cli_test_exp@1.0.0",
        ],
    )
    assert result.exit_code == 0
    assert "describe_cli_test_exp" in result.output


def test_describe_ambiguous_when_multiple_versions(
    global_registry: ActuatorRegistry,
) -> None:
    """Unversioned describe fails when multiple catalog versions exist."""
    catalog = global_registry.catalogForActuatorIdentifier("mock")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(
            _make_experiment("describe_ambiguous_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "mock"}
            )
        )
        catalog.addExperiment(
            _make_experiment("describe_ambiguous_exp", version="2.0.0").model_copy(
                update={"actuatorIdentifier": "mock"}
            )
        )

    runner = CliRunner()
    result = runner.invoke(
        ado,
        ["describe", "experiment", "mock.describe_ambiguous_exp"],
    )
    assert result.exit_code == 1
    assert "ambiguous" in result.output.lower()
    assert "1.0.0" in result.output
    assert "2.0.0" in result.output


def test_get_experiment_by_fully_qualified_resource_id() -> None:
    """Get experiment filters using consolidated resource id parsing."""
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "experiments", "mock.test-experiment"])
    assert result.exit_code == 0
    assert "mock" in result.output
    assert "test-experiment" in result.output
