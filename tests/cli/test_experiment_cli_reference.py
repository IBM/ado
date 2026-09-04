# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import warnings

import pytest
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.modules.actuators.errors import ExperimentVersionMismatchError
from ado.modules.actuators.registry import ActuatorRegistry
from ado.schema.reference import ExperimentReference
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


def test_get_experiment_unknown_actuator_handles_error_gracefully() -> None:
    """Get experiment with an unknown actuator exits with code 1 and error message."""
    runner = CliRunner()
    result = runner.invoke(
        ado, ["get", "experiment", "nonexistent_actuator.some_experiment"]
    )
    assert result.exit_code == 1
    assert (
        "ERROR:  No actuator called nonexistent_actuator has been added to the registry"
        in result.output
    )


def test_describe_experiment_unknown_actuator_handles_error_gracefully() -> None:
    """Describe experiment with an unknown actuator exits with code 1 and error message."""
    runner = CliRunner()
    result = runner.invoke(
        ado, ["describe", "experiment", "nonexistent_actuator.some_experiment"]
    )
    assert result.exit_code == 1
    assert (
        "ERROR:  No actuator called nonexistent_actuator has been added to the registry"
        in result.output
    )


def test_experimentForReference_bare_versioned_wrong_version(
    global_registry: ActuatorRegistry,
) -> None:
    """Version suffix must not be silently dropped when looking up by reference.

    Uses a patch-level mismatch (1.0.1 vs catalog 1.0.0) so that the major-version
    lookup succeeds but the fully-qualified check then raises ExperimentVersionMismatchError.
    """
    catalog = global_registry.catalogForActuatorIdentifier("mock")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(
            _make_experiment("bare_versioned_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "mock"}
            )
        )

    reference = ExperimentReference(
        actuatorIdentifier="mock",
        experimentIdentifier="bare_versioned_exp",
        experimentVersion="1.0.1",
    )
    with pytest.raises(ExperimentVersionMismatchError):
        global_registry.experimentForReference(
            reference,
            match_on="fully_qualified_version",
            resolve=True,
        )


def test_experimentForReference_bare_versioned_correct_version(
    global_registry: ActuatorRegistry,
) -> None:
    """The correct version is returned when the reference version matches the catalog."""
    catalog = global_registry.catalogForActuatorIdentifier("mock")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(
            _make_experiment("bare_versioned_correct_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "mock"}
            )
        )

    reference = ExperimentReference(
        actuatorIdentifier="mock",
        experimentIdentifier="bare_versioned_correct_exp",
        experimentVersion="1.0.0",
    )
    result = global_registry.experimentForReference(
        reference,
        match_on="fully_qualified_version",
    )
    assert result.identifier == "bare_versioned_correct_exp"
    assert result.version == "1.0.0"
