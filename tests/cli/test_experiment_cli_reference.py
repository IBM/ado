# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import warnings

import pytest
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.modules.actuators.catalog import ExperimentCatalog
from ado.modules.actuators.errors import ExperimentVersionMismatchError
from ado.modules.actuators.registry import ActuatorRegistry
from ado.schema.reference import ExperimentReference
from tests.schema.test_algorithm_versioning import (
    _make_experiment,
)


def test_describe_versioned_experiment_by_bare_name(
    test_actuator_catalog: ExperimentCatalog,
    global_registry: ActuatorRegistry,
) -> None:
    """Unversioned CLI args resolve via experiments_matching_identifier."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        test_actuator_catalog.addExperiment(
            _make_experiment("describe_cli_test_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "test"}
            )
        )

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "describe",
            "experiment",
            "test.describe_cli_test_exp",
        ],
    )
    assert result.exit_code == 0
    assert "describe_cli_test_exp" in result.output


def test_describe_versioned_experiment_with_version_suffix(
    test_actuator_catalog: ExperimentCatalog,
    global_registry: ActuatorRegistry,
) -> None:
    """Versioned CLI args resolve via experimentForReference."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        test_actuator_catalog.addExperiment(
            _make_experiment("describe_cli_test_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "test"}
            )
        )

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "describe",
            "experiment",
            "test.describe_cli_test_exp@1.0.0",
        ],
    )
    assert result.exit_code == 0
    assert "describe_cli_test_exp" in result.output


def test_describe_ambiguous_when_multiple_versions(
    test_actuator_catalog: ExperimentCatalog,
    global_registry: ActuatorRegistry,
) -> None:
    """Unversioned describe fails when multiple catalog versions exist."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        test_actuator_catalog.addExperiment(
            _make_experiment("describe_ambiguous_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "test"}
            )
        )
        test_actuator_catalog.addExperiment(
            _make_experiment("describe_ambiguous_exp", version="2.0.0").model_copy(
                update={"actuatorIdentifier": "test"}
            )
        )

    runner = CliRunner()
    result = runner.invoke(
        ado,
        ["describe", "experiment", "test.describe_ambiguous_exp"],
    )
    assert result.exit_code == 1
    assert "ambiguous" in result.output.lower()
    assert "1.0.0" in result.output
    assert "2.0.0" in result.output


def test_get_experiment_by_fully_qualified_resource_id() -> None:
    """Get experiment filters using consolidated resource id parsing."""
    runner = CliRunner()
    result = runner.invoke(
        ado, ["get", "experiments", "robotic_lab.peptide_mineralization"]
    )
    assert result.exit_code == 0
    assert "robotic_lab" in result.output
    assert "peptide_mineralization" in result.output


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
    test_actuator_catalog: ExperimentCatalog,
    global_registry: ActuatorRegistry,
) -> None:
    """Version suffix must not be silently dropped when looking up by reference.

    Uses a patch-level mismatch (1.0.1 vs catalog 1.0.0) so that the major-version
    lookup succeeds but the fully-qualified check then raises ExperimentVersionMismatchError.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        test_actuator_catalog.addExperiment(
            _make_experiment("bare_versioned_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "test"}
            )
        )

    reference = ExperimentReference(
        actuatorIdentifier="test",
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
    test_actuator_catalog: ExperimentCatalog,
    global_registry: ActuatorRegistry,
) -> None:
    """The correct version is returned when the reference version matches the catalog."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        test_actuator_catalog.addExperiment(
            _make_experiment("bare_versioned_correct_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "test"}
            )
        )

    reference = ExperimentReference(
        actuatorIdentifier="test",
        experimentIdentifier="bare_versioned_correct_exp",
        experimentVersion="1.0.0",
    )
    result = global_registry.experimentForReference(
        reference,
        match_on="fully_qualified_version",
    )
    assert result.identifier == "bare_versioned_correct_exp"
    assert result.version == "1.0.0"
