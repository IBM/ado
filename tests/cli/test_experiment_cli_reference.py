# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import warnings

import pytest
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.modules.actuators.errors import ExperimentVersionMismatchError
from ado.modules.actuators.registry import ActuatorRegistry
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


def test_experiment_for_experiment_identifier_bare_versioned_wrong_version(
    global_registry: ActuatorRegistry,
) -> None:
    """Version suffix in a bare (no-actuator-prefix) identifier must not be silently dropped.

    Regression test for the bug where ``experiment_for_experiment_identifier``
    parsed the version out of ``exp@MAJOR.MINOR.PATCH`` but then discarded it
    when reconstructing the reference, so a wrong version never raised an error.

    Uses a patch-level mismatch (1.0.1 vs catalog 1.0.0) so that the major-version
    lookup succeeds but the fully-qualified check then raises ExperimentVersionMismatchError.
    Without the fix, the version is dropped and the lookup returns 1.0.0 silently.
    """
    catalog = global_registry.catalogForActuatorIdentifier("mock")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(
            _make_experiment("bare_versioned_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "mock"}
            )
        )

    # @1.0.1 shares major version 1 with the catalog's 1.0.0, so the major-version
    # lookup finds the experiment and the fully-qualified check raises the mismatch error.
    with pytest.raises(ExperimentVersionMismatchError):
        global_registry.experiment_for_experiment_identifier(
            "bare_versioned_exp@1.0.1",
            match_on="fully_qualified_version",
            resolve=True,
        )


def test_experiment_for_experiment_identifier_bare_versioned_correct_version(
    global_registry: ActuatorRegistry,
) -> None:
    """Version suffix in a bare (no-actuator-prefix) identifier must be forwarded to the lookup.

    The matching version should be returned when the version suffix is correct.
    """
    catalog = global_registry.catalogForActuatorIdentifier("mock")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(
            _make_experiment("bare_versioned_correct_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "mock"}
            )
        )

    result = global_registry.experiment_for_experiment_identifier(
        "bare_versioned_correct_exp@1.0.0",
        match_on="fully_qualified_version",
    )
    assert result.identifier == "bare_versioned_correct_exp"
    assert result.version == "1.0.0"
