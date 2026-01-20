# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

import yaml
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.project import ProjectContext
from orchestrator.utilities.output import pydantic_model_as_yaml


def test_create_discovery_space_dry_run_success(tmp_path: pathlib.Path) -> None:
    space_configuration_file = "examples/ml-multi-cloud/ml_multicloud_space.yaml"
    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "space",
            "-f",
            space_configuration_file,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    expected_output = (
        "INFO:   Initializing contexts - local is now your default context.\n"
        "The configuration passed is valid!\n"
    )
    assert result.output == expected_output


def test_create_discovery_space_dry_run_failure(tmp_path: pathlib.Path) -> None:
    space_configuration_file = pathlib.Path(
        "examples/ml-multi-cloud/ml_multicloud_space.yaml"
    )
    invalid_space_configuration_file = tmp_path / "invalid_space.yaml"
    invalid_space_configuration_file.write_text(
        space_configuration_file.read_text() + "\nnonexistent-key: hello"
    )

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "space",
            "-f",
            invalid_space_configuration_file,
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    expected_output = (
        "INFO:   Initializing contexts - local is now your default context.\n"
        "ERROR:  The space configuration provided was not valid:\n"
    )
    assert result.output.startswith(expected_output)


def test_create_discovery_space_fail_no_sample_store(tmp_path: pathlib.Path) -> None:
    space_configuration_file = "examples/ml-multi-cloud/ml_multicloud_space.yaml"

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "space",
            "-f",
            space_configuration_file,
            "--set",
            "sampleStoreIdentifier=d976ee",
        ],
    )

    assert result.exit_code == 1, result.output
    expected_output = (
        "INFO:   Initializing contexts - local is now your default context.\n"
        "ERROR:  The database does not contain a resource with id d976ee and kind samplestore.\n"
        "HINT:   Your active context is local - are you sure it's the correct one?\n"
        "        You can change it with ado context\n"
    )
    assert result.output == expected_output


# 28/11/2025
# AP: It's important to have this test early on before successful tests because
# they will populate the global ActuatorRegistry with the replay.benchmark_performance
# experiment, causing this test to be able to succeed.
def test_create_discovery_space_fail_with_default_sample_store_with_replay_actuator(
    tmp_path: pathlib.Path,
) -> None:

    space_configuration_file = pathlib.Path(
        "examples/ml-multi-cloud/ml_multicloud_space.yaml"
    )

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "space",
            "-f",
            space_configuration_file,
            "--with",
            "store=default",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "The default sample store was requested to be used." in result.output
    assert (
        "The following experiment was not found: replay.benchmark_performance"
        in result.output
    )


def test_create_discovery_space_success(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    space_configuration_file = tmp_path / "space_configuration.yaml"
    space_configuration = DiscoverySpaceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path("examples/ml-multi-cloud/ml_multicloud_space.yaml").read_text()
        )
    )
    space_configuration.sampleStoreIdentifier = ml_multi_cloud_sample_store.identifier
    space_configuration_file.write_text(pydantic_model_as_yaml(space_configuration))

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "space",
            "-f",
            space_configuration_file,
        ],
    )

    assert result.exit_code == 0, result.output
    expected_output = "Success! Created space with identifier"
    assert result.output.startswith(expected_output)
    assert result.output.strip().endswith(ml_multi_cloud_sample_store.identifier)


def test_create_discovery_space_success_new_sample_store(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    space_configuration_file = pathlib.Path(
        "plugins/actuators/example_actuator/yamls/discoveryspace.yaml"
    )
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "space",
            "-f",
            space_configuration_file,
            "--new-sample-store",
        ],
    )
    assert result.exit_code == 0
    expected_output = (
        "INFO:   A new sample store was requested.\n"
        "        Sample store a267f0 referenced in the space definition will be ignored.\n"
        "Success! Created space with identifier:"
    )
    assert result.output.startswith(expected_output)


def test_create_discovery_space_success_with_latest_samplestore(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    space_configuration_file = pathlib.Path(
        "plugins/actuators/example_actuator/yamls/discoveryspace.yaml"
    )
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "samplestore",
            "--new-sample-store",
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "space",
            "-f",
            space_configuration_file,
            "--use-latest",
            "samplestore",
        ],
    )
    assert result.exit_code == 0
    expected_output = (
        "INFO:   The latest sample store was requested to be reused.\n"
        "        Sample Stores referenced in the space definition will be ignored and replaced with"
    )
    assert result.output.startswith(expected_output)


def test_create_discovery_space_fail_new_sample_store_with_replay(
    tmp_path: pathlib.Path,
) -> None:
    space_configuration_file = pathlib.Path(
        "examples/ml-multi-cloud/ml_multicloud_space.yaml"
    )
    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "space",
            "-f",
            space_configuration_file,
            "--new-sample-store",
        ],
    )

    assert result.exit_code == 1, result.output
    expected_output = (
        "INFO:   Initializing contexts - local is now your default context.\n"
        "ERROR:  You cannot use --new-sample-store with a space that uses the replay actuator.\n"
        "HINT:   Provide a sampleStoreIdentifier in the space configuration.\n"
    )
    assert result.output == expected_output


def test_create_discovery_space_success_set_sample_store(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    space_configuration_file = pathlib.Path(
        "examples/ml-multi-cloud/ml_multicloud_space.yaml"
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "space",
            "-f",
            space_configuration_file,
            "--set",
            f'sampleStoreIdentifier="{ml_multi_cloud_sample_store.identifier}"',
        ],
    )

    assert result.exit_code == 0, result.output
    expected_output = "Success! Created space with identifier"
    assert result.output.startswith(expected_output)
    assert result.output.strip().endswith(ml_multi_cloud_sample_store.identifier)


def test_create_discovery_space_success_with_sample_store_from_file_with_replay_actuator(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    space_configuration_file = pathlib.Path(
        "examples/ml-multi-cloud/ml_multicloud_space.yaml"
    )

    sample_store_configuration_file = pathlib.Path(
        "tests/resources/ml_multicloud_sample_store.yaml"
    )

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "space",
            "-f",
            space_configuration_file,
            "--with",
            f"store={sample_store_configuration_file}",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Success! Created sample store with identifier" in result.output
    assert "Success! Created space with identifier" in result.output
