# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import sys
from collections.abc import Callable

import pytest
from testcontainers.community.mysql import MySqlContainer
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import DiscoveryOperationResourceConfiguration
from ado.core.resources import CoreResourceKinds
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore
from ado.utilities.output import pydantic_model_as_yaml
from tests.conftest import requires_sqlite_3_38

try:
    from yaml import CSafeDumper as SafeDumper
except ImportError:  # pragma: nocover
    from yaml import SafeDumper  # type: ignore[assignment]


def _operation_configuration_as_yaml(
    configuration: DiscoveryOperationResourceConfiguration,
) -> str:
    """Serialize an operation configuration without derived ``spaces``."""
    import yaml

    return yaml.dump(
        configuration.model_dump(mode="json", exclude={"spaces"}),
        Dumper=SafeDumper,
    )


@pytest.fixture(
    params=[
        "ml_multi_cloud_operation_configuration",
        "peptide_mineralization_basic_operation_configuration",
    ]
)
def operations_to_be_run(
    request: pytest.FixtureRequest,
) -> DiscoveryOperationResourceConfiguration:
    return request.getfixturevalue(request.param)


def test_create_operation_dry_run_success(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    operations_to_be_run: DiscoveryOperationResourceConfiguration,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    operation_configuration_file = tmp_path / "operation.yaml"
    operation_configuration_file.write_text(
        _operation_configuration_as_yaml(operations_to_be_run)
    )

    result = runner.invoke(
        ado,
        [
            "create",
            "operation",
            "-f",
            operation_configuration_file,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    expected_output = (
        "INFO:   The operation YAML is syntactically valid.\n"
        "The configuration passed is valid!\n"
    )
    assert result.output == expected_output


def test_create_operation_dry_run_failure(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_operation_configuration: DiscoveryOperationResourceConfiguration,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    invalid_operation_configuration_file = tmp_path / "invalid_operation.yaml"
    invalid_operation_configuration_file.write_text(
        _operation_configuration_as_yaml(ml_multi_cloud_operation_configuration)
        + "\nnonexistent-key: hello"
    )

    result = runner.invoke(
        ado,
        [
            "create",
            "operation",
            "-f",
            invalid_operation_configuration_file,
            "--dry-run",
        ],
    )

    assert result.exit_code == 1
    expected_output = "ERROR:  The operation configuration provided was not valid:\n"
    assert result.output.startswith(expected_output)


def test_create_operation_success(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    operations_to_be_run: DiscoveryOperationResourceConfiguration,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    operation_file = tmp_path / "temp_op.yaml"
    operation_file.write_text(pydantic_model_as_yaml(operations_to_be_run))

    result = runner.invoke(
        ado,
        [
            "create",
            "operation",
            "-f",
            operation_file,
        ],
    )
    assert result.exit_code == 0


def test_create_operation_success_set_spaces(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    operation_configuration_file = pathlib.Path(
        "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
    )

    result = runner.invoke(
        ado,
        [
            "create",
            "operation",
            "-f",
            operation_configuration_file,
            "--set",
            f"inputs.discoverySpace.identifier={ml_multi_cloud_space.uri}",
        ],
    )
    assert result.exit_code == 0, result.output


def test_create_operation_success_with_discovery_space(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
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
        "examples/optimization_test_functions/space.yaml"
    )

    operation_configuration_file = pathlib.Path(
        "examples/optimization_test_functions/operation_bayesopt.yaml"
    )

    result = runner.invoke(
        ado,
        [
            "create",
            "operation",
            "-f",
            operation_configuration_file,
            "--set",
            "operation.parameters.tuneConfig.num_samples=2",
            "--with",
            f"space={space_configuration_file}",
        ],
    )
    assert result.exit_code == 0, result.output


def test_create_operation_dry_run_with_discovery_space(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    space_configuration_file = pathlib.Path(
        "examples/optimization_test_functions/space.yaml"
    )
    operation_configuration_file = pathlib.Path(
        "examples/optimization_test_functions/operation_bayesopt.yaml"
    )

    result = runner.invoke(
        ado,
        [
            "create",
            "operation",
            "-f",
            str(operation_configuration_file),
            "--with",
            f"space={space_configuration_file}",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    spaces = sql_store.getResourceIdentifiersOfKind(
        kind=CoreResourceKinds.DISCOVERYSPACE
    )
    assert spaces.empty


def test_create_operation_dry_run_with_actuator_configuration(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    actuator_configuration_file = pathlib.Path(
        "tests/resources/actuatorconfiguration/robotic_lab.yaml"
    )
    operation_configuration_file = pathlib.Path(
        "examples/optimization_test_functions/operation_bayesopt.yaml"
    )

    result = runner.invoke(
        ado,
        [
            "create",
            "operation",
            "-f",
            str(operation_configuration_file),
            "--with",
            f"actuatorconfiguration={actuator_configuration_file}",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    actuator_configurations = sql_store.getResourceIdentifiersOfKind(
        kind=CoreResourceKinds.ACTUATORCONFIGURATION
    )
    assert actuator_configurations.empty


@requires_sqlite_3_38
@pytest.mark.flaky(
    reruns=3,
    reruns_delay=5,
    condition=sys.version_info >= (3, 14),
    only_rerun="OutOfMemoryError",
    reason="numba OOM on Python 3.14 when JIT compilation fails for nevergrad",
)
def test_create_ml_multi_cloud_operation_success_lhc_sampler(
    tmp_path: pathlib.Path,
    ml_multi_cloud_sample_store_configuration_file: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    space_creation_result = runner.invoke(
        ado,
        [
            "create",
            "space",
            "-f",
            "examples/ml-multi-cloud/ml_multicloud_space.yaml",
            "--with",
            f"store={ml_multi_cloud_sample_store_configuration_file}",
        ],
    )
    assert space_creation_result.exit_code == 0, space_creation_result.output

    result = runner.invoke(
        ado,
        [
            "create",
            "operation",
            "-f",
            "examples/ml-multi-cloud/lhc_sampler.yaml",
            "--use-latest",
            "space",
        ],
    )
    assert result.exit_code == 0, result.output
