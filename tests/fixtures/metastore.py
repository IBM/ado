# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import os
import pathlib
from collections.abc import Callable

import pytest
import sqlalchemy
import testcontainers.mysql
import yaml
from typer.testing import CliRunner

import orchestrator.core
import orchestrator.metastore
import orchestrator.metastore.project
import orchestrator.metastore.sqlstore
from orchestrator.cli.core.cli import app as ado
from orchestrator.core import DiscoverySpaceResource
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLStore
from orchestrator.modules.module import ModuleConf, ModuleTypeEnum
from orchestrator.utilities.output import pydantic_model_as_yaml


@pytest.fixture
def orchestrator_project_name() -> str:
    return "spark-opt"


@pytest.fixture(
    params=[
        ModuleTypeEnum.ACTUATOR,
        ModuleTypeEnum.SAMPLE_STORE,
        ModuleTypeEnum.OPERATION,
    ]
)
def module_config(request: pytest.FixtureRequest) -> ModuleConf:
    return ModuleConf(moduleType=request.param)


@pytest.fixture(scope="session")
def mysql_test_password() -> str:
    return "a-possible-password-for-mysql"


@pytest.fixture(scope="session")
def mysql_test_port() -> int:
    return 3306


@pytest.fixture(scope="session")
def mysql_test_instance(
    mysql_test_password: str,
    mysql_test_port: int,
) -> testcontainers.mysql.MySqlContainer:

    # AP 20/03/2024: Ryuk is not connecting locally and we
    # don't really need it since we're using the context manager
    # ref: https://github.com/testcontainers/testcontainers-python/pull/314
    os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"

    # We need to use root as we need to create databases on-the-fly
    with testcontainers.mysql.MySqlContainer(
        image="mirror.gcr.io/mysql:8",
        username="root",
        root_password=mysql_test_password,
        password=mysql_test_password,
        port=mysql_test_port,
        dialect="pymysql",
    ) as mysql_instance:
        yield mysql_instance


@pytest.fixture
def valid_ado_mysql_context_yaml(
    random_identifier: Callable[[], str],
    mysql_test_password: str,
    mysql_test_port: int,
    mysql_test_instance: testcontainers.mysql.MySqlContainer,
) -> str:
    random_id = random_identifier()
    db_name = f"pytest_{random_id}"

    with sqlalchemy.create_engine(
        mysql_test_instance.get_connection_url()
    ).begin() as connection:
        connection.execute(sqlalchemy.text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))

    return f"""
    project: "{db_name}"
    metadataStore:
        scheme: "mysql+pymysql"
        host: {mysql_test_instance.get_container_host_ip()}
        port: {mysql_test_instance.get_exposed_port(mysql_test_port)}
        password: {mysql_test_password}
        user: root
        database: "{db_name}"
        sslVerify: false
    """


@pytest.fixture
def valid_ado_sqlite_context_yaml(random_identifier: Callable[[], str]) -> str:
    # AP: adding a string before random_identifier as it would
    # otherwise fail validation if it's only numbers
    project_id = f"sqlite-{random_identifier()}"
    yield f"""
        metadataStore:
          database: {project_id}
          path: {project_id}.db
          scheme: sqlite
          sslVerify: false
        project: {project_id}
    """
    pathlib.Path(f"{project_id}.db").unlink(missing_ok=True)


@pytest.fixture(params=["mysql", "sqlite"])
def valid_ado_project_context(
    valid_ado_mysql_context_yaml: str,
    valid_ado_sqlite_context_yaml: str,
    request: pytest.FixtureRequest,
) -> orchestrator.metastore.project.ProjectContext:

    context = (
        valid_ado_sqlite_context_yaml
        if request.param == "sqlite"
        else valid_ado_mysql_context_yaml
    )

    return orchestrator.metastore.project.ProjectContext.model_validate(
        yaml.safe_load(context)
    )


@pytest.fixture
def create_active_ado_context() -> (
    Callable[[CliRunner, pathlib.Path, ProjectContext], None]
):

    def _create_active_ado_context(
        runner: CliRunner,
        path: pathlib.Path,
        project_context: ProjectContext,
    ) -> None:
        context_file = path / "test_context.yaml"
        context_file.write_text(pydantic_model_as_yaml(project_context))

        creation_result = runner.invoke(
            ado,
            [
                "--override-ado-app-dir",
                str(path),
                "create",
                "context",
                "-f",
                context_file,
            ],
        )
        assert creation_result.exit_code == 0

        activation_result = runner.invoke(
            ado,
            [
                "--override-ado-app-dir",
                str(path),
                "context",
                project_context.project,
            ],
        )
        assert activation_result.exit_code == 0

        return

    return _create_active_ado_context


@pytest.fixture(scope="module", params=["mysql", "sqlite"])
def ado_test_file_project_context(request: pytest.FixtureRequest) -> ProjectContext:

    import yaml

    if request.param == "sqlite":
        path = "tests/sqlite_test_context.yaml"
    else:
        path = "tests/mysql_test_context.yaml"

    with open(path) as f:
        return ProjectContext.model_validate(yaml.safe_load(f))


@pytest.fixture
def resource_store(
    valid_ado_project_context: ProjectContext,
) -> orchestrator.metastore.sqlstore.SQLStore:

    return orchestrator.metastore.sqlstore.SQLStore(
        project_context=valid_ado_project_context
    )


@pytest.fixture(params=list(CoreResourceKinds))
def resource_type(request: pytest.FixtureRequest) -> CoreResourceKinds:
    return request.param


@pytest.fixture
def discovery_space_resource_and_store(
    resource_store: SQLStore,
) -> (DiscoverySpaceResource, SQLStore):
    """Returns a DiscoverySpace resource that can be used to test additions"""

    # Return an existing discovery space conf
    identifiers = resource_store.getResourceIdentifiersOfKind(
        kind=CoreResourceKinds.DISCOVERYSPACE.value
    )
    return (
        resource_store.getResource(
            identifier=identifiers["IDENTIFIER"][0],
            kind=CoreResourceKinds.DISCOVERYSPACE,
        ),
        resource_store,
    )


@pytest.fixture
def active_contest_test_sample_store_resource(
    ado_test_file_project_context: ProjectContext,
) -> orchestrator.core.samplestore.resource.SampleStoreResource:

    return orchestrator.core.samplestore.resource.SampleStoreResource(
        config=orchestrator.core.samplestore.config.SampleStoreConfiguration(
            specification=orchestrator.core.samplestore.config.SampleStoreSpecification(
                module=orchestrator.core.samplestore.config.SampleStoreModuleConf(
                    moduleClass="SQLSampleStore",
                    moduleName="orchestrator.core.samplestore.sql",
                ),
                storageLocation=ado_test_file_project_context.metadataStore,
            )
        ),
    )
