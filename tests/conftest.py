# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import os
import pathlib
import shutil
import sqlite3
import tempfile
import uuid
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import pytest
import ray

from .fixtures.ado_cli_isolation import *
from .fixtures.agents import *
from .fixtures.core.datacontainer import *
from .fixtures.core.samplestore import *
from .fixtures.core.generators import *
from .fixtures.core.operation import *
from .fixtures.core.space import *
from .fixtures.samplestore.crud import *
from .fixtures.samplestore.fixtures import *
from .fixtures.metastore import *
from .fixtures.shared_mysql_instance import *
from .fixtures.shared_mysql_instance import (
    MYSQL_OWNER_PID_INPUT_KEY,
    MYSQL_SHARED_DIRECTORY_INPUT_KEY,
    cleanup_shared_instance,
    remove_mysql_test_container,
)
from .fixtures.example_actuator.fixtures import *
from .fixtures.ml_multi_cloud.fixtures import *
from .fixtures.pfas.fixtures import *
from .fixtures.modules.actuators import *
from .fixtures.modules.operators import *
from .fixtures.schema.domain import *
from .fixtures.schema.entity import *
from .fixtures.schema.entityspace import *
from .fixtures.schema.experiment import *
from .fixtures.schema.measurementspace import *
from .fixtures.schema.properties import *
from .fixtures.schema.results import *

# This import is required to be run after the others,
# or we will get create_sample_store fixture not found.
from .fixtures.samplestore.crud_from_configurations import *

if TYPE_CHECKING:
    from xdist.workermanage import WorkerController

# SQLite version check for tests requiring JSON operators (-> and ->>)
# ref: https://sqlite.org/json1.html#jptr
sqlite3_version = sqlite3.sqlite_version_info
requires_sqlite_3_38 = pytest.mark.skipif(
    sqlite3_version < (3, 38, 0), reason="SQLite version 3.38.0 or higher is required"
)


mysql_shared_directory_key = pytest.StashKey[pathlib.Path]()


def pytest_configure_node(node: "WorkerController") -> None:
    """Pass an xdist worker the shared MySQL directory and the controller pid.

    The pid identifies the process that owns the MySQL container.

    Args:
        node: The worker being configured.
    """
    stash = node.config.stash
    if mysql_shared_directory_key not in stash:
        stash[mysql_shared_directory_key] = pathlib.Path(
            tempfile.mkdtemp(prefix="ado-pytest-mysql-")
        )
    node.workerinput[MYSQL_SHARED_DIRECTORY_INPUT_KEY] = str(
        stash[mysql_shared_directory_key]
    )
    node.workerinput[MYSQL_OWNER_PID_INPUT_KEY] = os.getpid()


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Stop the shared MySQL instance and delete its directory.

    Applies to the process that created the shared directory.

    Args:
        session: The finished test session.
    """
    shared_directory = session.config.stash.get(mysql_shared_directory_key, None)
    if shared_directory is None:
        return

    cleanup_shared_instance(shared_directory, remove_mysql_test_container)
    shutil.rmtree(shared_directory, ignore_errors=True)


@pytest.fixture(scope="session")
def random_identifier() -> Callable[[], str]:
    def _random_identifier() -> str:
        return str(uuid.uuid4()).replace("-", "_")[:8]

    return _random_identifier


@pytest.fixture(scope="session", autouse=True)
def initialize_ray() -> Generator[None, None, None]:
    """Start Ray for the test session.

    For ``uv run pytest``, the Ray "uv run" driver hook must be off by ensuring
    ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` when ``ray`` is first imported. This
    is set for pytest via ``pytest-env`` in ``pyproject.toml``
    """
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=200 * 1024 * 1024,  # 200 MB per xdist worker
    )
    yield
    ray.shutdown()
