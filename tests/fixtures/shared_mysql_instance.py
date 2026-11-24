# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Coordination of one MySQL instance shared by several pytest processes.

The first process that requests the instance starts it while holding a file
lock. Before starting, it records the name of the container in a shared
directory, so the container can be stopped even if the process dies before
the connection details are recorded. Once the instance is up, its connection
details are recorded in a state file in the same directory. Later requests read
the recorded details. Cleanup stops the recorded container and deletes both
files.

Each container carries a label with the pid of the pytest process that owns it,
so a later run only removes leftover containers whose owner has exited and
never the container of a concurrent run.

The module also provides the pytest fixtures that expose the shared instance
to tests.
"""

import functools
import os
import pathlib
import uuid
from collections.abc import Callable, Generator
from typing import Annotated

import docker
import docker.errors
import filelock
import pydantic
import pytest
import testcontainers.community.mysql as testcontainers_mysql

STATE_FILE_NAME = "mysql_instance.json"
CONTAINER_NAME_FILE_NAME = "mysql_container_name.txt"
LOCK_FILE_NAME = "mysql_instance.lock"
MYSQL_TEST_CONTAINER_NAME_PREFIX = "ado-pytest-mysql"
MYSQL_TEST_CONTAINER_OWNER_LABEL = "ado-pytest-mysql-owner-pid"
MYSQL_SHARED_DIRECTORY_INPUT_KEY = "mysql_shared_directory"
MYSQL_OWNER_PID_INPUT_KEY = "mysql_owner_pid"


class MySqlConnectionDetails(pydantic.BaseModel):
    """Connection details of a running MySQL instance.

    Exposes the accessors of ``MySqlContainer`` that the tests rely on.
    """

    containerId: Annotated[
        str, pydantic.Field(description="Identifier of the container running MySQL")
    ]
    host: Annotated[
        str, pydantic.Field(description="Host on which the instance is reachable")
    ]
    port: Annotated[
        int, pydantic.Field(description="Published port the instance listens on")
    ]
    password: Annotated[
        str, pydantic.Field(description="Password of the root user of the instance")
    ]

    def get_connection_url(self) -> str:
        """Build the SQLAlchemy URL of the server for the root user.

        Returns:
            A ``mysql+pymysql`` URL with the root credentials, host and
            published port.
        """
        return f"mysql+pymysql://root:{self.password}@{self.host}:{self.port}/"

    def get_container_host_ip(self) -> str:
        """Return the host on which the instance is reachable."""
        return self.host

    def get_exposed_port(self, container_port: int) -> int:
        """Return the published port of the instance.

        Args:
            container_port: The port MySQL listens on inside the container.

        Returns:
            The port published on the host.
        """
        return self.port


def read_shared_instance(
    shared_directory: pathlib.Path,
) -> MySqlConnectionDetails | None:
    """Read the recorded connection details.

    Args:
        shared_directory: Directory holding the state file.

    Returns:
        The recorded details, or None if no instance has been recorded.
    """
    state_file = shared_directory / STATE_FILE_NAME
    if not state_file.exists():
        return None
    return MySqlConnectionDetails.model_validate_json(state_file.read_text())


def stop_recorded_instance(
    shared_directory: pathlib.Path,
    stopper: Callable[[str], None],
) -> None:
    """Stop the container whose name is recorded and delete the records.

    The caller must hold the lock on ``shared_directory``. Does nothing if no
    container name is recorded.

    Args:
        shared_directory: Directory holding the container name file and the
            state file.
        stopper: Stops and removes the container with the given name.
    """
    container_name_file = shared_directory / CONTAINER_NAME_FILE_NAME
    if not container_name_file.exists():
        return

    stopper(container_name_file.read_text())
    container_name_file.unlink()
    (shared_directory / STATE_FILE_NAME).unlink(missing_ok=True)


def get_or_start_shared_instance(
    shared_directory: pathlib.Path,
    starter: Callable[[str], MySqlConnectionDetails],
    stopper: Callable[[str], None],
    stale_instance_remover: Callable[[], None],
) -> MySqlConnectionDetails:
    """Return the shared instance, starting it if none has been recorded.

    Concurrent callers are serialised with a ``filelock.FileLock`` located in
    ``shared_directory``. The caller that finds no recorded instance calls
    ``stale_instance_remover``, records a new container name, and calls
    ``starter`` exactly once with that name. It then records the result so that
    every other caller receives the same details. If ``starter`` or the
    recording fails, the container with the recorded name is stopped with
    ``stopper`` and the exception propagates, so a later caller can retry.

    Args:
        shared_directory: Directory holding the lock file, the container name
            file and the state file.
        starter: Starts a new instance in a container with the given name and
            returns its connection details.
        stopper: Stops and removes the container with the given name.
        stale_instance_remover: Removes instances left over from previous
            aborted runs; called before ``starter``.

    Returns:
        The connection details of the shared instance.
    """
    with filelock.FileLock(shared_directory / LOCK_FILE_NAME):
        recorded_details = read_shared_instance(shared_directory)
        if recorded_details is not None:
            return recorded_details

        stale_instance_remover()

        # The name is recorded before starting so that the container can be
        # found and stopped even if this process dies while starting it.
        container_name = f"{MYSQL_TEST_CONTAINER_NAME_PREFIX}-{uuid.uuid4().hex[:8]}"
        (shared_directory / CONTAINER_NAME_FILE_NAME).write_text(container_name)

        try:
            details = starter(container_name)

            # Writing to a temporary file and renaming keeps the state file
            # complete for readers that do not take the lock.
            temporary_file = shared_directory / f"{STATE_FILE_NAME}.tmp"
            temporary_file.write_text(details.model_dump_json())
            temporary_file.replace(shared_directory / STATE_FILE_NAME)
        except BaseException:
            stop_recorded_instance(shared_directory, stopper)
            raise

        return details


def cleanup_shared_instance(
    shared_directory: pathlib.Path,
    stopper: Callable[[str], None],
) -> None:
    """Stop the recorded instance and delete its records.

    Takes the same lock as ``get_or_start_shared_instance``, so an instance
    that is still being started is stopped once its start has finished. Does
    nothing if no container name is recorded.

    Args:
        shared_directory: Directory holding the lock file, the container name
            file and the state file.
        stopper: Stops and removes the container with the given name.
    """
    with filelock.FileLock(shared_directory / LOCK_FILE_NAME):
        stop_recorded_instance(shared_directory, stopper)


@pytest.fixture(scope="session")
def mysql_test_password() -> str:
    """Provide the password of the root user of the MySQL test instance."""
    return "a-possible-password-for-mysql"


@pytest.fixture(scope="session")
def mysql_test_port() -> int:
    """Provide the port MySQL listens on inside the test container."""
    return 3306


def start_mysql_test_container(
    container_name: str, password: str, port: int, owner_pid: int
) -> MySqlConnectionDetails:
    """Start the MySQL test container and leave it running.

    The container is not stopped when the calling process exits.

    Args:
        container_name: Name of the container.
        password: Password of the root user.
        port: Port MySQL listens on inside the container.
        owner_pid: Pid of the pytest process that owns the container; stored
            in the container's owner label.

    Returns:
        The connection details of the running container.
    """
    # We need to use root as we need to create databases on-the-fly
    mysql_container = testcontainers_mysql.MySqlContainer(
        image="mirror.gcr.io/mysql:8",
        username="root",
        root_password=password,
        password=password,
        port=port,
        dialect="pymysql",
        labels={MYSQL_TEST_CONTAINER_OWNER_LABEL: str(owner_pid)},
    ).with_name(container_name)
    mysql_container.start()

    return MySqlConnectionDetails(
        containerId=mysql_container.get_wrapped_container().id,
        host=mysql_container.get_container_host_ip(),
        port=int(mysql_container.get_exposed_port(port)),
        password=password,
    )


def remove_mysql_test_container(container_name_or_id: str) -> None:
    """Force-remove a container and its anonymous volumes if it exists.

    Args:
        container_name_or_id: Name or identifier of the container.
    """
    docker_client = docker.from_env()
    try:
        docker_client.containers.get(container_name_or_id).remove(force=True, v=True)
    except docker.errors.NotFound:
        pass
    finally:
        docker_client.close()


def remove_stale_mysql_test_containers() -> None:
    """Force-remove MySQL test containers whose owner process has exited.

    A container is stale when it carries the owner label and no process with
    the labelled pid exists. Containers owned by a running process are kept.
    """
    docker_client = docker.from_env()
    try:
        for container in docker_client.containers.list(
            all=True, filters={"label": MYSQL_TEST_CONTAINER_OWNER_LABEL}
        ):
            owner_pid = int(container.labels[MYSQL_TEST_CONTAINER_OWNER_LABEL])
            try:
                # Signal 0 only checks that the process exists.
                os.kill(owner_pid, 0)
            except ProcessLookupError:
                container.remove(force=True, v=True)
            except PermissionError:
                # The process exists but belongs to another user.
                continue
    finally:
        docker_client.close()


@pytest.fixture(scope="session")
def mysql_test_instance(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    mysql_test_password: str,
    mysql_test_port: int,
) -> Generator[MySqlConnectionDetails, None, None]:
    """Provide the MySQL instance shared by all tests of the run.

    Under pytest-xdist the first worker that requests this fixture starts the
    instance and every other worker reuses it; the xdist controller stops it
    when the run ends. Without xdist the instance is stopped when the session
    ends.

    Yields:
        The connection details of the shared instance.
    """
    # Ryuk removes a container when the process that started it exits. The
    # instance is shared by several processes, so it is removed explicitly
    # instead.
    os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"

    worker_input = getattr(request.config, "workerinput", None)
    if worker_input is None:
        shared_directory = tmp_path_factory.mktemp("mysql_shared")
        owner_pid = os.getpid()
    else:
        shared_directory = pathlib.Path(worker_input[MYSQL_SHARED_DIRECTORY_INPUT_KEY])
        owner_pid = worker_input[MYSQL_OWNER_PID_INPUT_KEY]

    try:
        yield get_or_start_shared_instance(
            shared_directory=shared_directory,
            starter=functools.partial(
                start_mysql_test_container,
                password=mysql_test_password,
                port=mysql_test_port,
                owner_pid=owner_pid,
            ),
            stopper=remove_mysql_test_container,
            stale_instance_remover=remove_stale_mysql_test_containers,
        )
    finally:
        if worker_input is None:
            cleanup_shared_instance(shared_directory, remove_mysql_test_container)
