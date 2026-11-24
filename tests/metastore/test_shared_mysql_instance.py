# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import functools
import multiprocessing
import os
import pathlib
import time
import uuid
from collections.abc import Generator
from typing import Any

import docker
import docker.errors
import pytest

from tests.fixtures.shared_mysql_instance import (
    CONTAINER_NAME_FILE_NAME,
    MYSQL_TEST_CONTAINER_OWNER_LABEL,
    MySqlConnectionDetails,
    cleanup_shared_instance,
    get_or_start_shared_instance,
    read_shared_instance,
    remove_mysql_test_container,
    remove_stale_mysql_test_containers,
)

CONCURRENT_PROCESSES = 6
STARTER_DURATION_SECONDS = 0.5
TEST_CONTAINER_IMAGE = "mirror.gcr.io/mysql:8"


def recording_starter(
    event_log: pathlib.Path, duration_seconds: float, container_name: str
) -> MySqlConnectionDetails:
    """Start a fake instance.

    Args:
        event_log: File that receives one "started:<container_name>" line per
            call.
        duration_seconds: Time to wait before returning, to widen the window in
            which concurrent callers contend for the lock.
        container_name: Name of the container to start.

    Returns:
        Connection details with a unique container id.
    """
    time.sleep(duration_seconds)
    with event_log.open("a") as log:
        log.write(f"started:{container_name}\n")
    return MySqlConnectionDetails(
        containerId=f"container-{uuid.uuid4().hex}",
        host="localhost",
        port=33306,
        password="a-test-password",
    )


def failing_starter(
    event_log: pathlib.Path, container_name: str
) -> MySqlConnectionDetails:
    """Create a fake container and fail before the instance is ready.

    Args:
        event_log: File that receives one "started:<container_name>" line.
        container_name: Name of the container to start.

    Raises:
        RuntimeError: Always.
    """
    with event_log.open("a") as log:
        log.write(f"started:{container_name}\n")
    raise RuntimeError("the instance did not become ready")


def recording_stale_remover(
    event_log: pathlib.Path, previous_instance_marker: pathlib.Path
) -> None:
    """Remove a leftover instance.

    Args:
        event_log: File that receives one "stale-removed" line per call.
        previous_instance_marker: File standing for the leftover instance;
            deleted if present.
    """
    previous_instance_marker.unlink(missing_ok=True)
    with event_log.open("a") as log:
        log.write("stale-removed\n")


def recording_stopper(event_log: pathlib.Path, container_name: str) -> None:
    """Stop an instance.

    Args:
        event_log: File that receives one "stopped:<container_name>" line per
            call.
        container_name: Name of the container to stop.
    """
    with event_log.open("a") as log:
        log.write(f"stopped:{container_name}\n")


def request_shared_instance(
    shared_directory: pathlib.Path,
    event_log: pathlib.Path,
    previous_instance_marker: pathlib.Path,
) -> dict[str, Any]:
    """Request the shared instance with recording callbacks.

    Args:
        shared_directory: Directory holding the lock file and the records.
        event_log: File in which the callbacks record their calls.
        previous_instance_marker: File standing for a leftover instance.

    Returns:
        The received connection details as a dictionary.
    """
    details = get_or_start_shared_instance(
        shared_directory=shared_directory,
        starter=functools.partial(
            recording_starter, event_log, STARTER_DURATION_SECONDS
        ),
        stopper=functools.partial(recording_stopper, event_log),
        stale_instance_remover=functools.partial(
            recording_stale_remover, event_log, previous_instance_marker
        ),
    )
    return details.model_dump()


def read_events(event_log: pathlib.Path) -> list[str]:
    """Read the recorded events.

    Args:
        event_log: File in which events are recorded.

    Returns:
        The recorded lines, or an empty list if nothing was recorded.
    """
    if not event_log.exists():
        return []
    return event_log.read_text().splitlines()


def started_container_names(event_log: pathlib.Path) -> list[str]:
    """Read the names of the containers that were started.

    Args:
        event_log: File in which events are recorded.

    Returns:
        The container names of the "started:<container_name>" events.
    """
    return [
        event.removeprefix("started:")
        for event in read_events(event_log)
        if event.startswith("started:")
    ]


@pytest.fixture
def shared_directory(tmp_path: pathlib.Path) -> pathlib.Path:
    """Provide an empty directory for the lock file and the records."""
    directory = tmp_path / "shared"
    directory.mkdir()
    return directory


@pytest.fixture
def event_log(tmp_path: pathlib.Path) -> pathlib.Path:
    """Provide the path of the file in which starter calls are recorded."""
    return tmp_path / "events.log"


@pytest.fixture
def previous_instance_marker(tmp_path: pathlib.Path) -> pathlib.Path:
    """Provide the path of the file standing for a leftover instance."""
    return tmp_path / "previous_instance.marker"


def test_connection_details_lifecycle_and_url() -> None:
    """Details survive a dump and re-creation, and construct the connection URL."""
    details = MySqlConnectionDetails(
        containerId="abc123", host="127.0.0.1", port=33306, password="pw"
    )

    recreated = MySqlConnectionDetails.model_validate(details.model_dump())
    recreated_from_json = MySqlConnectionDetails.model_validate_json(
        details.model_dump_json()
    )

    assert recreated == details
    assert recreated_from_json == details
    assert details.get_connection_url() == "mysql+pymysql://root:pw@127.0.0.1:33306/"


def test_concurrent_processes_start_instance_exactly_once(
    shared_directory: pathlib.Path,
    event_log: pathlib.Path,
    previous_instance_marker: pathlib.Path,
) -> None:
    """Concurrent processes start the instance once and receive equal details."""
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=CONCURRENT_PROCESSES) as pool:
        results = pool.starmap(
            request_shared_instance,
            [(shared_directory, event_log, previous_instance_marker)]
            * CONCURRENT_PROCESSES,
        )

    assert len(started_container_names(event_log)) == 1
    assert len(results) == CONCURRENT_PROCESSES
    assert all(result == results[0] for result in results)
    assert read_shared_instance(shared_directory) == MySqlConnectionDetails(
        **results[0]
    )


def test_read_shared_instance_without_state_returns_none(
    shared_directory: pathlib.Path,
) -> None:
    """Reading from a directory without a state file returns None."""
    assert read_shared_instance(shared_directory) is None


def test_stale_instance_is_removed_before_starting(
    shared_directory: pathlib.Path,
    event_log: pathlib.Path,
    previous_instance_marker: pathlib.Path,
) -> None:
    """A leftover instance is removed before the new one is started."""
    previous_instance_marker.write_text("left over from an aborted run")

    request_shared_instance(shared_directory, event_log, previous_instance_marker)

    events = read_events(event_log)
    assert len(events) == 2
    assert events[0] == "stale-removed"
    assert events[1].startswith("started:")
    assert not previous_instance_marker.exists()


def test_cleanup_stops_recorded_instance_and_removes_records(
    shared_directory: pathlib.Path,
    event_log: pathlib.Path,
    previous_instance_marker: pathlib.Path,
) -> None:
    """Cleanup stops the started container once and deletes the records."""
    request_shared_instance(shared_directory, event_log, previous_instance_marker)
    (started_name,) = started_container_names(event_log)

    cleanup_shared_instance(
        shared_directory=shared_directory,
        stopper=functools.partial(recording_stopper, event_log),
    )

    assert read_events(event_log).count(f"stopped:{started_name}") == 1
    assert read_shared_instance(shared_directory) is None
    assert not (shared_directory / CONTAINER_NAME_FILE_NAME).exists()


def test_cleanup_without_records_is_a_noop(
    shared_directory: pathlib.Path, event_log: pathlib.Path
) -> None:
    """Cleanup without a recorded container stops nothing."""
    cleanup_shared_instance(
        shared_directory=shared_directory,
        stopper=functools.partial(recording_stopper, event_log),
    )

    assert read_events(event_log) == []
    assert read_shared_instance(shared_directory) is None


def test_cleanup_stops_container_whose_start_never_finished(
    shared_directory: pathlib.Path, event_log: pathlib.Path
) -> None:
    """A container name recorded without connection details is still stopped."""
    (shared_directory / CONTAINER_NAME_FILE_NAME).write_text("interrupted-container")

    cleanup_shared_instance(
        shared_directory=shared_directory,
        stopper=functools.partial(recording_stopper, event_log),
    )

    assert read_events(event_log) == ["stopped:interrupted-container"]
    assert not (shared_directory / CONTAINER_NAME_FILE_NAME).exists()


def test_failed_start_stops_the_container_and_allows_a_retry(
    shared_directory: pathlib.Path,
    event_log: pathlib.Path,
    previous_instance_marker: pathlib.Path,
) -> None:
    """A start that fails stops its container, records nothing and can be retried."""
    with pytest.raises(RuntimeError, match="did not become ready"):
        get_or_start_shared_instance(
            shared_directory=shared_directory,
            starter=functools.partial(failing_starter, event_log),
            stopper=functools.partial(recording_stopper, event_log),
            stale_instance_remover=functools.partial(
                recording_stale_remover, event_log, previous_instance_marker
            ),
        )

    (failed_name,) = started_container_names(event_log)
    assert f"stopped:{failed_name}" in read_events(event_log)
    assert read_shared_instance(shared_directory) is None
    assert not (shared_directory / CONTAINER_NAME_FILE_NAME).exists()

    request_shared_instance(shared_directory, event_log, previous_instance_marker)

    assert read_shared_instance(shared_directory) is not None
    assert len(started_container_names(event_log)) == 2
    assert started_container_names(event_log)[1] != failed_name


@pytest.fixture
def docker_client() -> Generator[docker.DockerClient, None, None]:
    """Provide a Docker client with the MySQL test image available locally."""
    client = docker.from_env()
    try:
        client.images.get(TEST_CONTAINER_IMAGE)
    except docker.errors.ImageNotFound:
        client.images.pull(TEST_CONTAINER_IMAGE)
    yield client
    client.close()


def test_stale_removal_keeps_containers_owned_by_running_processes(
    docker_client: docker.DockerClient,
) -> None:
    """Only the container whose owner pid no longer exists is removed."""
    context = multiprocessing.get_context("spawn")
    exited_process = context.Process(target=time.sleep, args=(0,))
    exited_process.start()
    exited_process.join()

    suffix = uuid.uuid4().hex[:8]
    live_owner_name = f"ado-pytest-stale-test-live-{suffix}"
    dead_owner_name = f"ado-pytest-stale-test-dead-{suffix}"
    try:
        for name, owner_pid in (
            (live_owner_name, os.getpid()),
            (dead_owner_name, exited_process.pid),
        ):
            docker_client.containers.create(
                TEST_CONTAINER_IMAGE,
                name=name,
                labels={MYSQL_TEST_CONTAINER_OWNER_LABEL: str(owner_pid)},
            )

        remove_stale_mysql_test_containers()

        assert docker_client.containers.get(live_owner_name) is not None
        with pytest.raises(docker.errors.NotFound):
            docker_client.containers.get(dead_owner_name)
    finally:
        remove_mysql_test_container(live_owner_name)
        remove_mysql_test_container(dead_owner_name)
