# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Database and Ray fixtures for active-learning integration tests."""

from collections.abc import Callable, Generator

import pytest
import ray
from ray.actor import ActorHandle

from ado.core.discoveryspace.space import DiscoverySpace
from ado.modules.actuators.measurement_queue import MeasurementQueue
from ado.modules.operators.discovery_space_manager import DiscoverySpaceManager
from tests.conftest import random_identifier as random_identifier
from tests.fixtures.metastore import (
    valid_ado_sqlite_context_yaml as valid_ado_sqlite_context_yaml,
)
from tests.fixtures.metastore import (
    valid_ado_sqlite_project_context as valid_ado_sqlite_project_context,
)


@pytest.fixture
def remote_discovery_space(
    initialize_regression_ray: None,
) -> Generator[Callable[[DiscoverySpace], ActorHandle], None, None]:
    """Create real discovery-space actors and release their queues after a test."""

    resources: list[tuple[ActorHandle, MeasurementQueue]] = []

    def create(space: DiscoverySpace) -> ActorHandle:
        """Expose a discovery space through its production Ray manager."""

        queue = MeasurementQueue()
        manager = DiscoverySpaceManager.remote(space=space, queue=queue)
        resources.append((manager, queue))
        return manager

    yield create

    for manager, queue in resources:
        ray.kill(manager)
        queue.shutdown()


@pytest.fixture(scope="session")
def initialize_regression_ray() -> Generator[None, None, None]:
    """Start Ray when a regression test requests a discovery-space actor."""

    ray.init(
        ignore_reinit_error=True,
        object_store_memory=200 * 1024 * 1024,
    )
    yield
    ray.shutdown()
