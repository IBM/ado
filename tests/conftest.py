# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import sqlite3
import uuid
from collections.abc import Callable, Generator

import pytest
import ray

from orchestrator.core.legacy.registry import LegacyMigratorRegistry

from .fixtures.core.datacontainer import *
from .fixtures.core.samplestore import *
from .fixtures.core.generators import *
from .fixtures.core.operation import *
from .fixtures.core.space import *
from .fixtures.samplestore.crud import *
from .fixtures.samplestore.fixtures import *
from .fixtures.metastore import *
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

# SQLite version check for tests requiring JSON operators (-> and ->>)
# ref: https://sqlite.org/json1.html#jptr
sqlite3_version = sqlite3.sqlite_version_info
requires_sqlite_3_38 = pytest.mark.skipif(
    sqlite3_version < (3, 38, 0), reason="SQLite version 3.38.0 or higher is required"
)


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


@pytest.fixture(scope="session", autouse=True)
def session_legacy_migrators() -> dict:
    """Load legacy migrators once per session and return a copy.

    This session-scoped fixture ensures migrators are loaded once at the start
    of the test session and the registered migrators are saved. This copy can
    then be used by test-scoped fixtures to reset the registry state.

    Returns:
        A dictionary copy of all registered migrators
    """
    # Import to trigger registration - this happens once per test session
    import orchestrator.core.legacy.migrators  # noqa: F401

    # Return a copy of the registered migrators
    return LegacyMigratorRegistry._migrators.copy()


@pytest.fixture
def isolated_legacy_migrator_registry() -> Generator[None, None, None]:
    """Isolate the LegacyMigratorRegistry for each test.

    This fixture ensures that modifications to the registry in one test
    do not affect other tests, even when running with pytest -n auto.

    The fixture:
    1. Saves the current registry state before the test
    2. Clears the registry for the test
    3. Restores the original state after the test

    Usage:
        def test_something(isolated_legacy_migrator_registry):
            # Registry starts empty
            # Register validators as needed for this test
            # Changes won't affect other tests
    """
    # Save the current state
    original_migrators = LegacyMigratorRegistry._migrators.copy()

    # Clear for this test
    LegacyMigratorRegistry._migrators.clear()

    try:
        yield
    finally:
        # Restore original state
        LegacyMigratorRegistry._migrators = original_migrators


@pytest.fixture
def legacy_migrators_loaded(
    session_legacy_migrators: dict,
) -> Generator[None, None, None]:
    """Ensure legacy migrators are loaded and isolated for the test.

    This fixture:
    1. Resets the registry to the session state (all validators loaded)
    2. Allows the test to run (potentially modifying the registry)
    3. Restores the registry to the session state after the test

    This ensures:
    - All validators are available to the test
    - Test modifications don't affect other tests
    - Consistent behavior across pytest-xdist workers

    The session_legacy_migrators fixture loads validators once per test session,
    and this fixture resets to that known-good state before and after each test.

    Usage:
        def test_with_real_migrators(legacy_migrators_loaded):
            # All validators are registered and available
            # Test can use them without affecting other tests
    """
    # Reset registry to session state before test
    LegacyMigratorRegistry._migrators = session_legacy_migrators.copy()

    try:
        yield
    finally:
        # Restore registry to session state after test
        LegacyMigratorRegistry._migrators = session_legacy_migrators.copy()
