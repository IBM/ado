# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import uuid
from collections.abc import Callable, Generator

import pytest
import ray

from orchestrator.core.legacy.registry import LegacyValidatorRegistry

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


@pytest.fixture(scope="session")
def random_identifier() -> Callable[[], str]:
    def _random_identifier() -> str:
        return str(uuid.uuid4()).replace("-", "_")[:8]

    return _random_identifier


@pytest.fixture(scope="session", autouse=True)
def initialize_ray() -> Generator[None, None, None]:
    """Initialize Ray with working_dir=None to avoid package size issues during tests."""
    # Using dict form instead of RuntimeEnv object - they behave differently
    ray.init(
        runtime_env={"working_dir": None},
        ignore_reinit_error=True,
    )
    yield
    ray.shutdown()


@pytest.fixture(scope="session", autouse=True)
def session_legacy_validators() -> dict:
    """Load legacy validators once per session and return a copy.

    This session-scoped fixture ensures validators are loaded once at the start
    of the test session and the registered validators are saved. This copy can
    then be used by test-scoped fixtures to reset the registry state.

    Returns:
        A dictionary copy of all registered validators
    """
    # Import to trigger registration - this happens once per test session
    import orchestrator.core.legacy.validators  # noqa: F401

    # Return a copy of the registered validators
    return LegacyValidatorRegistry._validators.copy()


@pytest.fixture
def isolated_legacy_validator_registry() -> Generator[None, None, None]:
    """Isolate the LegacyValidatorRegistry for each test.

    This fixture ensures that modifications to the registry in one test
    do not affect other tests, even when running with pytest -n auto.

    The fixture:
    1. Saves the current registry state before the test
    2. Clears the registry for the test
    3. Restores the original state after the test

    Usage:
        def test_something(isolated_legacy_validator_registry):
            # Registry starts empty
            # Register validators as needed for this test
            # Changes won't affect other tests
    """
    # Save the current state
    original_validators = LegacyValidatorRegistry._validators.copy()

    # Clear for this test
    LegacyValidatorRegistry._validators.clear()

    try:
        yield
    finally:
        # Restore original state
        LegacyValidatorRegistry._validators = original_validators


@pytest.fixture
def legacy_validators_loaded(
    session_legacy_validators: dict,
) -> Generator[None, None, None]:
    """Ensure legacy validators are loaded and isolated for the test.

    This fixture:
    1. Resets the registry to the session state (all validators loaded)
    2. Allows the test to run (potentially modifying the registry)
    3. Restores the registry to the session state after the test

    This ensures:
    - All validators are available to the test
    - Test modifications don't affect other tests
    - Consistent behavior across pytest-xdist workers

    The session_legacy_validators fixture loads validators once per test session,
    and this fixture resets to that known-good state before and after each test.

    Usage:
        def test_with_real_validators(legacy_validators_loaded):
            # All validators are registered and available
            # Test can use them without affecting other tests
    """
    # Reset registry to session state before test
    LegacyValidatorRegistry._validators = session_legacy_validators.copy()

    try:
        yield
    finally:
        # Restore registry to session state after test
        LegacyValidatorRegistry._validators = session_legacy_validators.copy()
