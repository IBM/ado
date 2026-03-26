# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import importlib
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
def legacy_validators_loaded() -> Generator[None, None, None]:
    """Ensure legacy validators are loaded and isolated for the test.

    This fixture:
    1. Imports the validators module to trigger registration
    2. Uses importlib.reload() to force re-execution even if cached
    3. Saves a copy of the registered validators
    4. Provides isolation so test modifications don't affect other tests
    5. Restores the validators after the test

    Use this when your test needs the actual validators to be registered
    (e.g., for integration tests that use real validators).

    IMPORTANT: We use importlib.reload() to ensure the module is re-executed
    even if it was previously imported and cached by Python or pytest-xdist workers.

    Usage:
        def test_with_real_validators(legacy_validators_loaded):
            # All validators are registered and available
            # Test can use them without affecting other tests
    """
    # Import to trigger registration
    import orchestrator.core.legacy.validators

    # Force reload to ensure decorators execute even if module was cached
    importlib.reload(orchestrator.core.legacy.validators)

    # Save the current state (includes all registered validators)
    original_validators = LegacyValidatorRegistry._validators.copy()

    try:
        yield
    finally:
        # Restore to ensure other tests see the same state
        LegacyValidatorRegistry._validators = original_validators
