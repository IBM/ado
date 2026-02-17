# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import uuid
from collections.abc import Callable, Generator

import pytest
import ray

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
