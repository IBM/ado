# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

import pytest
import yaml

from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import DiscoveryOperationResourceConfiguration
from orchestrator.core.samplestore.sql import SQLSampleStore


@pytest.fixture
def peptide_mineralization_basic_space(
    empty_sample_store: SQLSampleStore,
    create_space: Callable[
        [DiscoverySpaceConfiguration, str],
        DiscoverySpace,
    ],
) -> DiscoverySpace:
    space_configuration = DiscoverySpaceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "plugins/actuators/example_actuator/yamls/discoveryspace.yaml"
            ).read_text()
        )
    )
    return create_space(space_configuration, empty_sample_store.identifier)


@pytest.fixture
def peptide_mineralization_basic_operation_configuration(
    peptide_mineralization_basic_space: DiscoverySpace,
) -> DiscoveryOperationResourceConfiguration:

    operation_configuration = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "plugins/actuators/example_actuator/yamls/random_walk_operation.yaml"
            ).read_text()
        )
    )
    operation_configuration.spaces = [peptide_mineralization_basic_space.uri]
    return operation_configuration
