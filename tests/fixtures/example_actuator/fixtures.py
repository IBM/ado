# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

import pytest
import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: nocover
    from yaml import SafeLoader  # type: ignore[assignment]

from ado.core.discoveryspace.config import DiscoverySpaceConfiguration
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import DiscoveryOperationResourceConfiguration
from ado.core.resources import ADOResourceReference, CoreResourceKinds
from ado.core.samplestore.sql import SQLSampleStore


@pytest.fixture
def peptide_mineralization_basic_space(
    empty_sample_store: SQLSampleStore,
    create_space: Callable[
        [DiscoverySpaceConfiguration, str],
        DiscoverySpace,
    ],
) -> DiscoverySpace:
    space_configuration = DiscoverySpaceConfiguration.model_validate(
        yaml.load(
            pathlib.Path(
                "plugins/actuators/example_actuator/yamls/discoveryspace.yaml"
            ).read_text(),
            Loader=SafeLoader,
        )
    )
    return create_space(space_configuration, empty_sample_store.identifier)


@pytest.fixture
def peptide_mineralization_basic_operation_configuration(
    peptide_mineralization_basic_space: DiscoverySpace,
) -> DiscoveryOperationResourceConfiguration:

    operation_configuration = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.load(
            pathlib.Path(
                "plugins/actuators/example_actuator/yamls/random_walk_operation.yaml"
            ).read_text(),
            Loader=SafeLoader,
        )
    )
    operation_configuration.inputs = {
        "discoverySpace": ADOResourceReference(
            identifier=peptide_mineralization_basic_space.uri,
            kind=CoreResourceKinds.DISCOVERYSPACE,
        )
    }
    return operation_configuration
