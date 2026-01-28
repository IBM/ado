# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
from collections.abc import Callable

import pytest
import yaml

from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.samplestore.base import ActiveSampleStore
from orchestrator.core.samplestore.config import SampleStoreConfiguration
from orchestrator.core.samplestore.sql import SQLSampleStore


@pytest.fixture
def pfas_sample_store_configuration_str() -> str:
    return """
    specification:
      module:
        moduleName: orchestrator.core.samplestore.sql
        moduleClass: SQLSampleStore
    copyFrom:
    - module:
        moduleClass: GT4SDTransformer
        moduleName: orchestrator.plugins.samplestores.gt4sd
      storageLocation:
        path: 'tests/test_generations.csv'
      parameters:
        generatorIdentifier: 'gt4sd-pfas-transformer-model-one'
    """


@pytest.fixture
def pfas_space_configuration_str() -> str:
    return """
    sampleStoreIdentifier: 'TBA'
    entitySpace:
      - identifier: 'smiles'
    experiments:
      - experimentIdentifier: 'transformer-toxicity-inference-experiment'
        actuatorIdentifier: 'replay'
    """


@pytest.fixture
def pfas_sample_store(
    pfas_sample_store_configuration_str: str,
    create_sample_store: Callable[[SampleStoreConfiguration], ActiveSampleStore],
) -> SQLSampleStore:
    sample_store_configuration = SampleStoreConfiguration.model_validate(
        yaml.safe_load(pfas_sample_store_configuration_str)
    )
    return create_sample_store(sample_store_configuration)


@pytest.fixture
def pfas_space_configuration(
    pfas_sample_store: SQLSampleStore,
    pfas_space_configuration_str: str,
    create_space: Callable[[DiscoverySpaceConfiguration, str], DiscoverySpace],
) -> DiscoverySpaceConfiguration:
    space_configuration = DiscoverySpaceConfiguration.model_validate(
        yaml.safe_load(pfas_space_configuration_str)
    )
    space_configuration.sampleStoreIdentifier = pfas_sample_store.identifier
    return space_configuration


@pytest.fixture
def pfas_space(
    pfas_sample_store: SQLSampleStore,
    pfas_space_configuration: DiscoverySpaceConfiguration,
    create_space: Callable[[DiscoverySpaceConfiguration, str], DiscoverySpace],
) -> DiscoverySpace:
    return create_space(pfas_space_configuration, pfas_sample_store.identifier)
