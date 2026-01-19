# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pytest
import yaml

import orchestrator.schema
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.modules.actuators.catalog import ExperimentCatalog
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.experiment import ParameterizedExperiment
from orchestrator.schema.measurementspace import (
    MeasurementSpace,
    MeasurementSpaceConfiguration,
)
from orchestrator.schema.reference import ExperimentReference


@pytest.fixture
def measurement_space_direct(
    measurement_space_configuration: list[ExperimentReference],
    experiment_catalogs: list[ExperimentCatalog],
) -> MeasurementSpace:
    selectedExperiments = measurement_space_configuration

    space = MeasurementSpace.measurementSpaceFromSelection(
        selectedExperiments, experiment_catalogs
    )

    assert (
        space.experimentForReference(
            reference=ExperimentReference(
                experimentIdentifier="transformer-toxicity-inference-experiment",
                actuatorIdentifier="replay",
            )
        )
        is not None
    )

    return space


@pytest.fixture
def measurement_space_from_discovery_configuration(
    pfas_sample_store: SQLSampleStore,
    pfas_space_configuration_str: str,
) -> MeasurementSpace:

    space_configuration = DiscoverySpaceConfiguration.model_validate(
        yaml.safe_load(pfas_space_configuration_str)
    )
    space_configuration.sampleStoreIdentifier = pfas_sample_store.identifier

    assert space_configuration.experiments is not None

    measurementspace = MeasurementSpace.measurementSpaceFromSelection(
        selectedExperiments=space_configuration.experiments,
        experimentCatalogs=[pfas_sample_store.experimentCatalog()],
    )

    catalog = ActuatorRegistry.globalRegistry().catalogForActuatorIdentifier("replay")
    for exp in measurementspace.experiments:
        if exp.actuatorIdentifier == "replay":
            catalog.addExperiment(exp)

    assert (
        measurementspace.experimentForReference(
            reference=ExperimentReference(
                experimentIdentifier="transformer-toxicity-inference-experiment",
                actuatorIdentifier="replay",
            )
        )
        is not None
    )

    return measurementspace


@pytest.fixture(params=["direct-mspace", "disconf-mspace"])
def measurement_space(
    request: pytest.FixtureRequest,
    measurement_space_from_discovery_configuration: MeasurementSpace,
    measurement_space_direct: MeasurementSpace,
) -> MeasurementSpace:
    d = {
        "direct-mspace": measurement_space_direct,
        "disconf-mspace": measurement_space_from_discovery_configuration,
    }

    return d[request.param]


@pytest.fixture(scope="module")
def parameterized_selectors(
    parameterized_experiments: list[ParameterizedExperiment],
    global_registry: ActuatorRegistry,
) -> list[ExperimentReference]:

    return [
        orchestrator.schema.measurementspace.ExperimentReference(
            experimentIdentifier=e.identifier,
            actuatorIdentifier=e.actuatorIdentifier,
            parameterization=e.parameterization,
        )
        for e in parameterized_experiments
    ]


@pytest.fixture
def measurement_space_configuration(
    measurement_space_configuration_smiles_yaml: dict[str, str],
) -> list[ExperimentReference]:
    return [
        ExperimentReference(**e) for e in measurement_space_configuration_smiles_yaml
    ]


@pytest.fixture
def measurement_space_configuration_smiles_yaml() -> dict[str, str]:
    y = """
      - experimentIdentifier: 'transformer-toxicity-inference-experiment'
        actuatorIdentifier: 'replay'
    """

    return yaml.safe_load(y)


@pytest.fixture
def measurement_space_from_multiple_parameterized_experiments(
    parameterized_experiments: list[ParameterizedExperiment],
    global_registry: ActuatorRegistry,
) -> MeasurementSpace:
    """A MeasurementSpace created from multiple parameterized experiments"""

    return MeasurementSpace(
        configuration=MeasurementSpaceConfiguration(
            experiments=parameterized_experiments
        )
    )


@pytest.fixture
def measurement_space_from_single_parameterized_experiment(
    parameterized_experiment: ParameterizedExperiment, global_registry: ActuatorRegistry
) -> MeasurementSpace:
    """A MeasurementSpace created from a single parameterized experiment"""

    return MeasurementSpace(
        configuration=MeasurementSpaceConfiguration(
            experiments=[parameterized_experiment]
        )
    )
