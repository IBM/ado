# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pytest
import yaml

import orchestrator.metastore.project
import orchestrator.modules.actuators.base
import orchestrator.modules.actuators.catalog
import orchestrator.modules.actuators.custom_experiments
import orchestrator.modules.actuators.replay
import orchestrator.modules.module
import orchestrator.schema.entity
import orchestrator.schema.experiment
import orchestrator.schema.property_value
import orchestrator.schema.reference
from orchestrator.core.actuatorconfiguration.config import (
    ActuatorConfiguration,
)
from orchestrator.modules.actuators.catalog import ExperimentCatalog


@pytest.fixture
def objectiveFunctionConfigurationYAML():

    y = """
actuatorIdentifier: "custom_experiments"
    """

    return yaml.safe_load(y)


@pytest.fixture
def objectiveFunctionConfiguration(objectiveFunctionConfigurationYAML):

    return ActuatorConfiguration(**objectiveFunctionConfigurationYAML)


@pytest.fixture
def actuatorModuleConfigurationYAML():

    y = """
        moduleName: "myactuator"
        modulePath: "examples/test-project"   #This is the path relative to where `ado` will be run from to this dir
        moduleClass: MyActuator
    """

    return yaml.safe_load(y)


@pytest.fixture
def actuatorCatalogExtensionConfigurationYAML():

    y = """
    name: custom_experiments.yaml
    location: 'examples/pfas-generative-models/'
    """

    return yaml.safe_load(y)


@pytest.fixture
def actuatorModuleConfiguration(
    actuatorModuleConfigurationYAML,
) -> orchestrator.modules.actuators.base.ActuatorModuleConf:

    return orchestrator.modules.actuators.base.ActuatorModuleConf(
        **actuatorModuleConfigurationYAML
    )


@pytest.fixture
def actuatorCatalogExtensionConfiguration(
    actuatorCatalogExtensionConfigurationYAML,
) -> orchestrator.modules.actuators.catalog.ActuatorCatalogExtensionConf:
    return orchestrator.modules.actuators.catalog.ActuatorCatalogExtensionConf(
        **actuatorCatalogExtensionConfigurationYAML
    )


def test_custom_experiments(objectiveFunctionConfiguration, experiment_catalogs):

    import ray

    import orchestrator.modules.actuators.base
    import orchestrator.modules.actuators.registry

    # noinspection PyUnresolvedReferences
    custom_experiments = (
        orchestrator.modules.actuators.custom_experiments.CustomExperiments.remote(
            queue=None, params=objectiveFunctionConfiguration
        )
    )

    # This is to test that the ObjectiveFunction instance has got the extended catalog
    # from the registry
    catalog = custom_experiments.current_catalog.remote()
    catalog: ExperimentCatalog = ray.get(catalog)

    assert catalog, "custom_experiments returned None for catalog"
    # AP 18/10/24:
    # This should work on Travis as we now install
    #  - examples/pfas-generative-models/custom_actuator_function
    #  - examples/optimization_test_functions/custom_experiments
    # Locally this may not work because we might have more or less of these.
    assert (
        len(catalog.experiments) == 2
    ), "Expected 2 experiments in the custom_experiments catalog for testing "

    identifiers = {e.identifier for e in catalog.experiments}
    assert {
        "acid_test",
        "nevergrad_opt_3d_test_func",
    } == identifiers, f"Expected the experiments to be called - acid_test and nevergrad_opt_3d_test_func but they are called {identifiers}"
    loaded = custom_experiments.loadedExperiment.remote(
        orchestrator.schema.reference.ExperimentReference(
            actuatorIdentifier="custom_experiments", experimentIdentifier="acid_test"
        )
    )

    assert ray.get(loaded), "Experiment found but not loaded by custom_experiments"

    c = orchestrator.modules.actuators.registry.ActuatorRegistry().catalogForActuatorIdentifier(
        "custom_experiments"
    )

    assert len(c.experiments) == 2

    for e in c.experiments:
        assert catalog.experimentForReference(e.reference) is not None

    for e in catalog.experiments:
        assert c.experimentForReference(e.reference) is not None
