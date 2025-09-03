# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pytest

import orchestrator.core.samplestore.csv
import orchestrator.plugins.samplestores.gt4sd
from orchestrator.core.samplestore.base import ExperimentDescription
from orchestrator.modules.actuators.catalog import ExperimentCatalog
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.reference import ExperimentReference


@pytest.fixture(scope="module")
def catalog_with_parameterizable_experiments(
    mock_parameterizable_experiment,
    mock_parameterizable_experiment_no_required,
    mock_parameterizable_experiment_with_required_observed,
) -> ExperimentCatalog:
    """Returns a catalog for the Mock actuator with a parameterized experiment"""

    return ExperimentCatalog(
        experiments={
            mock_parameterizable_experiment.identifier: mock_parameterizable_experiment,
            mock_parameterizable_experiment_with_required_observed.identifier: mock_parameterizable_experiment_with_required_observed,
            mock_parameterizable_experiment_no_required.identifier: mock_parameterizable_experiment_no_required,
        }
    )


@pytest.fixture(scope="module")
def global_registry(
    catalog_with_parameterizable_experiments,
) -> ActuatorRegistry:

    r = ActuatorRegistry.globalRegistry()
    r.updateCatalogs(catalogExtension=catalog_with_parameterizable_experiments)

    return r


@pytest.fixture
def experiment_catalogs() -> (
    list[orchestrator.modules.actuators.catalog.ExperimentCatalog]
):
    parameters = {}

    experimentDescription = ExperimentDescription(
        experimentIdentifier="transformer-toxicity-inference-experiment",
        propertyMap=orchestrator.plugins.samplestores.gt4sd.GT4SDTransformer.propertyMap,
    )

    parameters["experiments"] = [experimentDescription]
    parameters["identifierColumn"] = "smiles"
    parameters["source"] = "tests/test_generations.csv"
    parameters["generatorIdentifier"] = "gt4sd-pfas-transformer-model-one"
    parameters["constitutivePropertyColumns"] = ["smiles"]

    sourceDescription = orchestrator.core.samplestore.csv.CSVSampleStoreDescription(
        **parameters
    )

    assert (
        sourceDescription.catalog.experimentForReference(
            reference=ExperimentReference(
                experimentIdentifier="transformer-toxicity-inference-experiment",
                actuatorIdentifier="replay",
            )
        )
        is not None
    )

    return [sourceDescription.catalog]
