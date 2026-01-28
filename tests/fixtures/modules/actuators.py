# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pytest

import orchestrator.core.samplestore.csv
import orchestrator.plugins.samplestores.gt4sd
from orchestrator.modules.actuators.catalog import ExperimentCatalog
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.reference import ExperimentReference


@pytest.fixture(scope="module")
def catalog_with_parameterizable_experiments(
    mock_parameterizable_experiment: Experiment,
    mock_parameterizable_experiment_no_required: Experiment,
    mock_parameterizable_experiment_with_required_observed: Experiment,
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
    catalog_with_parameterizable_experiments: ExperimentCatalog,
) -> ActuatorRegistry:

    r = ActuatorRegistry.globalRegistry()
    r.updateCatalogs(catalogExtension=catalog_with_parameterizable_experiments)

    return r


@pytest.fixture
def experiment_catalogs() -> list[ExperimentCatalog]:
    parameters = {
        "identifierColumn": "smiles",
        "generatorIdentifier": "gt4sd-pfas-transformer-model-one",
        "experiments": [
            {
                "experimentIdentifier": "transformer-toxicity-inference-experiment",
                "actuatorIdentifier": "replay",
                "observedPropertyMap": dict(
                    orchestrator.plugins.samplestores.gt4sd.GT4SDTransformer.propertyMap
                ),
                "constitutivePropertyMap": ["smiles"],
            }
        ],
    }

    sourceDescription = (
        orchestrator.core.samplestore.csv.CSVSampleStoreDescription.model_validate(
            parameters
        )
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
