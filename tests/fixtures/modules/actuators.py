# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


import pytest

import ado.core.samplestore.csv
from ado.core.actuatorconfiguration.config import GenericActuatorParameters
from ado.modules.actuators.base import ActuatorBase
from ado.modules.actuators.catalog import ExperimentCatalog
from ado.modules.actuators.measurement_queue import MeasurementQueue, NullQueue
from ado.modules.actuators.registry import ActuatorRegistry
from ado.schema.entity import Entity
from ado.schema.experiment import Experiment
from ado.schema.reference import ExperimentReference


class TestActuator(ActuatorBase):
    """Minimal actuator used only in tests."""

    identifier = "test"

    def __init__(
        self, queue: MeasurementQueue | NullQueue, params: dict | None = None
    ) -> None:
        """Initialize the TestActuator.

        Args:
            queue: A measurement queue the actuator can use to put results.
            params: Optional parameters dict.
        """
        super().__init__(queue=queue, params=params)

    def submit(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
    ) -> list[str]:
        """Not implemented — test actuator is never used for real submissions.

        Args:
            entities: Entities to measure.
            experimentReference: Experiment reference.
            requesterid: Requester identifier.
            requestIndex: Request index.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> ExperimentCatalog:
        """Return an empty experiment catalog for the test actuator.

        Args:
            actuator_configuration: Ignored.

        Returns:
            An empty ExperimentCatalog with catalogIdentifier "test".
        """
        return ExperimentCatalog(catalogIdentifier="test")


@pytest.fixture
def test_actuator_catalog() -> ExperimentCatalog:
    """Register TestActuator into the global registry and yield a fresh catalog.

    Registers the test actuator under identifier "test" (idempotent if already
    registered), replaces any existing catalog entry with a fresh empty
    ExperimentCatalog, and yields it. On teardown, restores the previous
    catalog entry so that module-scoped fixtures (e.g. ``global_registry``)
    that added experiments to the "test" catalog are not disrupted.

    Yields:
        A fresh, empty ExperimentCatalog registered under "test".
    """
    registry = ActuatorRegistry.globalRegistry()
    registry.registerActuator("test", TestActuator, is_builtin=True)
    previous_catalog = registry.catalogIdentifierMap.get("test")
    fresh_catalog = ExperimentCatalog(catalogIdentifier="test")
    registry.catalogIdentifierMap["test"] = fresh_catalog
    yield fresh_catalog
    if previous_catalog is None:
        registry.catalogIdentifierMap.pop("test", None)
    else:
        registry.catalogIdentifierMap["test"] = previous_catalog


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
    r.registerActuator("test", TestActuator, is_builtin=True)
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
                "observedPropertyMap": {
                    "logws": "GenLogws",
                    "logd": "GenLogd",
                    "loghl": "GenLoghl",
                    "pka": "GenPka",
                    "biodegradation halflife": "GenBiodeg",
                    "bcf": "GenBcf",
                    "ld50": "GenLd50",
                    "scscore": "GenScscore",
                },
                "constitutivePropertyMap": ["smiles"],
            }
        ],
    }

    sourceDescription = (
        ado.core.samplestore.csv.CSVSampleStoreDescription.model_validate(parameters)
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
