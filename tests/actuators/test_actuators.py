# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import sys
from typing import Any

import pytest
import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: nocover
    from yaml import SafeLoader  # type: ignore[assignment]

import ado.metastore.project
import ado.modules.actuators.base
import ado.modules.actuators.catalog
import ado.modules.actuators.custom_experiments
import ado.modules.actuators.replay
import ado.modules.module
import ado.schema.entity
import ado.schema.experiment
import ado.schema.property_value
import ado.schema.reference
from ado.core.actuatorconfiguration.config import (
    ActuatorConfiguration,
)
from ado.modules.actuators.catalog import ExperimentCatalog
from ado.modules.actuators.registry import ActuatorRegistry
from ado.schema.experiment import Experiment
from ado.schema.observed_property import ObservedProperty, ObservedPropertyValue
from ado.schema.point import SpacePoint


@pytest.fixture
def objectiveFunctionConfigurationYAML() -> dict[str, Any]:

    y = """
actuatorIdentifier: "custom_experiments"
    """

    return yaml.load(y, Loader=SafeLoader)


@pytest.fixture
def objectiveFunctionConfiguration(
    objectiveFunctionConfigurationYAML: dict[str, Any],
) -> ActuatorConfiguration:

    return ActuatorConfiguration(**objectiveFunctionConfigurationYAML)


@pytest.fixture
def actuatorCatalogExtensionConfigurationYAML() -> dict[str, Any]:

    y = """
    name: custom_experiments.yaml
    location: 'examples/pfas-generative-models/'
    """

    return yaml.load(y, Loader=SafeLoader)


@pytest.fixture
def actuatorCatalogExtensionConfiguration(
    actuatorCatalogExtensionConfigurationYAML: dict[str, Any],
) -> ado.modules.actuators.catalog.ActuatorCatalogExtensionConf:
    return ado.modules.actuators.catalog.ActuatorCatalogExtensionConf(
        **actuatorCatalogExtensionConfigurationYAML
    )


def test_custom_experiments(
    objectiveFunctionConfiguration: ActuatorConfiguration,
    experiment_catalogs: list[ExperimentCatalog],
) -> None:

    import ray

    ray.init(ignore_reinit_error=True)

    # noinspection PyUnresolvedReferences
    custom_experiments = ray.remote(
        ado.modules.actuators.custom_experiments.CustomExperiments
    ).remote(queue=None, params=objectiveFunctionConfiguration.parameters)

    # This is to test that the ObjectiveFunction instance has got the extended catalog
    # from the registry
    catalog = custom_experiments.current_catalog.remote()
    catalog: ExperimentCatalog = ray.get(catalog)

    assert catalog, "custom_experiments returned None for catalog"
    expected_identifiers = {
        "acid_test",
        "calculate_density",
        "ml_multicloud_cost",
        "min_gpu_recommender",
        "avoid_oom_recommender",
        "throughput_recommender",
        "nevergrad_opt_3d_test_func",
        "calculate_pressure_ideal_gas",
        "calculate_pressure_gas",
        "controlled_error",
    }
    if sys.version_info >= (3, 14):
        # TODO: add autoconf experiments back once it supports Python 3.14+.
        expected_identifiers -= {
            "min_gpu_recommender",
            "avoid_oom_recommender",
            "throughput_recommender",
        }

    # AP 18/10/24:
    # Locally this may not work because we might have more or less of these.
    # SV 7/02/26
    # This test needs to be updated every time a new custom experiment is added to ado
    assert len(catalog.experiments) == len(expected_identifiers), (
        "Unexpected number of experiments in the custom_experiments catalog for testing"
    )

    identifiers = {e.identifier for e in catalog.experiments}
    assert expected_identifiers == identifiers, (
        f"Expected experiment identifiers {expected_identifiers} but got {identifiers}"
    )
    loaded = custom_experiments.loadedExperiment.remote(
        ado.schema.reference.ExperimentReference(
            actuatorIdentifier="custom_experiments", experimentIdentifier="acid_test"
        )
    )

    assert ray.get(loaded), "Experiment found but not loaded by custom_experiments"

    c = ado.modules.actuators.registry.ActuatorRegistry().catalogForActuatorIdentifier(
        "custom_experiments"
    )

    assert len(c.experiments) == len(expected_identifiers)

    for e in c.experiments:
        assert catalog.experimentForReference(e.reference) is not None

    for e in catalog.experiments:
        assert c.experimentForReference(e.reference) is not None


def test_execute_nevergrad_opt_3d_test_func(
    experiment_catalogs: list[ExperimentCatalog],
) -> None:
    from ado.schema.point import SpacePoint
    from ado.utilities.run_experiment import local_execution_closure

    execute = local_execution_closure(
        registry=ado.modules.actuators.registry.ActuatorRegistry()
    )

    point = SpacePoint(
        entity={"x0": 1, "x1": 2, "x2": -1},
        experiments=[
            ado.schema.reference.ExperimentReference(
                actuatorIdentifier="custom_experiments",
                experimentIdentifier="nevergrad_opt_3d_test_func",
                experimentVersion="1.0.0",
            )
        ],
    )
    entity = point.to_entity()
    request: ado.schema.request.MeasurementRequest = execute(
        point.experiments[0], entity
    )

    assert request is not None
    assert request.status == ado.schema.request.MeasurementRequestStateEnum.SUCCESS
    assert request.measurements is not None
    assert len(request.measurements) == 1
    assert request.measurements[0].entityIdentifier == entity.identifier
    assert isinstance(request.measurements[0], ado.schema.result.ValidMeasurementResult)


def _entity_with_observed_value(
    constitutive_values: dict[str, Any],
    observed_property: ObservedProperty,
    value: float,
) -> ado.schema.entity.Entity:
    """Return an entity with the given constitutive values and one observed value.

    Args:
        constitutive_values: Constitutive property identifiers and their values.
        observed_property: The observed property the value is a measurement of.
        value: The measured value.

    Returns:
        The entity.
    """
    entity = SpacePoint(entity=constitutive_values).to_entity()
    entity.add_measurement_result(
        ado.schema.result.ValidMeasurementResult(
            entityIdentifier=entity.identifier,
            measurements=[
                ObservedPropertyValue(property=observed_property, value=value)
            ],
        )
    )
    return entity


def _measured_target_values(
    request: ado.schema.request.MeasurementRequest,
) -> dict[str, dict[str, Any]]:
    """Return each entity's measured values keyed by target property identifier.

    Args:
        request: A measurement request whose results are all valid.

    Returns:
        A dictionary mapping entity identifiers to their measured values.
    """
    return {
        result.entityIdentifier: {
            value.property.targetProperty.identifier: value.value
            for value in result.measurements
        }
        for result in request.measurements
    }


def test_execute_acid_test() -> None:
    """acid_test classifies a negative pKa as acid and a non-negative pKa as not."""
    reference = ado.schema.reference.ExperimentReference(
        actuatorIdentifier="custom_experiments", experimentIdentifier="acid_test"
    )
    experiment = ActuatorRegistry.globalRegistry().experimentForReference(reference)
    pka = experiment.requiredObservedProperties[0]
    acid = _entity_with_observed_value({"smiles": "CC(=O)O"}, pka, -1.5)
    not_acid = _entity_with_observed_value({"smiles": "CCO"}, pka, 4.0)

    request = ado.modules.actuators.custom_experiments.CustomExperiments().execute(
        entities=[acid, not_acid],
        experimentReference=reference,
        requesterid="test-acid-test",
        requestIndex=0,
    )

    assert request.status == ado.schema.request.MeasurementRequestStateEnum.SUCCESS
    measured_values = _measured_target_values(request)
    assert measured_values[acid.identifier]["isAcid"] == 1
    assert measured_values[not_acid.identifier]["isAcid"] == 0


def test_execute_ml_multicloud_cost(ml_multi_cloud_cost_experiment: Experiment) -> None:
    """ml_multicloud_cost reproduces the total cost in the search-custom-objective docs."""
    entity = _entity_with_observed_value(
        {"provider": "B", "cpu_family": 0, "vcpu_size": 0, "nodes": 3},
        ml_multi_cloud_cost_experiment.requiredObservedProperties[0],
        153.51639366149902,
    )

    request = ado.modules.actuators.custom_experiments.CustomExperiments().execute(
        entities=[entity],
        experimentReference=ml_multi_cloud_cost_experiment.reference,
        requesterid="test-ml-multicloud-cost",
        requestIndex=0,
    )

    assert request.status == ado.schema.request.MeasurementRequestStateEnum.SUCCESS
    assert _measured_target_values(request)[entity.identifier][
        "total_cost"
    ] == pytest.approx(1.2793032805124918)
