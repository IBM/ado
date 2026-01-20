# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
import random
from collections.abc import Callable

import pytest
import yaml

import orchestrator.core.actuatorconfiguration.config
import orchestrator.core.discoveryspace.config
import orchestrator.core.samplestore.csv
import orchestrator.utilities.location
from orchestrator.core import ActuatorConfigurationResource, OperationResource
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
)
from orchestrator.core.samplestore.base import ActiveSampleStore
from orchestrator.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreReference,
)
from orchestrator.core.samplestore.csv import CSVSampleStore
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLResourceStore
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from orchestrator.schema.property import AbstractPropertyDescriptor
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import (
    MeasurementRequest,
    MeasurementRequestStateEnum,
    ReplayedMeasurement,
)
from orchestrator.schema.result import (
    InvalidMeasurementResult,
    MeasurementResult,
    MeasurementResultStateEnum,
    ValidMeasurementResult,
)


@pytest.fixture
def ml_multi_cloud_sample_store(
    create_sample_store: Callable[[SampleStoreConfiguration], ActiveSampleStore],
) -> SQLSampleStore:
    sample_store_configuration = SampleStoreConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path("tests/resources/ml_multicloud_sample_store.yaml").read_text()
        )
    )
    return create_sample_store(sample_store_configuration)


@pytest.fixture
def ml_multi_cloud_csv_sample_store() -> CSVSampleStore:
    sample_store_configuration = SampleStoreConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path("tests/resources/ml_multicloud_sample_store.yaml").read_text()
        )
    )

    csv_sample_store_parameters: SampleStoreReference = (
        sample_store_configuration.copyFrom[0]
    )

    return CSVSampleStore(
        storageLocation=orchestrator.utilities.location.FilePathLocation.model_validate(
            csv_sample_store_parameters.storageLocation
        ),
        parameters=orchestrator.core.samplestore.csv.CSVSampleStoreDescription.model_validate(
            csv_sample_store_parameters.parameters
        ),
    )


@pytest.fixture
def ml_multi_cloud_space(
    ml_multi_cloud_sample_store: SQLSampleStore,
    create_space: Callable[
        [orchestrator.core.discoveryspace.config.DiscoverySpaceConfiguration, str],
        DiscoverySpace,
    ],
) -> DiscoverySpace:
    space_configuration = orchestrator.core.discoveryspace.config.DiscoverySpaceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path("examples/ml-multi-cloud/ml_multicloud_space.yaml").read_text()
        )
    )
    return create_space(space_configuration, ml_multi_cloud_sample_store.identifier)


@pytest.fixture
def ml_multi_cloud_operation_configuration(
    ml_multi_cloud_space: DiscoverySpace,
) -> DiscoveryOperationResourceConfiguration:

    operation_configuration = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
            ).read_text()
        )
    )
    operation_configuration.spaces = [ml_multi_cloud_space.uri]
    return operation_configuration


@pytest.fixture
def ml_multi_cloud_correct_actuatorconfiguration(
    create_actuatorconfiguration: Callable[
        [orchestrator.core.actuatorconfiguration.config.ActuatorConfiguration],
        ActuatorConfigurationResource,
    ],
) -> ActuatorConfigurationResource:
    actuator_configuration = orchestrator.core.actuatorconfiguration.config.ActuatorConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "tests/resources/replay_actuatorconfiguration.yaml"
            ).read_text()
        )
    )
    return create_actuatorconfiguration(actuator_configuration)


@pytest.fixture
def ml_multi_cloud_invalid_actuatorconfiguration(
    create_actuatorconfiguration: Callable[
        [orchestrator.core.actuatorconfiguration.config.ActuatorConfiguration],
        ActuatorConfigurationResource,
    ],
) -> ActuatorConfigurationResource:
    actuator_configuration = orchestrator.core.actuatorconfiguration.config.ActuatorConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path("tests/resources/mock_actuatorconfiguration.yaml").read_text()
        )
    )
    return create_actuatorconfiguration(actuator_configuration)


@pytest.fixture
def ml_multi_cloud_cost_experiment() -> Experiment:
    return ActuatorRegistry.globalRegistry().experimentForReference(
        ExperimentReference(
            experimentIdentifier="ml-multicloud-cost-v1.0",
            actuatorIdentifier="custom_experiments",
        )
    )


@pytest.fixture
def ml_multi_cloud_benchmark_performance_experiment(
    ml_multi_cloud_csv_sample_store: CSVSampleStore,
) -> Experiment:
    return ml_multi_cloud_csv_sample_store.experimentCatalog().experimentForReference(
        ExperimentReference(
            experimentIdentifier="benchmark_performance",
            actuatorIdentifier="replay",
        )
    )


@pytest.fixture
def random_ml_multi_cloud_benchmark_performance_entities(
    ml_multi_cloud_csv_sample_store: CSVSampleStore,
) -> Callable[[int], list[Entity]]:
    def _random_ml_multi_cloud_benchmark_performance_entities(
        quantity: int,
    ) -> list[Entity]:
        return random.sample(
            population=ml_multi_cloud_csv_sample_store.entities, k=quantity
        )

    return _random_ml_multi_cloud_benchmark_performance_entities


@pytest.fixture
def random_ml_multi_cloud_benchmark_performance_measurement_results(
    random_identifier: str,
) -> Callable[[Entity, int, MeasurementResultStateEnum | None], MeasurementResult]:
    def _random_ml_multi_cloud_benchmark_performance_measurement_results(
        entity: Entity,
        measurements_per_result: int,
        status: MeasurementResultStateEnum | None = None,
    ) -> MeasurementResult:
        assert (
            measurements_per_result > 0
        ), "There need to be at least 1 measurement per result"
        status = status if status else MeasurementResultStateEnum.VALID

        if status == MeasurementResultStateEnum.VALID:
            return ValidMeasurementResult(
                entityIdentifier=entity.identifier,
                measurements=[
                    ObservedPropertyValue(
                        value=random.random(),
                        property=ObservedProperty(
                            targetProperty=AbstractPropertyDescriptor(
                                identifier="wallClockRuntime"
                            ),
                            experimentReference=ExperimentReference(
                                experimentIdentifier="benchmark_performance",
                                actuatorIdentifier="replay",
                            ),
                        ),
                    )
                    for _ in range(measurements_per_result)
                ],
            )
        return InvalidMeasurementResult(
            entityIdentifier=entity.identifier,
            reason=random_identifier(),
            experimentReference=ExperimentReference(
                experimentIdentifier="benchmark_performance",
                actuatorIdentifier="replay",
            ),
        )

    return _random_ml_multi_cloud_benchmark_performance_measurement_results


@pytest.fixture
def random_ml_multi_cloud_benchmark_performance_measurement_requests(
    random_identifier: Callable[[], str],
    random_ml_multi_cloud_benchmark_performance_entities: Callable[[int], list[Entity]],
    random_ml_multi_cloud_benchmark_performance_measurement_results: Callable[
        [Entity, int, MeasurementResultStateEnum | None], MeasurementResult
    ],
) -> Callable[
    [int, int, MeasurementRequestStateEnum | None, str | None],
    ReplayedMeasurement,
]:

    def _random_ml_multi_cloud_benchmark_performance_measurement_requests(
        number_entities: int,
        measurements_per_result: int,
        status: MeasurementRequestStateEnum | None = None,
        operation_id: str | None = None,
    ) -> ReplayedMeasurement:
        assert number_entities > 0, "There need to be at least 1 entity"
        entities = random_ml_multi_cloud_benchmark_performance_entities(number_entities)
        status = status if status else MeasurementRequestStateEnum.SUCCESS
        operation_id = operation_id if operation_id else random_identifier()

        return ReplayedMeasurement(
            operation_id=operation_id,
            requestIndex=random.randint(0, number_entities),
            experimentReference=ExperimentReference(
                experimentIdentifier="benchmark_performance",
                actuatorIdentifier="replay",
            ),
            entities=entities,
            requestid=random_identifier(),
            status=status,
            measurements=tuple(
                [
                    random_ml_multi_cloud_benchmark_performance_measurement_results(
                        entity=e, measurements_per_result=measurements_per_result
                    )
                    for e in entities
                ]
            ),
        )

    return _random_ml_multi_cloud_benchmark_performance_measurement_requests


@pytest.fixture
def simulate_ml_multi_cloud_random_walk_operation(
    valid_ado_project_context: ProjectContext,
    ml_multi_cloud_operation_configuration: DiscoveryOperationResourceConfiguration,
    ml_multi_cloud_sample_store: SQLSampleStore,
    random_identifier: Callable[[], str],
    random_ml_multi_cloud_benchmark_performance_measurement_requests: Callable[
        [int, int, MeasurementRequestStateEnum | None, str | None],
        ReplayedMeasurement,
    ],
    ml_multi_cloud_benchmark_performance_experiment: Experiment,
) -> Callable[
    [int, int, int, str | None],
    tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
]:
    def _simulate_ml_multi_cloud_random_walk_operation(
        number_entities: int = 3,
        number_requests: int = 3,
        measurements_per_result: int = 2,
        operation_id: str | None = None,
    ) -> tuple[SQLSampleStore, list[MeasurementRequest], list[str]]:
        operation_id = operation_id if operation_id else random_identifier()
        sample_store = ml_multi_cloud_sample_store

        sql = SQLResourceStore(
            project_context=valid_ado_project_context, ensureExists=True
        )

        sql.addResourceWithRelationships(
            OperationResource(
                identifier=operation_id,
                config=ml_multi_cloud_operation_configuration,
                operationType=DiscoveryOperationEnum.SEARCH,
                operatorIdentifier="doesnt-matter",
            ),
            relatedIdentifiers=ml_multi_cloud_operation_configuration.spaces,
        )

        requests = [
            random_ml_multi_cloud_benchmark_performance_measurement_requests(
                number_entities=number_entities,
                measurements_per_result=measurements_per_result,
                operation_id=operation_id,
            )
            for _ in range(number_requests)
        ]

        assert len(requests) == number_requests
        for r in requests:
            assert len(r.measurements) == number_entities
            for m in r.measurements:
                assert len(m.measurements) == measurements_per_result

        request_ids = [
            sample_store.add_measurement_request(request=requests[i])
            for i in range(number_requests)
        ]

        assert len(request_ids) == number_requests
        assert all(request_ids)

        for i in range(number_requests):
            sample_store.add_measurement_results(
                results=list(requests[i].measurements),
                skip_relationship_to_request=False,
                request_db_id=request_ids[i],
            )

        return sample_store, requests, request_ids

    return _simulate_ml_multi_cloud_random_walk_operation
