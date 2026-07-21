# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import datetime
import pathlib
import random
from collections.abc import Callable

import pytest
import yaml

import ado.core.actuatorconfiguration.config
import ado.core.discoveryspace.config
import ado.core.samplestore.csv
import ado.utilities.location
from ado.core import (
    ActuatorConfigurationResource,
    CoreResourceKinds,
    OperationResource,
)
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
)
from ado.core.samplestore.base import ActiveSampleStore
from ado.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreReference,
)
from ado.core.samplestore.csv import CSVSampleStore
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLResourceStore, SQLStore
from ado.modules.actuators.registry import ActuatorRegistry
from ado.schema.entity import Entity
from ado.schema.experiment import Experiment
from ado.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from ado.schema.property import AbstractPropertyDescriptor
from ado.schema.reference import ExperimentReference
from ado.schema.request import (
    MeasurementRequest,
    MeasurementRequestStateEnum,
    ReplayedMeasurement,
)
from ado.schema.result import (
    InvalidMeasurementResult,
    MeasurementResult,
    MeasurementResultStateEnum,
    ValidMeasurementResult,
)
from ado.utilities.output import pydantic_model_as_yaml


@pytest.fixture
def ml_multi_cloud_sample_store_configuration() -> SampleStoreConfiguration:

    # The file in the examples assumes ml_export.csv is in the same directory
    raw_sample_store_configuration = yaml.safe_load(
        pathlib.Path(
            "examples/ml-multi-cloud/ml_multicloud_sample_store.yaml"
        ).read_text()
    )

    raw_sample_store_configuration["copyFrom"][0]["storageLocation"]["path"] = (
        "examples/ml-multi-cloud/ml_export.csv"
    )

    return SampleStoreConfiguration.model_validate(raw_sample_store_configuration)


@pytest.fixture
def ml_multi_cloud_sample_store_configuration_file(
    tmp_path: pathlib.Path,
    ml_multi_cloud_sample_store_configuration: SampleStoreConfiguration,
) -> pathlib.Path:
    file = tmp_path / "ml_multicloud_sample_store.yaml"
    file.write_text(pydantic_model_as_yaml(ml_multi_cloud_sample_store_configuration))
    return file


@pytest.fixture
def ml_multi_cloud_sample_store(
    ml_multi_cloud_sample_store_configuration: SampleStoreConfiguration,
    create_sample_store: Callable[[SampleStoreConfiguration], ActiveSampleStore],
) -> SQLSampleStore:
    return create_sample_store(ml_multi_cloud_sample_store_configuration)


@pytest.fixture
def ml_multi_cloud_csv_sample_store(
    ml_multi_cloud_sample_store_configuration: SampleStoreConfiguration,
) -> CSVSampleStore:

    csv_sample_store_parameters: SampleStoreReference = (
        ml_multi_cloud_sample_store_configuration.copyFrom[0]
    )

    return CSVSampleStore(
        storageLocation=ado.utilities.location.FilePathLocation.model_validate(
            csv_sample_store_parameters.storageLocation
        ),
        parameters=ado.core.samplestore.csv.CSVSampleStoreDescription.model_validate(
            csv_sample_store_parameters.parameters
        ),
    )


@pytest.fixture
def ml_multi_cloud_space(
    ml_multi_cloud_sample_store: SQLSampleStore,
    create_space: Callable[
        [ado.core.discoveryspace.config.DiscoverySpaceConfiguration, str],
        DiscoverySpace,
    ],
) -> DiscoverySpace:
    space_configuration = (
        ado.core.discoveryspace.config.DiscoverySpaceConfiguration.model_validate(
            yaml.safe_load(
                pathlib.Path(
                    "examples/ml-multi-cloud/ml_multicloud_space.yaml"
                ).read_text()
            )
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
    from ado.core.resources import ADOResourceReference, CoreResourceKinds

    operation_configuration.inputs["discoverySpace"] = ADOResourceReference(
        identifier=ml_multi_cloud_space.uri,
        kind=CoreResourceKinds.DISCOVERYSPACE,
    )
    return operation_configuration


@pytest.fixture
def ml_multi_cloud_correct_actuatorconfiguration(
    create_actuatorconfiguration: Callable[
        [ado.core.actuatorconfiguration.config.ActuatorConfiguration],
        ActuatorConfigurationResource,
    ],
) -> ActuatorConfigurationResource:
    actuator_configuration = (
        ado.core.actuatorconfiguration.config.ActuatorConfiguration.model_validate(
            yaml.safe_load(
                pathlib.Path(
                    "tests/resources/replay_actuatorconfiguration.yaml"
                ).read_text()
            )
        )
    )
    return create_actuatorconfiguration(actuator_configuration)


@pytest.fixture
def ml_multi_cloud_invalid_actuatorconfiguration(
    create_actuatorconfiguration: Callable[
        [ado.core.actuatorconfiguration.config.ActuatorConfiguration],
        ActuatorConfigurationResource,
    ],
) -> ActuatorConfigurationResource:
    actuator_configuration = (
        ado.core.actuatorconfiguration.config.ActuatorConfiguration.model_validate(
            yaml.safe_load(
                pathlib.Path(
                    "tests/resources/mock_actuatorconfiguration.yaml"
                ).read_text()
            )
        )
    )
    return create_actuatorconfiguration(actuator_configuration)


@pytest.fixture
def ml_multi_cloud_cost_experiment() -> Experiment:
    return ActuatorRegistry.globalRegistry().experimentForReference(
        ExperimentReference(
            experimentIdentifier="ml-multicloud-cost",
            actuatorIdentifier="custom_experiments",
            experimentVersion="1.0.0",
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
        assert measurements_per_result > 0, (
            "There need to be at least 1 measurement per result"
        )
        status = status or MeasurementResultStateEnum.VALID

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
        status = status or MeasurementRequestStateEnum.SUCCESS
        operation_id = operation_id or random_identifier()

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
    [int, int, int, str | None, "datetime.datetime | None"],
    tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
]:
    def _simulate_ml_multi_cloud_random_walk_operation(
        number_entities: int = 3,
        number_requests: int = 3,
        measurements_per_result: int = 2,
        operation_id: str | None = None,
        created: "datetime.datetime | None" = None,
    ) -> tuple[SQLSampleStore, list[MeasurementRequest], list[str]]:
        operation_id = operation_id or random_identifier()
        sample_store = ml_multi_cloud_sample_store

        sql = SQLResourceStore(
            project_context=valid_ado_project_context, ensureExists=True
        )

        resource = OperationResource(
            identifier=operation_id,
            config=ml_multi_cloud_operation_configuration,
            operationType=DiscoveryOperationEnum.EXPLORE,
            operatorIdentifier="doesnt-matter",
        )
        if created is not None:
            resource.created = created
        sql.addResourceWithRelationships(
            resource,
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


@pytest.fixture
def backdate_resource(
    valid_ado_project_context: ProjectContext,
) -> Callable[[str, CoreResourceKinds, datetime.datetime], None]:
    """Return a callable that overwrites the ``created`` timestamp of any resource.

    Args:
        identifier: The resource identifier to backdate.
        kind: The ``CoreResourceKinds`` of the resource.
        created: The new ``created`` timestamp to set.
    """

    def _backdate_resource(
        identifier: str,
        kind: CoreResourceKinds,
        created: datetime.datetime,
    ) -> None:
        sql = SQLStore(project_context=valid_ado_project_context)
        resource = sql.getResource(identifier=identifier, kind=kind)
        resource.created = created
        sql.updateResource(resource)

    return _backdate_resource
