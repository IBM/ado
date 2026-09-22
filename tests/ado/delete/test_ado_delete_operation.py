# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core import DataContainerResource
from ado.core.datacontainer.resource import DataContainer
from ado.core.discoveryspace.config import DiscoverySpaceConfiguration
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.resource import OperationResource
from ado.core.resources import CoreResourceKinds
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore
from ado.schema.experiment import Experiment
from ado.schema.request import (
    MeasurementRequest,
    MeasurementRequestStateEnum,
    ReplayedMeasurement,
)
from tests.conftest import requires_sqlite_3_38


@requires_sqlite_3_38
def test_delete_ml_multi_cloud_operation(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
    ml_multi_cloud_benchmark_performance_experiment: Experiment,
    random_ml_multi_cloud_benchmark_performance_measurement_requests: Callable[
        [int, int, MeasurementRequestStateEnum | None, str | None],
        ReplayedMeasurement,
    ],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_identifier: Callable[[], str],
) -> None:
    assert ml_multi_cloud_benchmark_performance_experiment is not None
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_entities = 3
    number_requests = 3
    measurements_per_result = 2
    operation_id = random_identifier()

    sample_store, _, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=number_requests,
        measurements_per_result=measurements_per_result,
        operation_id=operation_id,
    )

    # Check expected status for the setup
    assert (
        sample_store.measurement_requests_count_for_operation(operation_id=operation_id)
        == number_requests
    )
    assert (
        sample_store.measurement_results_count_for_operation(operation_id=operation_id)
        == number_requests * number_entities
    )

    # Delete the operation
    result = runner.invoke(
        ado,
        [
            "delete",
            "operation",
            operation_id,
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        sample_store.measurement_requests_count_for_operation(operation_id=operation_id)
        == 0
    )
    assert (
        sample_store.measurement_results_count_for_operation(operation_id=operation_id)
        == 0
    )


# ---------------------------------------------------------------------------
# Datacontainer cascading deletion tests
# ---------------------------------------------------------------------------


@requires_sqlite_3_38
def test_delete_operation_with_datacontainer_child_cascades(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
    data_container_resource: DataContainerResource,
    random_sql_sample_store: Callable[[], SQLSampleStore],
    discovery_space_configuration: DiscoverySpaceConfiguration,
    create_space: Callable[[DiscoverySpaceConfiguration, str], DiscoverySpace],
    create_operation: Callable[[DiscoverySpace], OperationResource],
) -> None:
    """Deleting an operation with a single DataContainer child removes both atomically."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    sample_store = random_sql_sample_store()
    space = create_space(discovery_space_configuration, sample_store.identifier)
    operation = create_operation(space)
    op_id = operation.identifier

    # Link the DataContainer as a child of the operation
    sql_store.addResourceWithRelationships(
        data_container_resource,
        relatedIdentifiers=[op_id],
    )
    dc_id = data_container_resource.identifier

    # Both resources exist before delete
    assert sql_store.containsResourceWithIdentifier(
        identifier=op_id, kind=CoreResourceKinds.OPERATION
    )
    assert sql_store.containsResourceWithIdentifier(
        identifier=dc_id, kind=CoreResourceKinds.DATACONTAINER
    )

    result = runner.invoke(ado, ["delete", "operation", op_id, "--force"])
    assert result.exit_code == 0, result.output

    # Both resources must be gone after the cascade delete
    assert not sql_store.containsResourceWithIdentifier(
        identifier=op_id, kind=CoreResourceKinds.OPERATION
    )
    assert not sql_store.containsResourceWithIdentifier(
        identifier=dc_id, kind=CoreResourceKinds.DATACONTAINER
    )


@requires_sqlite_3_38
def test_delete_operation_with_non_datacontainer_child_raises(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
    random_sql_sample_store: Callable[[], SQLSampleStore],
    discovery_space_configuration: DiscoverySpaceConfiguration,
    create_space: Callable[[DiscoverySpaceConfiguration, str], DiscoverySpace],
    create_operation: Callable[[DiscoverySpace], OperationResource],
) -> None:
    """Deleting an operation whose child is a DiscoverySpace must fail and leave both intact."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    sample_store = random_sql_sample_store()
    space = create_space(discovery_space_configuration, sample_store.identifier)
    operation = create_operation(space)
    op_id = operation.identifier

    # Link an additional DiscoverySpace (non-DataContainer) as a child of the operation.
    extra_sample_store = random_sql_sample_store()
    extra_space = create_space(
        discovery_space_configuration, extra_sample_store.identifier
    )
    sql_store.addRelationship(
        subjectIdentifier=op_id,
        objectIdentifier=extra_space.uri,
    )
    extra_space_id = extra_space.uri

    result = runner.invoke(ado, ["delete", "operation", op_id, "--force"])
    assert result.exit_code != 0

    # Both resources must still exist
    assert sql_store.containsResourceWithIdentifier(
        identifier=op_id, kind=CoreResourceKinds.OPERATION
    )
    assert sql_store.containsResourceWithIdentifier(
        identifier=extra_space_id, kind=CoreResourceKinds.DISCOVERYSPACE
    )


@requires_sqlite_3_38
def test_delete_operation_with_datacontainer_having_grandchildren_raises(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
    data_container_resource: DataContainerResource,
    random_sql_sample_store: Callable[[], SQLSampleStore],
    testTabularDataString: object,
    test_sample_store_location: object,
    discovery_space_configuration: DiscoverySpaceConfiguration,
    create_space: Callable[[DiscoverySpaceConfiguration, str], DiscoverySpace],
    create_operation: Callable[[DiscoverySpace], OperationResource],
) -> None:
    """Deleting an operation whose DataContainer child itself has children must fail."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    sample_store = random_sql_sample_store()
    space = create_space(discovery_space_configuration, sample_store.identifier)
    operation = create_operation(space)
    op_id = operation.identifier

    # Link the DataContainer as a child of the operation
    sql_store.addResourceWithRelationships(
        data_container_resource,
        relatedIdentifiers=[op_id],
    )
    dc_id = data_container_resource.identifier

    # Create a grandchild DataContainer and link it as a child of the first DataContainer
    grandchild_dc = DataContainerResource(
        config=DataContainer(
            tabularData={"entities": testTabularDataString},
            data={"key": "value"},
            locationData={"loc": test_sample_store_location},
        )
    )
    sql_store.addResourceWithRelationships(
        grandchild_dc,
        relatedIdentifiers=[dc_id],
    )
    grandchild_id = grandchild_dc.identifier

    result = runner.invoke(ado, ["delete", "operation", op_id, "--force"])
    assert result.exit_code != 0

    # All three resources must still exist
    assert sql_store.containsResourceWithIdentifier(
        identifier=op_id, kind=CoreResourceKinds.OPERATION
    )
    assert sql_store.containsResourceWithIdentifier(
        identifier=dc_id, kind=CoreResourceKinds.DATACONTAINER
    )
    assert sql_store.containsResourceWithIdentifier(
        identifier=grandchild_id, kind=CoreResourceKinds.DATACONTAINER
    )
