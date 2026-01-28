# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import re
import sqlite3
import uuid
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

import orchestrator.core.datacontainer.resource
import orchestrator.core.discoveryspace.config
import orchestrator.core.discoveryspace.resource
import orchestrator.core.operation.config
import orchestrator.core.samplestore.config
import orchestrator.core.samplestore.resource
import orchestrator.metastore.base
import orchestrator.metastore.sqlstore
import orchestrator.modules.module
import orchestrator.modules.operators.base
import orchestrator.modules.operators.collections
from orchestrator.core.datacontainer.resource import DataContainerResource
from orchestrator.core.discoveryspace.resource import DiscoverySpaceResource
from orchestrator.core.operation.config import DiscoveryOperationResourceConfiguration
from orchestrator.core.operation.resource import OperationResource
from orchestrator.core.resources import (
    ADOResourceEventEnum,
    CoreResourceKinds,
)
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLStore

# Methods to test:
# READ
# getResources -> tested in test_get_resources
# CREATE
# addRelationshipForResources
# addResource
# addResourceWithRelationships


# Test for new resource store

sqlite3_version = sqlite3.sqlite_version_info


def test_get_resources_of_kind(
    resource_store: SQLStore, resource_type: CoreResourceKinds
) -> None:
    """Test can we get resource of the given kind from the resource_store"""

    # AP: the -> and ->> syntax in SQLite is only supported from version 3.38.0
    # ref: https://sqlite.org/json1.html#jptr
    if resource_store.engine.dialect.name == "sqlite" and sqlite3_version < (3, 38, 0):
        pytest.xfail("SQLite version 3.38.0 or higher is required")

    resources = resource_store.getResourcesOfKind(resource_type.value)
    for resource in resources.values():
        assert resource
        assert isinstance(resource, orchestrator.core.kindmap[resource_type.value])

    # Check we retrieved the resources for all the resource ids
    expected_ids = resource_store.getResourceIdentifiersOfKind(resource_type.value)
    assert len(expected_ids.IDENTIFIER) == len(resources.keys())


def test_get_resources_and_get_resource_identifiers_of_kind(
    sql_store_with_resources_preloaded: SQLStore, resource_type: CoreResourceKinds
) -> None:
    """
    Test can we get resource of the given kind from the resource_store of type new"""

    # AP: the -> and ->> syntax in SQLite is only supported from version 3.38.0
    # ref: https://sqlite.org/json1.html#jptr
    if (
        sql_store_with_resources_preloaded.engine.dialect.name == "sqlite"
        and sqlite3_version < (3, 38, 0)
    ):
        pytest.xfail("SQLite version 3.38.0 or higher is required")

    x = sql_store_with_resources_preloaded.getResourceIdentifiersOfKind(
        resource_type.value
    )
    assert isinstance(x, pd.DataFrame)
    assert x.columns[0] == "IDENTIFIER"

    objects = sql_store_with_resources_preloaded.getResources(x["IDENTIFIER"])

    r = [
        isinstance(e, orchestrator.core.kindmap[resource_type.value])
        for e in objects.values()
    ]
    assert False not in r

    # We expect there to be some of the following resource types
    if resource_type in [
        CoreResourceKinds.DISCOVERYSPACE,
        CoreResourceKinds.SAMPLESTORE,
        CoreResourceKinds.OPERATION,
    ]:
        assert len(objects) > 0


def test_get_related_resource_identifiers(
    sql_store_with_resources_preloaded: SQLStore, resource_type: CoreResourceKinds
) -> None:
    """
    Tests getting the identifiers of related resources

    """

    # AP: the -> and ->> syntax in SQLite is only supported from version 3.38.0
    # ref: https://sqlite.org/json1.html#jptr
    if (
        sql_store_with_resources_preloaded.engine.dialect.name == "sqlite"
        and sqlite3_version < (3, 38, 0)
    ):
        pytest.xfail("SQLite version 3.38.0 or higher is required")

    identifiers = sql_store_with_resources_preloaded.getResourceIdentifiersOfKind(
        resource_type.value
    )
    if identifiers.shape[0] > 0:
        identifier = identifiers["IDENTIFIER"][0]
        for rt2 in list(CoreResourceKinds):
            x = sql_store_with_resources_preloaded.getRelatedResourceIdentifiers(
                identifier, kind=rt2.value
            )
            assert x is not None

            # Test some relationships know to exist
            # All operations should have related discovery space
            # All DiscoverySpaces should have related SampleStore
            if (
                resource_type == CoreResourceKinds.OPERATION
                and rt2 == CoreResourceKinds.DISCOVERYSPACE
            ):
                assert x.shape[0] > 0
                assert x.columns[0] == "IDENTIFIER"
                assert x.columns[1] == "TYPE"
                if x.shape[0] > 0:
                    types = np.unique(x["TYPE"].values)
                    assert len(types) == 1
                    assert types[0] == rt2.value
            elif (
                resource_type == CoreResourceKinds.DISCOVERYSPACE
                and rt2 == CoreResourceKinds.SAMPLESTORE
            ):
                assert x.columns[0] == "IDENTIFIER"
                assert x.columns[1] == "TYPE"
                if x.shape[0] > 0:
                    types = np.unique(x["TYPE"].values)
                    assert len(types) == 1
                    assert types[0] == rt2.value


#
# CREATE
#


def test_add_invalid_resource(
    resource_store: SQLStore, operation_resource: OperationResource
) -> None:
    """

    Tests we cannot add non ADOResource models to new store"""

    # Try adding the OperationResource config instead of the actual resource
    with pytest.raises(
        ValueError,
        match=r"Cannot add resource, .*, that is not a subclass of ADOResource",
    ):
        resource_store.addResource(resource=operation_resource.config)


def test_get_resource_identifiers_of_kind_exception_unknown_kind(
    resource_store: SQLStore,
) -> None:

    with pytest.raises(ValueError, match="Unknown kind specified: unknown_kind"):
        resource_store.getResourceIdentifiersOfKind("unknown_kind")


def test_add_and_delete_discovery_space(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    sql_store: SQLStore,
) -> None:
    """Tests adding a discovery space resource"""

    space_resource = random_space_resource_from_db()

    assert space_resource.status[-1].event == ADOResourceEventEnum.ADDED
    assert sql_store.containsResourceWithIdentifier(space_resource.identifier)

    # Test that adding it again raises an error
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Resource with id {space_resource.identifier} already present. "
            f"Use updateResource if you want to overwrite it"
        ),
    ):
        sql_store.addResource(resource=space_resource)

    # Delete it
    sql_store.deleteResource(identifier=space_resource.identifier)

    # Test it's not there
    assert not sql_store.containsResourceWithIdentifier(
        identifier=space_resource.identifier
    )

    assert (
        sql_store.getRelatedResourceIdentifiers(
            identifier=space_resource.identifier
        ).shape[0]
        == 0
    )


def test_add_update_and_delete_operation_related_to_discovery_space(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    sql_store: SQLStore,
    operation_resource: OperationResource,
) -> None:
    """
    Tests adding an operation and its relation to a discovery space and then deleting it
    """

    # AP: the -> and ->> syntax in SQLite is only supported from version 3.38.0
    # ref: https://sqlite.org/json1.html#jptr
    if sql_store.engine.dialect.name == "sqlite" and sqlite3_version < (
        3,
        38,
        0,
    ):
        pytest.xfail("SQLite version 3.38.0 or higher is required")

    space_resource = random_space_resource_from_db()
    space_identifier = space_resource.identifier

    # Add the operation along with a relationship to space_identifier
    sql_store.addResourceWithRelationships(
        operation_resource, relatedIdentifiers=[space_identifier]
    )

    assert operation_resource.status[-1].event == ADOResourceEventEnum.ADDED

    # Test the operation is there
    assert (
        operation_resource.identifier
        in sql_store.getResourceIdentifiersOfKind(
            kind=CoreResourceKinds.OPERATION.value
        )["IDENTIFIER"].values
    )

    # Test the relationship to space_identifier is there
    assert (
        space_identifier
        in sql_store.getRelatedResourceIdentifiers(
            identifier=operation_resource.identifier
        )["IDENTIFIER"].values
    )

    # Test is there in the other direction
    assert (
        operation_resource.identifier
        in sql_store.getRelatedResourceIdentifiers(identifier=space_identifier)[
            "IDENTIFIER"
        ].values
    )

    # Update

    metadata = {
        "new_samples_generated": 10,
        "entities_submitted": 20,
        "experiments_requested": 40,
    }
    operation_resource.metadata = metadata
    # Creating a new model as the behaviour of exclude_unset in model_dump has been observed
    # to exclude data added after model creation (as of pydantic 2.6.3)

    updatedResource = OperationResource(**operation_resource.model_dump())
    sql_store.updateResource(updatedResource)

    # Check the update was made
    resource = sql_store.getResource(
        operation_resource.identifier, kind=CoreResourceKinds.OPERATION
    )  # type: OperationResource
    print(resource.metadata)
    assert resource.metadata["new_samples_generated"] == 10
    assert resource.metadata["entities_submitted"] == 20
    assert resource.metadata["experiments_requested"] == 40
    assert len(resource.status) == 3
    assert resource.status[0].event == ADOResourceEventEnum.CREATED
    assert resource.status[1].event == ADOResourceEventEnum.ADDED
    assert resource.status[2].event == ADOResourceEventEnum.UPDATED

    # Delete
    sql_store.deleteResource(identifier=operation_resource.identifier)

    # Test its gone
    assert (
        operation_resource.identifier
        not in sql_store.getResourceIdentifiersOfKind(
            kind=CoreResourceKinds.OPERATION.value
        )["IDENTIFIER"].values
    )

    assert (
        operation_resource.identifier
        not in sql_store.getRelatedResourceIdentifiers(identifier=space_identifier)[
            "IDENTIFIER"
        ].values
    )

    assert (
        sql_store.getRelatedResourceIdentifiers(
            identifier=operation_resource.identifier
        ).shape[0]
        == 0
    )


def test_add_operation_and_output(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    sql_store: SQLStore,
    random_walk_multicloud_operation_configuration: DiscoveryOperationResourceConfiguration,
    data_container_resource: orchestrator.core.datacontainer.resource.DataContainerResource,
) -> None:

    # AP: the -> and ->> syntax in SQLite is only supported from version 3.38.0
    # ref: https://sqlite.org/json1.html#jptr
    if sql_store.engine.dialect.name == "sqlite" and sqlite3_version < (
        3,
        38,
        0,
    ):
        pytest.xfail("SQLite version 3.38.0 or higher is required")

    import orchestrator.core.resources
    import orchestrator.modules.operators.base

    space_resource = random_space_resource_from_db()
    space_identifier = space_resource.identifier
    random_walk_multicloud_operation_configuration.spaces = [space_identifier]

    op_resource = orchestrator.modules.operators.base.add_operation_and_output_to_metastore(
        operation_resource_configuration=random_walk_multicloud_operation_configuration,
        metastore=sql_store,
        output=orchestrator.modules.operators.base.OperationOutput(
            resources=[data_container_resource]
        ),
    )

    # Test we can get the datacontainer
    dcs = sql_store.getRelatedResourceIdentifiers(
        identifier=op_resource.identifier, kind=CoreResourceKinds.DATACONTAINER.value
    )

    assert (dcs.shape[0]) == 1
    ident = dcs["IDENTIFIER"][0]

    res = sql_store.getResource(identifier=ident, kind=CoreResourceKinds.DATACONTAINER)
    assert isinstance(res, DataContainerResource)
    assert res.identifier == ident
    # Check the datacontainer has two statuses - CREATED and ADDED
    assert len(res.status) == 2
    assert res.status[0].event == ADOResourceEventEnum.CREATED
    assert res.status[1].event == ADOResourceEventEnum.ADDED

    data_container = (
        res.config
    )  # type: orchestrator.core.datacontainer.resource.DataContainer
    for k in data_container.tabularData:
        assert (
            data_container.tabularData[k].data
            == data_container_resource.config.tabularData[k].data
        )

    for k in data_container.locationData:
        assert (
            data_container.locationData[k].url()
            == data_container_resource.config.locationData[k].url()
        )

    for k in data_container.data:
        assert data_container.data[k] == data_container_resource.config.data[k]

    # Test we can delete the datacontainer
    sql_store.deleteResource(identifier=res.identifier)

    # Test its gone
    assert (
        res.identifier
        not in sql_store.getResourceIdentifiersOfKind(
            kind=CoreResourceKinds.DATACONTAINER.value
        )["IDENTIFIER"].values
    )

    assert (
        res.identifier
        not in sql_store.getRelatedResourceIdentifiers(
            identifier=op_resource.identifier
        )["IDENTIFIER"].values
    )

    # Delete the resource
    sql_store.deleteResource(identifier=op_resource.identifier)


def test_add_resource_and_relationship_exception_if_resource_does_not_exist(
    resource_store: SQLStore, operation_resource: OperationResource
) -> None:
    """
    - Test if the resource doesn't exist a value error is raised
    """

    fake_identifier = f"space-pytest-fake-{str(uuid.uuid4())[:6]}"

    with pytest.raises(
        ValueError,
        match=f"Unknown resource identifier passed {re.escape(str([fake_identifier]))}",
    ):
        resource_store.addResourceWithRelationships(
            operation_resource,
            relatedIdentifiers=[fake_identifier],
        )


def test_delete_unknown_resource_raise_exception(resource_store: SQLStore) -> None:

    fake_identifier = f"space-pytest-fake-{str(uuid.uuid4())[:6]}"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Cannot delete resource with id {fake_identifier} - it is not present"
        ),
    ):
        resource_store.deleteResource(identifier=fake_identifier)


### Custom Serializations


def test_custom_sample_store_dump(
    active_contest_test_sample_store_resource: orchestrator.core.samplestore.resource.SampleStoreResource,
) -> None:
    """Tests that the custom dumper removes storage location information from sample store
    model dict"""

    assert (
        active_contest_test_sample_store_resource.config.specification.storageLocation
        is not None
    )

    # Return JSON serialization
    custom = orchestrator.metastore.base.kind_custom_model_dump[
        active_contest_test_sample_store_resource.kind.value
    ](active_contest_test_sample_store_resource)

    import json

    custom = json.loads(custom)

    assert custom["config"]["specification"].get("storageLocation") is None


def test_custom_sample_store_loading(
    active_contest_test_sample_store_resource: orchestrator.core.samplestore.resource.SampleStoreResource,
    ado_test_file_project_context: ProjectContext,
) -> None:
    """Tests that the custom loader inserts the given storage location information into a sample store
    model dict that does not have storage location"""

    custom = orchestrator.metastore.base.kind_custom_model_dump[
        active_contest_test_sample_store_resource.kind.value
    ](active_contest_test_sample_store_resource)

    import json

    custom = json.loads(custom)

    assert custom["config"]["specification"].get("storageLocation") is None

    model = orchestrator.metastore.base.kind_custom_model_load[
        active_contest_test_sample_store_resource.kind.value
    ](
        custom, ado_test_file_project_context.metadataStore
    )  # type: orchestrator.core.samplestore.resource.SampleStoreResource

    assert (
        model.config.specification.storageLocation
        == ado_test_file_project_context.metadataStore
    )
