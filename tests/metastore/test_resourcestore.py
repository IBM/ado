# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import re
import uuid
from collections.abc import Callable

import pandas as pd
import pytest

import ado.core.datacontainer.resource
import ado.core.discoveryspace.config
import ado.core.discoveryspace.resource
import ado.core.operation.config
import ado.core.samplestore.config
import ado.core.samplestore.resource
import ado.metastore.base
import ado.metastore.sqlstore
import ado.modules.module
import ado.modules.operators.base
import ado.modules.operators.collections
from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.resource import DiscoverySpaceResource
from ado.core.document.config import DocumentConfiguration
from ado.core.document.resource import DocumentResource
from ado.core.operation.config import DiscoveryOperationResourceConfiguration
from ado.core.operation.resource import OperationResource
from ado.core.resources import (
    ADOResourceEventEnum,
    ADOResourceReference,
    CoreResourceKinds,
)
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore
from tests.conftest import requires_sqlite_3_38

# Methods to test:
# READ
# getResources -> tested in test_get_resources
# CREATE
# addRelationshipForResources
# addResource
# addResourceWithRelationships


# Test for new resource store


@requires_sqlite_3_38
def test_get_resources_of_kind(
    resource_store: SQLStore, resource_type: CoreResourceKinds
) -> None:
    """Test can we get resource of the given kind from the resource_store"""

    resources = resource_store.getResourcesOfKind(resource_type.value)
    for resource in resources.values():
        assert resource
        assert isinstance(resource, ado.core.kindmap[resource_type.value])

    # Check we retrieved the resources for all the resource ids
    expected_ids = resource_store.getResourceIdentifiersOfKind(resource_type.value)
    assert len(expected_ids.IDENTIFIER) == len(resources.keys())


@requires_sqlite_3_38
def test_get_resources_and_get_resource_identifiers_of_kind(
    sql_store_with_resources_preloaded: SQLStore, resource_type: CoreResourceKinds
) -> None:
    """
    Test can we get resource of the given kind from the resource_store of type new"""

    x = sql_store_with_resources_preloaded.getResourceIdentifiersOfKind(
        resource_type.value
    )
    assert isinstance(x, pd.DataFrame)
    assert x.columns[0] == "IDENTIFIER"

    objects = sql_store_with_resources_preloaded.getResources(x["IDENTIFIER"])

    r = [isinstance(e, ado.core.kindmap[resource_type.value]) for e in objects.values()]
    assert False not in r

    # We expect there to be some of the following resource types
    if resource_type in [
        CoreResourceKinds.DISCOVERYSPACE,
        CoreResourceKinds.SAMPLESTORE,
        CoreResourceKinds.OPERATION,
    ]:
        assert len(objects) > 0


@requires_sqlite_3_38
def test_get_resources_sorted_by_created_ascending(
    sql_store: SQLStore,
    random_space_resource_from_file: Callable[[str | None], DiscoverySpaceResource],
    create_resources: Callable[[list[ado.core.resources.ADOResource], SQLStore], None],
) -> None:
    """Test that getResources returns resources sorted by created timestamp in ascending order (oldest first)."""
    import datetime

    # Create multiple resources from file with explicit created timestamps
    space1 = random_space_resource_from_file()
    space1.created = datetime.datetime(
        2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )

    space2 = random_space_resource_from_file()
    space2.created = datetime.datetime(
        2024, 1, 2, 12, 0, 0, tzinfo=datetime.timezone.utc
    )

    space3 = random_space_resource_from_file()
    space3.created = datetime.datetime(
        2024, 1, 3, 12, 0, 0, tzinfo=datetime.timezone.utc
    )

    # Add resources to database using create_resources fixture
    create_resources([space1, space2, space3])

    # Get all three resources
    identifiers = [space1.identifier, space2.identifier, space3.identifier]
    resources = sql_store.getResources(identifiers)

    # Convert to list to check order
    resource_list = list(resources.values())

    # Verify we got all three resources
    assert len(resource_list) == 3

    # Verify they are sorted by created timestamp in ascending order (oldest first)
    assert resource_list[0].created <= resource_list[1].created
    assert resource_list[1].created <= resource_list[2].created

    # Verify the oldest is space1 and most recent is space3
    assert resource_list[0].identifier == space1.identifier
    assert resource_list[2].identifier == space3.identifier


@requires_sqlite_3_38
def test_get_resources_by_relationship(
    sql_store_with_resources_preloaded: SQLStore, resource_type: CoreResourceKinds
) -> None:
    """
    Tests getting the identifiers of related resources via get_resources_by_relationship.
    """

    identifiers = sql_store_with_resources_preloaded.getResourceIdentifiersOfKind(
        resource_type.value
    )
    if identifiers.shape[0] > 0:
        identifier = identifiers["IDENTIFIER"][0]
        result = sql_store_with_resources_preloaded.get_resources_by_relationship(
            kind=resource_type,
            identifier=identifier,
            relationship="both",
            max_hops=None,
            identifiers_only=True,
        )
        assert result is not None

        # Test some relationships known to exist:
        # All operations should have a related discovery space
        if resource_type == CoreResourceKinds.OPERATION:
            assert CoreResourceKinds.DISCOVERYSPACE in result
            assert len(result[CoreResourceKinds.DISCOVERYSPACE]) == 1

        # All DiscoverySpaces should have a related SampleStore
        if resource_type == CoreResourceKinds.DISCOVERYSPACE:
            assert CoreResourceKinds.SAMPLESTORE in result
            assert len(result[CoreResourceKinds.SAMPLESTORE]) == 1


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

    assert not sql_store.get_resources_by_relationship(
        kind=CoreResourceKinds.DISCOVERYSPACE,
        identifier=space_resource.identifier,
        relationship="both",
        max_hops=None,
        identifiers_only=True,
    )


@requires_sqlite_3_38
def test_add_update_and_delete_operation_related_to_discovery_space(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    sql_store: SQLStore,
    operation_resource: OperationResource,
) -> None:
    """
    Tests adding an operation and its relation to a discovery space and then deleting it
    """

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

    # Test the relationship to space_identifier is there (operation → space: up)
    assert space_identifier in sql_store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=operation_resource.identifier,
        relationship="parent",
        max_hops=1,
        identifiers_only=True,
    ).get(CoreResourceKinds.DISCOVERYSPACE, set())

    # Test is there in the other direction (space → operation: down)
    assert operation_resource.identifier in sql_store.get_resources_by_relationship(
        kind=CoreResourceKinds.DISCOVERYSPACE,
        identifier=space_identifier,
        relationship="child",
        max_hops=1,
        identifiers_only=True,
    ).get(CoreResourceKinds.OPERATION, set())

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

    assert operation_resource.identifier not in sql_store.get_resources_by_relationship(
        kind=CoreResourceKinds.DISCOVERYSPACE,
        identifier=space_identifier,
        relationship="child",
        max_hops=1,
        identifiers_only=True,
    ).get(CoreResourceKinds.OPERATION, set())

    assert not sql_store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=operation_resource.identifier,
        relationship="both",
        max_hops=None,
        identifiers_only=True,
    )


@requires_sqlite_3_38
def test_add_operation_and_output(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    sql_store: SQLStore,
    random_walk_multicloud_operation_configuration: DiscoveryOperationResourceConfiguration,
    data_container_resource: ado.core.datacontainer.resource.DataContainerResource,
) -> None:

    space_resource = random_space_resource_from_db()
    space_identifier = space_resource.identifier
    random_walk_multicloud_operation_configuration.inputs["discoverySpace"] = (
        ADOResourceReference(
            identifier=space_identifier,
            kind=CoreResourceKinds.DISCOVERYSPACE,
        )
    )

    op_resource = ado.modules.operators.base.add_operation_and_output_to_metastore(
        operation_resource_configuration=random_walk_multicloud_operation_configuration,
        metastore=sql_store,
        output=ado.modules.operators.base.OperationOutput(
            resources=[data_container_resource]
        ),
    )

    # Test we can get the datacontainer
    dcs = sql_store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=op_resource.identifier,
        relationship="child",
        max_hops=1,
        identifiers_only=True,
    ).get(CoreResourceKinds.DATACONTAINER, set())

    assert len(dcs) == 1
    ident = next(iter(dcs))

    res = sql_store.getResource(identifier=ident, kind=CoreResourceKinds.DATACONTAINER)
    assert isinstance(res, DataContainerResource)
    assert res.identifier == ident
    # Check the datacontainer has two statuses - CREATED and ADDED
    assert len(res.status) == 2
    assert res.status[0].event == ADOResourceEventEnum.CREATED
    assert res.status[1].event == ADOResourceEventEnum.ADDED

    data_container = res.config  # type: ado.core.datacontainer.resource.DataContainer
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

    assert res.identifier not in sql_store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=op_resource.identifier,
        relationship="child",
        max_hops=1,
        identifiers_only=True,
    ).get(CoreResourceKinds.DATACONTAINER, set())

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
    active_contest_test_sample_store_resource: ado.core.samplestore.resource.SampleStoreResource,
) -> None:
    """Tests that the custom dumper removes storage location information from sample store
    model dict"""

    assert (
        active_contest_test_sample_store_resource.config.specification.storageLocation
        is not None
    )

    # Return JSON serialization
    custom = ado.metastore.base.kind_custom_model_dump[
        active_contest_test_sample_store_resource.kind.value
    ](active_contest_test_sample_store_resource)

    import json

    custom = json.loads(custom)

    assert custom["config"]["specification"].get("storageLocation") is None


def test_custom_sample_store_loading(
    active_contest_test_sample_store_resource: ado.core.samplestore.resource.SampleStoreResource,
    ado_test_file_project_context: ProjectContext,
) -> None:
    """Tests that the custom loader inserts the given storage location information into a sample store
    model dict that does not have storage location"""

    custom = ado.metastore.base.kind_custom_model_dump[
        active_contest_test_sample_store_resource.kind.value
    ](active_contest_test_sample_store_resource)

    import json

    custom = json.loads(custom)

    assert custom["config"]["specification"].get("storageLocation") is None

    model = ado.metastore.base.kind_custom_model_load[
        active_contest_test_sample_store_resource.kind.value
    ](custom, ado_test_file_project_context.metadataStore)  # type: ado.core.samplestore.resource.SampleStoreResource

    assert (
        model.config.specification.storageLocation
        == ado_test_file_project_context.metadataStore
    )


@requires_sqlite_3_38
def test_get_latest_resource_identifiers_of_kinds_empty_database(
    resource_store: SQLStore,
) -> None:
    """Test get_latest_resource_identifiers_of_kinds with empty database returns empty dict."""

    result = resource_store.get_latest_resource_identifiers_of_kinds(
        kinds=[CoreResourceKinds.DISCOVERYSPACE, CoreResourceKinds.OPERATION]
    )

    assert isinstance(result, dict)
    assert len(result) == 0


@requires_sqlite_3_38
def test_get_latest_resource_identifiers_of_kinds_single_kind(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    resource_store: SQLStore,
) -> None:
    """Test get_latest_resource_identifiers_of_kinds with single kind returns correct identifier."""

    # Create a space resource
    space = random_space_resource_from_db(None)

    # Query for latest discoveryspace
    result = resource_store.get_latest_resource_identifiers_of_kinds(
        kinds=[CoreResourceKinds.DISCOVERYSPACE]
    )

    assert isinstance(result, dict)
    assert len(result) == 1
    assert CoreResourceKinds.DISCOVERYSPACE in result
    assert result[CoreResourceKinds.DISCOVERYSPACE] == space.identifier


@requires_sqlite_3_38
def test_get_latest_resource_identifiers_of_kinds_multiple_kinds(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    resource_store: SQLStore,
    operation_resource: OperationResource,
) -> None:
    """Test get_latest_resource_identifiers_of_kinds with multiple kinds in single query."""

    # Create a space resource
    space = random_space_resource_from_db(None)

    # Add an operation resource
    resource_store.addResource(operation_resource)

    # Query for both kinds in single batch query
    result = resource_store.get_latest_resource_identifiers_of_kinds(
        kinds=[CoreResourceKinds.DISCOVERYSPACE, CoreResourceKinds.OPERATION]
    )

    assert isinstance(result, dict)
    assert len(result) == 2
    assert CoreResourceKinds.DISCOVERYSPACE in result
    assert CoreResourceKinds.OPERATION in result
    assert result[CoreResourceKinds.DISCOVERYSPACE] == space.identifier
    assert result[CoreResourceKinds.OPERATION] == operation_resource.identifier


@requires_sqlite_3_38
def test_get_latest_resource_identifiers_of_kinds_multiple_resources_same_kind(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    resource_store: SQLStore,
) -> None:
    """Test get_latest_resource_identifiers_of_kinds returns most recent when multiple resources of same kind exist."""

    import time

    # Create first space
    _space1 = random_space_resource_from_db(None)

    # Small delay to ensure different timestamps
    time.sleep(0.1)

    # Create second space (should be more recent)
    space2 = random_space_resource_from_db(None)

    # Query for latest discoveryspace
    result = resource_store.get_latest_resource_identifiers_of_kinds(
        kinds=[CoreResourceKinds.DISCOVERYSPACE]
    )

    assert isinstance(result, dict)
    assert len(result) == 1
    assert CoreResourceKinds.DISCOVERYSPACE in result
    # Should return the most recently created space
    assert result[CoreResourceKinds.DISCOVERYSPACE] == space2.identifier, (
        f"Previous one was {_space1.identifier}"
    )


@requires_sqlite_3_38
def test_get_latest_resource_identifiers_of_kinds_some_kinds_missing(
    random_space_resource_from_db: Callable[[str | None], DiscoverySpaceResource],
    resource_store: SQLStore,
) -> None:
    """Test get_latest_resource_identifiers_of_kinds omits kinds with no resources."""

    # Create only a space resource
    space = random_space_resource_from_db(None)

    # Query for both space and operation, but only space exists
    result = resource_store.get_latest_resource_identifiers_of_kinds(
        kinds=[CoreResourceKinds.DISCOVERYSPACE, CoreResourceKinds.OPERATION]
    )

    assert isinstance(result, dict)
    assert len(result) == 1
    assert CoreResourceKinds.DISCOVERYSPACE in result
    assert CoreResourceKinds.OPERATION not in result
    assert result[CoreResourceKinds.DISCOVERYSPACE] == space.identifier


@requires_sqlite_3_38
def test_get_latest_resource_identifiers_of_kinds_invalid_kind(
    resource_store: SQLStore,
) -> None:
    """Test get_latest_resource_identifiers_of_kinds raises ValueError for invalid kind."""

    # This should raise ValueError because we're passing an invalid kind
    # Note: This test assumes the method validates kinds before querying
    invalid_kind = "invalid_kind"  # type: ignore[arg-type]
    with pytest.raises(
        ValueError, match="All kinds must be CoreResourceKinds instances"
    ):
        resource_store.get_latest_resource_identifiers_of_kinds(kinds=[invalid_kind])


###############################################################################
# get_resources_by_relationship
###############################################################################

# ---------------------------------------------------------------------------
# Helper fixture: a complete samplestore → discoveryspace → operation →
#                 datacontainer / actuatorconfiguration hierarchy
# ---------------------------------------------------------------------------


@pytest.fixture
def resource_hierarchy(
    sql_store: SQLStore,
    random_space_resource_from_file: Callable[[str | None], DiscoverySpaceResource],
    operation_resource: OperationResource,
    data_container_resource: ado.core.datacontainer.resource.DataContainerResource,
) -> dict:
    """Build and persist a minimal linked hierarchy.

    Returns a dict with keys:
        samplestore_id, discoveryspace_id, operation_id,
        datacontainer_id, actuatorconfiguration_id, store
    """
    import pathlib

    import yaml

    from ado.core import ActuatorConfigurationResource, SampleStoreResource
    from ado.core.samplestore.config import (
        SampleStoreConfiguration,
        SampleStoreModuleConf,
        SampleStoreSpecification,
    )

    # 1. samplestore
    ss = SampleStoreResource(
        config=SampleStoreConfiguration(
            specification=SampleStoreSpecification(
                module=SampleStoreModuleConf(
                    moduleClass="SQLSampleStore",
                    moduleName="ado.core.samplestore.sql",
                )
            )
        )
    )
    sql_store.addResource(ss)

    # 2. discoveryspace (child of samplestore)
    ds = random_space_resource_from_file(sample_store_id=ss.identifier)
    sql_store.addResourceWithRelationships(ds, relatedIdentifiers=[ss.identifier])

    # 3. operation (child of discoveryspace)
    operation_resource.config.inputs["discoverySpace"] = ADOResourceReference(
        identifier=ds.identifier,
        kind=CoreResourceKinds.DISCOVERYSPACE,
    )
    sql_store.addResourceWithRelationships(
        operation_resource, relatedIdentifiers=[ds.identifier]
    )

    # 4. datacontainer (child of operation)
    sql_store.addResourceWithRelationships(
        data_container_resource,
        relatedIdentifiers=[operation_resource.identifier],
    )

    # 5. actuatorconfiguration — stored as subject, operation as object,
    #    matching the production path in addResourceWithRelationships(operation,
    #    relatedIdentifiers=[..., actconf_id]).
    ac_config = (
        ado.core.actuatorconfiguration.config.ActuatorConfiguration.model_validate(
            yaml.safe_load(
                pathlib.Path(
                    "tests/resources/replay_actuatorconfiguration.yaml"
                ).read_text()
            )
        )
    )
    ac = ActuatorConfigurationResource(config=ac_config)
    sql_store.addResource(ac)
    sql_store.addRelationship(
        subjectIdentifier=ac.identifier,
        objectIdentifier=operation_resource.identifier,
    )

    return {
        "samplestore_id": ss.identifier,
        "discoveryspace_id": ds.identifier,
        "operation_id": operation_resource.identifier,
        "datacontainer_id": data_container_resource.identifier,
        "actuatorconfiguration_id": ac.identifier,
        "store": sql_store,
    }


# ---------------------------------------------------------------------------
# up from operation
# ---------------------------------------------------------------------------


@requires_sqlite_3_38
def test_up_from_operation_max_hops_1(
    resource_hierarchy: dict,
) -> None:
    """up from operation with max_hops=1 returns only discoveryspace ids."""
    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]
    ds_id = resource_hierarchy["discoveryspace_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=op_id,
        relationship="parent",
        max_hops=1,
        identifiers_only=True,
    )

    assert CoreResourceKinds.DISCOVERYSPACE in result
    assert ds_id in result[CoreResourceKinds.DISCOVERYSPACE]
    assert CoreResourceKinds.SAMPLESTORE not in result


@requires_sqlite_3_38
def test_up_from_operation_uncapped_returns_discoveryspace_and_samplestore(
    resource_hierarchy: dict,
) -> None:
    """up from operation without cap returns discoveryspace and samplestore ids."""
    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]
    ds_id = resource_hierarchy["discoveryspace_id"]
    ss_id = resource_hierarchy["samplestore_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=op_id,
        relationship="parent",
        identifiers_only=True,
    )

    assert CoreResourceKinds.DISCOVERYSPACE in result
    assert ds_id in result[CoreResourceKinds.DISCOVERYSPACE]
    assert CoreResourceKinds.SAMPLESTORE in result
    assert ss_id in result[CoreResourceKinds.SAMPLESTORE]


# ---------------------------------------------------------------------------
# up from discoveryspace
# ---------------------------------------------------------------------------


@requires_sqlite_3_38
def test_up_from_discoveryspace_returns_samplestore_only(
    resource_hierarchy: dict,
) -> None:
    """up from discoveryspace returns only samplestore identifiers."""
    store: SQLStore = resource_hierarchy["store"]
    ds_id = resource_hierarchy["discoveryspace_id"]
    ss_id = resource_hierarchy["samplestore_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.DISCOVERYSPACE,
        identifier=ds_id,
        relationship="parent",
        identifiers_only=True,
    )

    assert list(result.keys()) == [CoreResourceKinds.SAMPLESTORE]
    assert ss_id in result[CoreResourceKinds.SAMPLESTORE]


# ---------------------------------------------------------------------------
# down from samplestore
# ---------------------------------------------------------------------------


@requires_sqlite_3_38
def test_down_from_samplestore_max_hops_1(
    resource_hierarchy: dict,
) -> None:
    """down from samplestore with max_hops=1 returns only discoveryspace ids."""
    store: SQLStore = resource_hierarchy["store"]
    ss_id = resource_hierarchy["samplestore_id"]
    ds_id = resource_hierarchy["discoveryspace_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.SAMPLESTORE,
        identifier=ss_id,
        relationship="child",
        max_hops=1,
        identifiers_only=True,
    )

    assert CoreResourceKinds.DISCOVERYSPACE in result
    assert ds_id in result[CoreResourceKinds.DISCOVERYSPACE]
    assert CoreResourceKinds.OPERATION not in result
    assert CoreResourceKinds.DATACONTAINER not in result
    assert CoreResourceKinds.ACTUATORCONFIGURATION not in result


@requires_sqlite_3_38
def test_down_from_samplestore_max_hops_2(
    resource_hierarchy: dict,
) -> None:
    """down from samplestore with max_hops=2 returns discoveryspace and operation ids."""
    store: SQLStore = resource_hierarchy["store"]
    ss_id = resource_hierarchy["samplestore_id"]
    op_id = resource_hierarchy["operation_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.SAMPLESTORE,
        identifier=ss_id,
        relationship="child",
        max_hops=2,
        identifiers_only=True,
    )

    assert CoreResourceKinds.DISCOVERYSPACE in result
    assert CoreResourceKinds.OPERATION in result
    assert op_id in result[CoreResourceKinds.OPERATION]
    assert CoreResourceKinds.DATACONTAINER not in result
    assert CoreResourceKinds.ACTUATORCONFIGURATION not in result


@requires_sqlite_3_38
def test_outgoing_from_samplestore_uncapped_returns_discoveryspace_operation_and_datacontainer(
    resource_hierarchy: dict,
) -> None:
    """outgoing from samplestore returns resources reachable via stored edge direction."""
    store: SQLStore = resource_hierarchy["store"]
    ss_id = resource_hierarchy["samplestore_id"]
    dc_id = resource_hierarchy["datacontainer_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.SAMPLESTORE,
        identifier=ss_id,
        relationship="child",
        identifiers_only=True,
    )

    assert CoreResourceKinds.DISCOVERYSPACE in result
    assert CoreResourceKinds.OPERATION in result
    assert CoreResourceKinds.DATACONTAINER in result
    assert CoreResourceKinds.ACTUATORCONFIGURATION not in result
    assert dc_id in result[CoreResourceKinds.DATACONTAINER]


# ---------------------------------------------------------------------------
# outgoing from discoveryspace
# ---------------------------------------------------------------------------


@requires_sqlite_3_38
def test_outgoing_from_discoveryspace_max_hops_1(
    resource_hierarchy: dict,
) -> None:
    """outgoing from discoveryspace with max_hops=1 returns only operation ids."""
    store: SQLStore = resource_hierarchy["store"]
    ds_id = resource_hierarchy["discoveryspace_id"]
    op_id = resource_hierarchy["operation_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.DISCOVERYSPACE,
        identifier=ds_id,
        relationship="child",
        max_hops=1,
        identifiers_only=True,
    )

    assert CoreResourceKinds.OPERATION in result
    assert op_id in result[CoreResourceKinds.OPERATION]
    assert CoreResourceKinds.DATACONTAINER not in result
    assert CoreResourceKinds.ACTUATORCONFIGURATION not in result


@requires_sqlite_3_38
def test_outgoing_from_discoveryspace_uncapped_returns_operation_and_datacontainer(
    resource_hierarchy: dict,
) -> None:
    """outgoing from discoveryspace returns resources reachable via stored edge direction."""
    store: SQLStore = resource_hierarchy["store"]
    ds_id = resource_hierarchy["discoveryspace_id"]
    dc_id = resource_hierarchy["datacontainer_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.DISCOVERYSPACE,
        identifier=ds_id,
        relationship="child",
        identifiers_only=True,
    )

    assert CoreResourceKinds.OPERATION in result
    assert CoreResourceKinds.DATACONTAINER in result
    assert CoreResourceKinds.ACTUATORCONFIGURATION not in result
    assert dc_id in result[CoreResourceKinds.DATACONTAINER]


# ---------------------------------------------------------------------------
# outgoing from operation
# ---------------------------------------------------------------------------


@requires_sqlite_3_38
def test_outgoing_from_operation_returns_datacontainer_only(
    resource_hierarchy: dict,
) -> None:
    """outgoing from operation returns only resources reachable via stored edge direction."""
    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]
    dc_id = resource_hierarchy["datacontainer_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=op_id,
        relationship="child",
        identifiers_only=True,
    )

    assert CoreResourceKinds.DATACONTAINER in result
    assert CoreResourceKinds.ACTUATORCONFIGURATION not in result
    assert dc_id in result[CoreResourceKinds.DATACONTAINER]


@requires_sqlite_3_38
def test_both_from_operation_returns_ancestors_and_descendants(
    resource_hierarchy: dict,
) -> None:
    """both from operation returns discoveryspace, samplestore, datacontainer and actuatorconfiguration."""
    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]
    ds_id = resource_hierarchy["discoveryspace_id"]
    ss_id = resource_hierarchy["samplestore_id"]
    dc_id = resource_hierarchy["datacontainer_id"]
    ac_id = resource_hierarchy["actuatorconfiguration_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=op_id,
        identifiers_only=True,
    )

    assert ds_id in result[CoreResourceKinds.DISCOVERYSPACE]
    assert ss_id in result[CoreResourceKinds.SAMPLESTORE]
    assert dc_id in result[CoreResourceKinds.DATACONTAINER]
    assert ac_id in result[CoreResourceKinds.ACTUATORCONFIGURATION]


@requires_sqlite_3_38
def test_incoming_from_datacontainer_returns_operation_discoveryspace_and_samplestore(
    resource_hierarchy: dict,
) -> None:
    """incoming from datacontainer returns operation, discoveryspace and samplestore."""
    store: SQLStore = resource_hierarchy["store"]
    dc_id = resource_hierarchy["datacontainer_id"]
    op_id = resource_hierarchy["operation_id"]
    ds_id = resource_hierarchy["discoveryspace_id"]
    ss_id = resource_hierarchy["samplestore_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.DATACONTAINER,
        identifier=dc_id,
        relationship="parent",
        identifiers_only=True,
    )

    assert op_id in result[CoreResourceKinds.OPERATION]
    assert ds_id in result[CoreResourceKinds.DISCOVERYSPACE]
    assert ss_id in result[CoreResourceKinds.SAMPLESTORE]


@requires_sqlite_3_38
def test_both_from_datacontainer_reaches_ancestors_and_siblings(
    resource_hierarchy: dict,
) -> None:
    """both from datacontainer reaches ancestors and siblings via the shared operation.

    The open-graph traversal follows all stored edges in both directions, so
    from a datacontainer it goes up to the parent operation and then down to the
    actuatorconfiguration that shares the same operation.  This is the expected
    behaviour with the raw-graph traversal.
    """
    store: SQLStore = resource_hierarchy["store"]
    dc_id = resource_hierarchy["datacontainer_id"]
    op_id = resource_hierarchy["operation_id"]
    ds_id = resource_hierarchy["discoveryspace_id"]
    ss_id = resource_hierarchy["samplestore_id"]
    ac_id = resource_hierarchy["actuatorconfiguration_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.DATACONTAINER,
        identifier=dc_id,
        identifiers_only=True,
    )

    assert op_id in result[CoreResourceKinds.OPERATION]
    assert ds_id in result[CoreResourceKinds.DISCOVERYSPACE]
    assert ss_id in result[CoreResourceKinds.SAMPLESTORE]
    # Sibling actuatorconfiguration is now reachable via dc → op → ac
    assert ac_id in result[CoreResourceKinds.ACTUATORCONFIGURATION]


@requires_sqlite_3_38
def test_incoming_from_actuatorconfiguration_returns_no_results(
    resource_hierarchy: dict,
) -> None:
    """incoming from actuatorconfiguration returns no resources with stored edge direction."""
    store: SQLStore = resource_hierarchy["store"]
    ac_id = resource_hierarchy["actuatorconfiguration_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.ACTUATORCONFIGURATION,
        identifier=ac_id,
        relationship="parent",
        identifiers_only=True,
    )

    assert result == {}


@requires_sqlite_3_38
def test_both_from_actuatorconfiguration_reaches_ancestors_and_siblings(
    resource_hierarchy: dict,
) -> None:
    """both from actuatorconfiguration reaches ancestors and sibling datacontainer via operation."""
    store: SQLStore = resource_hierarchy["store"]
    ac_id = resource_hierarchy["actuatorconfiguration_id"]
    op_id = resource_hierarchy["operation_id"]
    ds_id = resource_hierarchy["discoveryspace_id"]
    ss_id = resource_hierarchy["samplestore_id"]
    dc_id = resource_hierarchy["datacontainer_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.ACTUATORCONFIGURATION,
        identifier=ac_id,
        identifiers_only=True,
    )

    assert op_id in result[CoreResourceKinds.OPERATION]
    assert ds_id in result[CoreResourceKinds.DISCOVERYSPACE]
    assert ss_id in result[CoreResourceKinds.SAMPLESTORE]
    # Sibling datacontainer is now reachable via ac → op → dc
    assert dc_id in result[CoreResourceKinds.DATACONTAINER]


@requires_sqlite_3_38
def test_both_from_discoveryspace_returns_rooted_ancestors_and_descendants_only(
    resource_hierarchy: dict,
) -> None:
    """both from discoveryspace excludes sibling discoveryspaces under the same samplestore."""
    store: SQLStore = resource_hierarchy["store"]
    ds_id = resource_hierarchy["discoveryspace_id"]
    ss_id = resource_hierarchy["samplestore_id"]
    op_id = resource_hierarchy["operation_id"]
    dc_id = resource_hierarchy["datacontainer_id"]
    ac_id = resource_hierarchy["actuatorconfiguration_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.DISCOVERYSPACE,
        identifier=ds_id,
        identifiers_only=True,
    )

    assert CoreResourceKinds.DISCOVERYSPACE not in result
    assert ss_id in result[CoreResourceKinds.SAMPLESTORE]
    assert op_id in result[CoreResourceKinds.OPERATION]
    assert dc_id in result[CoreResourceKinds.DATACONTAINER]
    assert ac_id in result[CoreResourceKinds.ACTUATORCONFIGURATION]


# ---------------------------------------------------------------------------
# multi-start
# ---------------------------------------------------------------------------


@pytest.fixture
def two_op_hierarchy(
    sql_store: SQLStore,
    random_space_resource_from_file: Callable[[str | None], DiscoverySpaceResource],
    data_container_resource: ado.core.datacontainer.resource.DataContainerResource,
) -> dict:
    """Two operations sharing the same discoveryspace, each with its own datacontainer."""

    from ado.core import (
        SampleStoreResource,
    )
    from ado.core.operation.config import (
        DiscoveryOperationConfiguration,
        DiscoveryOperationEnum,
        DiscoveryOperationResourceConfiguration,
    )
    from ado.core.operation.resource import OperationResource
    from ado.core.samplestore.config import (
        SampleStoreConfiguration,
        SampleStoreModuleConf,
        SampleStoreSpecification,
    )

    ss = SampleStoreResource(
        config=SampleStoreConfiguration(
            specification=SampleStoreSpecification(
                module=SampleStoreModuleConf(
                    moduleClass="SQLSampleStore",
                    moduleName="ado.core.samplestore.sql",
                )
            )
        )
    )
    sql_store.addResource(ss)

    ds = random_space_resource_from_file(sample_store_id=ss.identifier)
    sql_store.addResourceWithRelationships(ds, relatedIdentifiers=[ss.identifier])

    op1_config = DiscoveryOperationResourceConfiguration(
        spaces=[ds.identifier],
        operation=DiscoveryOperationConfiguration(),
    )
    op1 = OperationResource(
        config=op1_config,
        operationType=DiscoveryOperationEnum.EXPLORE,
        operatorIdentifier="test-op-1",
    )
    sql_store.addResourceWithRelationships(op1, relatedIdentifiers=[ds.identifier])

    op2_config = DiscoveryOperationResourceConfiguration(
        spaces=[ds.identifier],
        operation=DiscoveryOperationConfiguration(),
    )
    op2 = OperationResource(
        config=op2_config,
        operationType=DiscoveryOperationEnum.EXPLORE,
        operatorIdentifier="test-op-2",
    )
    sql_store.addResourceWithRelationships(op2, relatedIdentifiers=[ds.identifier])

    # dc1 belongs to op1; dc2 belongs to op2
    sql_store.addResourceWithRelationships(
        data_container_resource, relatedIdentifiers=[op1.identifier]
    )

    import pandas as pd

    from ado.core.datacontainer.resource import (
        DataContainer,
        DataContainerResource,
        TabularData,
    )

    df = pd.read_csv("examples/ml-multi-cloud/ml_export.csv")
    dc2 = DataContainerResource(
        config=DataContainer(
            tabularData={"entities": TabularData.from_dataframe(df)},
        )
    )
    sql_store.addResourceWithRelationships(dc2, relatedIdentifiers=[op2.identifier])

    return {
        "store": sql_store,
        "ss_id": ss.identifier,
        "ds_id": ds.identifier,
        "op1_id": op1.identifier,
        "op2_id": op2.identifier,
        "dc1_id": data_container_resource.identifier,
        "dc2_id": dc2.identifier,
    }


@requires_sqlite_3_38
def test_multi_start_returns_per_origin_grouping(
    two_op_hierarchy: dict,
) -> None:
    """Multi-start identifier mode returns dict keyed by origin identifier."""
    store: SQLStore = two_op_hierarchy["store"]
    op1_id = two_op_hierarchy["op1_id"]
    op2_id = two_op_hierarchy["op2_id"]
    dc1_id = two_op_hierarchy["dc1_id"]
    dc2_id = two_op_hierarchy["dc2_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier={op1_id, op2_id},
        relationship="child",
        identifiers_only=True,
    )

    # Top-level keys are the origin identifiers
    assert op1_id in result
    assert op2_id in result

    # Each origin bucket contains the correct datacontainer
    assert dc1_id in result[op1_id][CoreResourceKinds.DATACONTAINER]
    assert dc2_id in result[op2_id][CoreResourceKinds.DATACONTAINER]

    # dc2 should not appear under op1 and vice-versa
    assert dc2_id not in result[op1_id][CoreResourceKinds.DATACONTAINER]
    assert dc1_id not in result[op2_id][CoreResourceKinds.DATACONTAINER]


@requires_sqlite_3_38
def test_multi_start_shared_resource_appears_under_each_origin(
    two_op_hierarchy: dict,
) -> None:
    """Shared discovered resources appear under every origin that reaches them."""
    store: SQLStore = two_op_hierarchy["store"]
    op1_id = two_op_hierarchy["op1_id"]
    op2_id = two_op_hierarchy["op2_id"]
    ds_id = two_op_hierarchy["ds_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier={op1_id, op2_id},
        relationship="parent",
        identifiers_only=True,
    )

    # Both operations share the same discoveryspace
    assert ds_id in result[op1_id][CoreResourceKinds.DISCOVERYSPACE]
    assert ds_id in result[op2_id][CoreResourceKinds.DISCOVERYSPACE]


# ---------------------------------------------------------------------------
# identifier=None (all resources of start kind)
# ---------------------------------------------------------------------------


@requires_sqlite_3_38
def test_identifier_none_seeds_from_all_resources_of_kind(
    resource_hierarchy: dict,
) -> None:
    """identifier=None seeds from all resources of the given kind and returns multi-start shape."""
    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]
    ds_id = resource_hierarchy["discoveryspace_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=None,
        relationship="parent",
        identifiers_only=True,
    )

    # Result shape is dict[origin_id → dict[kind → list[str]]]
    assert isinstance(result, dict)
    # The operation we created must appear as an origin key
    assert op_id in result
    assert ds_id in result[op_id][CoreResourceKinds.DISCOVERYSPACE]


# ---------------------------------------------------------------------------
# hydrated mode
# ---------------------------------------------------------------------------


@requires_sqlite_3_38
def test_hydrated_single_start_returns_resources(
    resource_hierarchy: dict,
) -> None:
    """Single-start hydrated mode returns ADOResource objects, no outer origin key."""
    from ado.core.resources import ADOResource

    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]
    dc_id = resource_hierarchy["datacontainer_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=op_id,
        relationship="child",
        identifiers_only=False,
    )

    # Top-level keys are CoreResourceKinds, not origin identifiers
    assert CoreResourceKinds.DATACONTAINER in result
    # Values are dicts of identifier → ADOResource
    dc_bucket = result[CoreResourceKinds.DATACONTAINER]
    assert dc_id in dc_bucket
    assert isinstance(dc_bucket[dc_id], ADOResource)


@requires_sqlite_3_38
def test_hydrated_multi_start_grouping_matches_identifier_mode(
    two_op_hierarchy: dict,
) -> None:
    """Hydrated multi-start grouping matches identifier-only grouping."""
    store: SQLStore = two_op_hierarchy["store"]
    op1_id = two_op_hierarchy["op1_id"]
    op2_id = two_op_hierarchy["op2_id"]
    dc1_id = two_op_hierarchy["dc1_id"]
    dc2_id = two_op_hierarchy["dc2_id"]

    result_ids = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier={op1_id, op2_id},
        relationship="child",
        identifiers_only=True,
    )
    result_hydrated = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier={op1_id, op2_id},
        relationship="child",
        identifiers_only=False,
    )

    for origin in [op1_id, op2_id]:
        for kind in result_ids[origin]:
            assert set(result_ids[origin][kind]) == set(
                result_hydrated[origin][kind].keys()
            )

    # Spot-check actual resource objects
    from ado.core.resources import ADOResource

    assert isinstance(
        result_hydrated[op1_id][CoreResourceKinds.DATACONTAINER][dc1_id], ADOResource
    )
    assert isinstance(
        result_hydrated[op2_id][CoreResourceKinds.DATACONTAINER][dc2_id], ADOResource
    )


@requires_sqlite_3_38
def test_hydrated_start_identifier_excluded(
    resource_hierarchy: dict,
) -> None:
    """Start identifiers are excluded from hydrated results."""
    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=op_id,
        relationship="child",
        identifiers_only=False,
    )

    all_returned_identifiers = {
        ident for kind_bucket in result.values() for ident in kind_bucket
    }
    assert op_id not in all_returned_identifiers


# ---------------------------------------------------------------------------
# invalid input
# ---------------------------------------------------------------------------


def test_invalid_direction_raises_value_error(
    resource_hierarchy: dict,
) -> None:
    """Unsupported relationship raises ValueError."""
    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]
    with pytest.raises(
        ValueError,
        match="relationship must be 'child', 'parent' or 'both'",
    ):
        store.get_resources_by_relationship(
            kind=CoreResourceKinds.OPERATION,
            identifier=op_id,
            relationship="sideways",  # type: ignore[arg-type]
        )


def test_invalid_max_hops_zero_raises_value_error(
    resource_hierarchy: dict,
) -> None:
    """max_hops=0 raises ValueError before any DB query."""
    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]
    with pytest.raises(ValueError, match="max_hops must be a positive integer"):
        store.get_resources_by_relationship(
            kind=CoreResourceKinds.OPERATION,
            identifier=op_id,
            relationship="parent",
            max_hops=0,
            identifiers_only=True,
        )


def test_invalid_max_hops_negative_raises_value_error(
    resource_hierarchy: dict,
) -> None:
    """max_hops=-1 raises ValueError before any DB query."""
    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]
    with pytest.raises(ValueError, match="max_hops must be a positive integer"):
        store.get_resources_by_relationship(
            kind=CoreResourceKinds.OPERATION,
            identifier=op_id,
            relationship="child",
            max_hops=-1,
            identifiers_only=True,
        )


@requires_sqlite_3_38
def test_valid_kind_direction_combo_with_no_reachable_resources_returns_empty(
    resource_hierarchy: dict,
) -> None:
    """A valid traversal family with no reachable resources returns an empty dict."""
    store: SQLStore = resource_hierarchy["store"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.SAMPLESTORE,
        identifier=resource_hierarchy["samplestore_id"],
        relationship="parent",
        identifiers_only=True,
    )

    assert result == {}


@requires_sqlite_3_38
def test_both_from_operation_max_hops_1_returns_one_level_each_direction(
    resource_hierarchy: dict,
) -> None:
    """both from operation with max_hops=1 returns one hop up and one hop down."""
    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]
    ds_id = resource_hierarchy["discoveryspace_id"]
    dc_id = resource_hierarchy["datacontainer_id"]
    ac_id = resource_hierarchy["actuatorconfiguration_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=op_id,
        max_hops=1,
        identifiers_only=True,
    )

    # One hop up → discoveryspace only (not samplestore)
    assert CoreResourceKinds.DISCOVERYSPACE in result
    assert ds_id in result[CoreResourceKinds.DISCOVERYSPACE]
    assert CoreResourceKinds.SAMPLESTORE not in result
    # One hop down → datacontainer and actuatorconfiguration
    assert CoreResourceKinds.DATACONTAINER in result
    assert dc_id in result[CoreResourceKinds.DATACONTAINER]
    assert CoreResourceKinds.ACTUATORCONFIGURATION in result
    assert ac_id in result[CoreResourceKinds.ACTUATORCONFIGURATION]


def test_empty_identifier_set_returns_empty(
    resource_hierarchy: dict,
) -> None:
    """Empty identifier set returns an empty result immediately."""
    store: SQLStore = resource_hierarchy["store"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=set(),
        relationship="child",
        identifiers_only=True,
    )

    assert result == {}


# ---------------------------------------------------------------------------
# no-results: valid traversal but no related resources
# ---------------------------------------------------------------------------


@requires_sqlite_3_38
def test_valid_traversal_with_no_related_resources(
    sql_store: SQLStore,
    random_space_resource_from_file: Callable[[str | None], DiscoverySpaceResource],
) -> None:
    """A valid traversal that simply finds no related resources returns an empty dict."""
    from ado.core import SampleStoreResource
    from ado.core.samplestore.config import (
        SampleStoreConfiguration,
        SampleStoreModuleConf,
        SampleStoreSpecification,
    )

    # samplestore with no children
    ss = SampleStoreResource(
        config=SampleStoreConfiguration(
            specification=SampleStoreSpecification(
                module=SampleStoreModuleConf(
                    moduleClass="SQLSampleStore",
                    moduleName="ado.core.samplestore.sql",
                )
            )
        )
    )
    sql_store.addResource(ss)

    result = sql_store.get_resources_by_relationship(
        kind=CoreResourceKinds.SAMPLESTORE,
        identifier=ss.identifier,
        relationship="child",
        identifiers_only=True,
    )

    assert result == {}


# ---------------------------------------------------------------------------
# include_start_resources
# ---------------------------------------------------------------------------


@requires_sqlite_3_38
def test_include_start_resources_single_identifier(
    resource_hierarchy: dict,
) -> None:
    """include_start_resources=True adds the start resource to the result."""
    from ado.core.resources import ADOResource

    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=op_id,
        relationship="child",
        identifiers_only=False,
        include_start_resources=True,
    )

    assert CoreResourceKinds.OPERATION in result
    assert op_id in result[CoreResourceKinds.OPERATION]
    assert isinstance(result[CoreResourceKinds.OPERATION][op_id], ADOResource)


@requires_sqlite_3_38
def test_include_start_resources_multi_identifier(
    two_op_hierarchy: dict,
) -> None:
    """include_start_resources=True adds each start resource in the multi-identifier result."""
    from ado.core.resources import ADOResource

    store: SQLStore = two_op_hierarchy["store"]
    op1_id = two_op_hierarchy["op1_id"]
    op2_id = two_op_hierarchy["op2_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier={op1_id, op2_id},
        relationship="child",
        identifiers_only=False,
        include_start_resources=True,
    )

    for op_id in [op1_id, op2_id]:
        assert op_id in result
        assert CoreResourceKinds.OPERATION in result[op_id]
        assert op_id in result[op_id][CoreResourceKinds.OPERATION]
        assert isinstance(
            result[op_id][CoreResourceKinds.OPERATION][op_id], ADOResource
        )


@requires_sqlite_3_38
def test_include_start_resources_no_related_resources(
    resource_hierarchy: dict,
) -> None:
    """include_start_resources=True still returns the start resource when traversal finds nothing."""
    from ado.core.resources import ADOResource

    store: SQLStore = resource_hierarchy["store"]
    ss_id = resource_hierarchy["samplestore_id"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.SAMPLESTORE,
        identifier=ss_id,
        relationship="parent",
        identifiers_only=False,
        include_start_resources=True,
    )

    assert CoreResourceKinds.SAMPLESTORE in result
    assert ss_id in result[CoreResourceKinds.SAMPLESTORE]
    assert isinstance(result[CoreResourceKinds.SAMPLESTORE][ss_id], ADOResource)


def test_include_start_resources_with_identifiers_only_raises(
    resource_hierarchy: dict,
) -> None:
    """include_start_resources=True combined with identifiers_only=True raises ValueError."""
    store: SQLStore = resource_hierarchy["store"]
    op_id = resource_hierarchy["operation_id"]

    with pytest.raises(
        ValueError,
        match="include_start_resources=True requires identifiers_only=False",
    ):
        store.get_resources_by_relationship(
            kind=CoreResourceKinds.OPERATION,
            identifier=op_id,
            relationship="child",
            identifiers_only=True,
            include_start_resources=True,
        )


def test_include_start_resources_with_identifier_none_raises(
    resource_hierarchy: dict,
) -> None:
    """include_start_resources=True combined with identifier=None raises ValueError."""
    store: SQLStore = resource_hierarchy["store"]

    with pytest.raises(
        ValueError,
        match="include_start_resources=True requires identifier to be a str or set",
    ):
        store.get_resources_by_relationship(
            kind=CoreResourceKinds.OPERATION,
            identifier=None,
            relationship="child",
            identifiers_only=False,
            include_start_resources=True,
        )


# ---------------------------------------------------------------------------
# operation → operation edges
# ---------------------------------------------------------------------------


@pytest.fixture
def resource_hierarchy_with_child_operation(
    sql_store: SQLStore,
    random_space_resource_from_file: Callable[[str | None], DiscoverySpaceResource],
    operation_resource: OperationResource,
) -> dict:
    """Build a hierarchy with a parent and a child operation linked to a second space.

    Layout:
        samplestore_1 → space_1 → parent_op → space_2 → child_op
    """
    from ado.core import SampleStoreResource
    from ado.core.operation.config import (
        DiscoveryOperationConfiguration,
        DiscoveryOperationEnum,
        DiscoveryOperationResourceConfiguration,
    )
    from ado.core.samplestore.config import (
        SampleStoreConfiguration,
        SampleStoreModuleConf,
        SampleStoreSpecification,
    )

    ss1 = SampleStoreResource(
        config=SampleStoreConfiguration(
            specification=SampleStoreSpecification(
                module=SampleStoreModuleConf(
                    moduleClass="SQLSampleStore",
                    moduleName="ado.core.samplestore.sql",
                )
            )
        )
    )
    sql_store.addResource(ss1)

    space1 = random_space_resource_from_file(sample_store_id=ss1.identifier)
    sql_store.addResourceWithRelationships(space1, relatedIdentifiers=[ss1.identifier])

    operation_resource.config.inputs["discoverySpace"] = ADOResourceReference(
        identifier=space1.identifier,
        kind=CoreResourceKinds.DISCOVERYSPACE,
    )
    sql_store.addResourceWithRelationships(
        operation_resource, relatedIdentifiers=[space1.identifier]
    )
    parent_op = operation_resource

    # A second space produced by the parent operation
    space2 = random_space_resource_from_file(sample_store_id=ss1.identifier)
    sql_store.addResourceWithRelationships(space2, relatedIdentifiers=[ss1.identifier])
    # Link parent_op → space2
    sql_store.addRelationship(
        subjectIdentifier=parent_op.identifier,
        objectIdentifier=space2.identifier,
    )

    child_op_config = DiscoveryOperationResourceConfiguration(
        spaces=[space2.identifier],
        operation=DiscoveryOperationConfiguration(),
    )
    child_op = OperationResource(
        config=child_op_config,
        operationType=DiscoveryOperationEnum.EXPLORE,
        operatorIdentifier="test-child-op",
    )
    sql_store.addResourceWithRelationships(
        child_op, relatedIdentifiers=[space2.identifier]
    )
    # Link parent_op → child_op directly
    sql_store.addRelationship(
        subjectIdentifier=parent_op.identifier,
        objectIdentifier=child_op.identifier,
    )

    return {
        "store": sql_store,
        "samplestore_id": ss1.identifier,
        "space1_id": space1.identifier,
        "space2_id": space2.identifier,
        "parent_op_id": parent_op.identifier,
        "child_op_id": child_op.identifier,
    }


@requires_sqlite_3_38
def test_outgoing_from_parent_op_returns_child_op(
    resource_hierarchy_with_child_operation: dict,
) -> None:
    """outgoing max_hops=1 from parent op returns child op only (among operations)."""
    h = resource_hierarchy_with_child_operation
    store: SQLStore = h["store"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=h["parent_op_id"],
        relationship="child",
        result_kinds={CoreResourceKinds.OPERATION},
        max_hops=1,
        identifiers_only=True,
    )

    assert CoreResourceKinds.OPERATION in result
    assert h["child_op_id"] in result[CoreResourceKinds.OPERATION]
    assert h["parent_op_id"] not in result.get(CoreResourceKinds.OPERATION, set())


@requires_sqlite_3_38
def test_incoming_from_child_op_returns_parent_op(
    resource_hierarchy_with_child_operation: dict,
) -> None:
    """incoming max_hops=1 from child op returns parent op only."""
    h = resource_hierarchy_with_child_operation
    store: SQLStore = h["store"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=h["child_op_id"],
        relationship="parent",
        result_kinds={CoreResourceKinds.OPERATION},
        max_hops=1,
        identifiers_only=True,
    )

    assert CoreResourceKinds.OPERATION in result
    assert h["parent_op_id"] in result[CoreResourceKinds.OPERATION]
    assert h["child_op_id"] not in result.get(CoreResourceKinds.OPERATION, set())


@requires_sqlite_3_38
def test_both_from_parent_op_returns_parent_and_child_op(
    resource_hierarchy_with_child_operation: dict,
) -> None:
    """both max_hops=1 from parent op returns the child op (and space1 as incoming)."""
    h = resource_hierarchy_with_child_operation
    store: SQLStore = h["store"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=h["parent_op_id"],
        result_kinds={CoreResourceKinds.OPERATION},
        max_hops=1,
        identifiers_only=True,
    )

    # child_op is reachable via outgoing edge parent_op → child_op
    assert CoreResourceKinds.OPERATION in result
    assert h["child_op_id"] in result[CoreResourceKinds.OPERATION]


@requires_sqlite_3_38
def test_space_op_space_op_multi_hop(
    resource_hierarchy_with_child_operation: dict,
) -> None:
    """outgoing result_kinds={OPERATION} from space1 traverses space1→parent_op→child_op chain."""
    h = resource_hierarchy_with_child_operation
    store: SQLStore = h["store"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.DISCOVERYSPACE,
        identifier=h["space1_id"],
        relationship="child",
        result_kinds={CoreResourceKinds.OPERATION},
        max_hops=4,
        identifiers_only=True,
    )

    assert CoreResourceKinds.OPERATION in result
    assert h["parent_op_id"] in result[CoreResourceKinds.OPERATION]
    assert h["child_op_id"] in result[CoreResourceKinds.OPERATION]


# ---------------------------------------------------------------------------
# document edges
# ---------------------------------------------------------------------------


@pytest.fixture
def resource_hierarchy_with_document(
    sql_store: SQLStore,
    random_space_resource_from_file: Callable[[str | None], DiscoverySpaceResource],
) -> dict:
    """Build a hierarchy with a document that has both a parent and a child operation.

    Layout:
        samplestore → space → op_a   (parent of document)
        document (subject=op_a, object=document  stored as addRelationship)
        document → op_b  (child of document: subject=document, object=op_b)
    """
    from ado.core import SampleStoreResource
    from ado.core.operation.config import (
        DiscoveryOperationConfiguration,
        DiscoveryOperationEnum,
        DiscoveryOperationResourceConfiguration,
    )
    from ado.core.samplestore.config import (
        SampleStoreConfiguration,
        SampleStoreModuleConf,
        SampleStoreSpecification,
    )

    ss = SampleStoreResource(
        config=SampleStoreConfiguration(
            specification=SampleStoreSpecification(
                module=SampleStoreModuleConf(
                    moduleClass="SQLSampleStore",
                    moduleName="ado.core.samplestore.sql",
                )
            )
        )
    )
    sql_store.addResource(ss)

    space = random_space_resource_from_file(sample_store_id=ss.identifier)
    sql_store.addResourceWithRelationships(space, relatedIdentifiers=[ss.identifier])

    op_a_config = DiscoveryOperationResourceConfiguration(
        spaces=[space.identifier],
        operation=DiscoveryOperationConfiguration(),
    )
    op_a = OperationResource(
        config=op_a_config,
        operationType=DiscoveryOperationEnum.EXPLORE,
        operatorIdentifier="test-op-a",
    )
    sql_store.addResourceWithRelationships(op_a, relatedIdentifiers=[space.identifier])

    doc = DocumentResource(
        config=DocumentConfiguration(content="# Test report", contentType="markdown")
    )
    sql_store.addResource(doc)
    # op_a is the parent of the document: subject=op_a, object=doc
    sql_store.addRelationship(
        subjectIdentifier=op_a.identifier,
        objectIdentifier=doc.identifier,
    )

    op_b_config = DiscoveryOperationResourceConfiguration(
        spaces=[space.identifier],
        operation=DiscoveryOperationConfiguration(),
    )
    op_b = OperationResource(
        config=op_b_config,
        operationType=DiscoveryOperationEnum.EXPLORE,
        operatorIdentifier="test-op-b",
    )
    sql_store.addResourceWithRelationships(op_b, relatedIdentifiers=[space.identifier])
    # doc is the parent of op_b: subject=doc, object=op_b
    sql_store.addRelationship(
        subjectIdentifier=doc.identifier,
        objectIdentifier=op_b.identifier,
    )

    return {
        "store": sql_store,
        "op_a_id": op_a.identifier,
        "op_b_id": op_b.identifier,
        "doc_id": doc.identifier,
    }


@requires_sqlite_3_38
def test_outgoing_from_operation_returns_document(
    resource_hierarchy_with_document: dict,
) -> None:
    """outgoing from op_a returns the document (parent role)."""
    h = resource_hierarchy_with_document
    store: SQLStore = h["store"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=h["op_a_id"],
        relationship="child",
        result_kinds={CoreResourceKinds.DOCUMENT},
        max_hops=1,
        identifiers_only=True,
    )

    assert CoreResourceKinds.DOCUMENT in result
    assert h["doc_id"] in result[CoreResourceKinds.DOCUMENT]


@requires_sqlite_3_38
def test_incoming_from_document_returns_parent_operation(
    resource_hierarchy_with_document: dict,
) -> None:
    """incoming from document returns op_a (the parent operation) only."""
    h = resource_hierarchy_with_document
    store: SQLStore = h["store"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.DOCUMENT,
        identifier=h["doc_id"],
        relationship="parent",
        result_kinds={CoreResourceKinds.OPERATION},
        max_hops=1,
        identifiers_only=True,
    )

    assert CoreResourceKinds.OPERATION in result
    assert h["op_a_id"] in result[CoreResourceKinds.OPERATION]
    assert h["op_b_id"] not in result.get(CoreResourceKinds.OPERATION, set())


@requires_sqlite_3_38
def test_outgoing_from_document_returns_child_operation(
    resource_hierarchy_with_document: dict,
) -> None:
    """outgoing from document returns op_b (the child operation) only."""
    h = resource_hierarchy_with_document
    store: SQLStore = h["store"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.DOCUMENT,
        identifier=h["doc_id"],
        relationship="child",
        result_kinds={CoreResourceKinds.OPERATION},
        max_hops=1,
        identifiers_only=True,
    )

    assert CoreResourceKinds.OPERATION in result
    assert h["op_b_id"] in result[CoreResourceKinds.OPERATION]
    assert h["op_a_id"] not in result.get(CoreResourceKinds.OPERATION, set())


@requires_sqlite_3_38
def test_both_from_document_returns_both_operations(
    resource_hierarchy_with_document: dict,
) -> None:
    """both from document returns op_a and op_b."""
    h = resource_hierarchy_with_document
    store: SQLStore = h["store"]

    result = store.get_resources_by_relationship(
        kind=CoreResourceKinds.DOCUMENT,
        identifier=h["doc_id"],
        result_kinds={CoreResourceKinds.OPERATION},
        max_hops=1,
        identifiers_only=True,
    )

    assert CoreResourceKinds.OPERATION in result
    assert h["op_a_id"] in result[CoreResourceKinds.OPERATION]
    assert h["op_b_id"] in result[CoreResourceKinds.OPERATION]
