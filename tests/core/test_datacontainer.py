# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import ado.core
import ado.utilities.location
from ado.core import DataContainerResource
from ado.core.datacontainer.resource import DataContainer, TabularData
from ado.utilities.location import SQLStoreConfiguration


def test_tabular_data(testTabularDataString: TabularData) -> None:

    df = testTabularDataString.dataframe()
    newdf = TabularData.from_dataframe(df)
    assert testTabularDataString.data == newdf.data


def test_data_container_resource(
    data_container_resource: DataContainerResource,
    testTabularDataString: TabularData,
    test_sample_store_location: SQLStoreConfiguration,
) -> None:

    assert (
        data_container_resource.kind
        == ado.core.resources.CoreResourceKinds.DATACONTAINER
    )
    assert data_container_resource.identifier.split("-")[0] == "datacontainer"

    assert isinstance(data_container_resource.config, DataContainer)
    assert (
        data_container_resource.config.tabularData["important_entities"]
        == testTabularDataString
    )
    assert (
        data_container_resource.config.locationData["entity_location"]
        == test_sample_store_location
    )

    assert isinstance(
        data_container_resource.config.tabularData["important_entities"],
        TabularData,
    )
    assert isinstance(
        data_container_resource.config.locationData["entity_location"],
        ado.utilities.location.SQLStoreConfiguration,
    )

    # Test ser/deser

    dc = DataContainerResource.model_validate(data_container_resource.model_dump())

    assert isinstance(dc.config.tabularData["important_entities"], TabularData)
    assert isinstance(
        dc.config.locationData["entity_location"],
        ado.utilities.location.SQLStoreConfiguration,
    )

    assert (
        data_container_resource.config.tabularData["important_entities"]
        == testTabularDataString
    )
    assert (
        data_container_resource.config.locationData["entity_location"]
        == test_sample_store_location
    )


def test_datacontainer_rich_print(
    data_container_resource: DataContainerResource,
) -> None:
    from rich.console import Console

    assert hasattr(data_container_resource, "__rich__")
    Console().print(data_container_resource)
