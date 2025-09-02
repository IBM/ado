# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pytest

from orchestrator.core import DataContainerResource
from orchestrator.core.datacontainer.resource import DataContainer, TabularData


@pytest.fixture
def data_container_resource(
    testTabularDataString, test_sample_store_location
) -> DataContainerResource:

    data = {"person": {"name": "mj", "age": 2}, "important_info": ["t1", 1, "t2"]}

    return DataContainerResource(
        config=DataContainer(
            tabularData={"important_entities": testTabularDataString},
            data=data,
            locationData={"entity_location": test_sample_store_location},
        )
    )


@pytest.fixture
def testTabularDataString():

    import pandas as pd

    df = pd.read_csv("examples/ml-multi-cloud/ml_export.csv")
    return TabularData.from_dataframe(df)
