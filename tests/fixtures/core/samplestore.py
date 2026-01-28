# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import copy
import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
import yaml

import orchestrator.core
import orchestrator.core.samplestore.resource
import orchestrator.modules
import orchestrator.utilities
import orchestrator.utilities.location
from orchestrator.core import ADOResource, SampleStoreResource
from orchestrator.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreModuleConf,
    SampleStoreReference,
)
from orchestrator.core.samplestore.csv import (
    CSVSampleStore,
    CSVSampleStoreDescription,
)
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLStore
from orchestrator.utilities.location import FilePathLocation, SQLStoreConfiguration


@pytest.fixture
def random_sample_store_resource_from_file(
    valid_ado_project_context: ProjectContext, random_identifier: Callable[[], str]
) -> Callable[[], SampleStoreResource]:
    def _random_sample_store_resource_from_file() -> SampleStoreResource:
        file = pathlib.Path("tests/resources/samplestore/sample_store_resource.json")
        random_id = random_identifier()

        # We must set the storageLocation field before validation, or it will fail
        file_content = json.loads(file.read_text())
        file_content["config"]["specification"][
            "storageLocation"
        ] = valid_ado_project_context.metadataStore.model_dump()

        # Get the model
        sample_store = (
            orchestrator.core.samplestore.resource.SampleStoreResource.model_validate(
                file_content
            )
        )

        # Final touch-ups
        sample_store.identifier = random_id
        sample_store.config.specification.parameters["identifier"] = random_id
        return sample_store

    return _random_sample_store_resource_from_file


@pytest.fixture
def random_sample_store_resource_from_db(
    random_sample_store_resource_from_file: Callable[[], SampleStoreResource],
    create_resources: Callable[[list[ADOResource], SQLStore], None],
) -> Callable[[SampleStoreResource], SampleStoreResource]:
    def _random_sample_store_resource_from_db() -> SampleStoreResource:
        sample_store = random_sample_store_resource_from_file()
        create_resources(resources=[sample_store])
        return sample_store

    return _random_sample_store_resource_from_db


@pytest.fixture
def random_sql_sample_store(
    random_sample_store_resource_from_db: Callable[
        [SampleStoreResource], SampleStoreResource
    ],
    valid_ado_project_context: ProjectContext,
) -> Callable[[], SQLSampleStore]:
    def _random_sql_sample_store() -> SQLSampleStore:
        return SQLSampleStore(
            identifier=random_sample_store_resource_from_db().identifier,
            storageLocation=valid_ado_project_context.metadataStore,
            parameters={},
        )

    return _random_sql_sample_store


entity_data = """
,Unnamed: 0,SMILES,"Real_MW (60.02, 610.38)",Real_MW score,"Real_BCF (0.2, 2.71)",Real_BCF score,"Real_BioDeg (0.47, 2.66)",Real_BioDeg score,"Real_LD50 (3.9, 7543.0)",Real_LD50 score,"Real_LogD (-5.66, 7.82)",Real_LogD score,"Real_LogHL (-10.96, -3.54)",Real_LogHL score,"Real_LogWS (-6.19, 1.13)",Real_LogWS score,"Real_SCScore (2.68, 5.0)",Real_SCScore score,"Real_pKa (-0.83, 10.58)",Real_pKa score
0,0,[O-]SC1=C([O-])OCCC1,110.49989702393891,0.0,0.693083124519546,0.0,0.7804221726615277,0.0,4140.458216348032,0.0,-1.5567766053454803,0.0,-6.808897117754416,0.0,-0.4605503035328822,0.0,4.569592629049344,0.0,7.077925981347688,0.0
1,1,[O-]SC1=COC([O-])CC1,117.18694300304168,0.0,0.693083124519546,0.0,0.7804221726615277,0.0,4059.57355394821,0.0,-1.5567766053454803,0.0,-6.808897117754416,0.0,-0.4605503035328822,0.0,4.569592629049344,0.0,3.809480882942567,0.0
2,2,[O-]SC1=COCC([O-])C1,121.02629023155076,0.0,0.693083124519546,0.0,0.7804221726615277,0.0,1048.2053695490351,0.0,-2.795277524518923,0.0,-7.264012858253796,0.0,-0.4605503035328822,0.0,4.569592629049344,0.0,4.195514499313307,0.0
3,3,[O-]SC1=COCCC1[O-],117.8551277336886,0.0,0.693083124519546,0.0,0.7804221726615277,0.0,1217.339513792586,0.0,-2.795277524518923,0.0,-7.145540410315964,0.0,-0.4605503035328822,0.0,4.569592629049344,0.0,3.597523320475148,0.0
"""


@pytest.fixture
def csv_sample_store_identifier() -> str:
    return "gt4sd-pfas-molgx-model-one-92f4b88651b213bf3cf742db1ce84138"


@pytest.fixture
def csv_sample_store_parameters() -> tuple[
    dict[str, str],
    dict[str, str | list[str] | list[dict[str, str | dict[str, str]]]],
]:
    parameters = {
        "generatorIdentifier": "gt4sd-pfas-molgx-model-one",
        "identifierColumn": "smiles",
        "experiments": [
            {
                "experimentIdentifier": "molgx-toxicity-inference-experiment",
                "observedPropertyMap": {
                    "pka": "Real_pKa (-0.83, 10.58)",
                    "logws": "Real_LogWS (-6.19, 1.13)",
                    "biodegradation halflife": "Real_BioDeg (0.47, 2.66)",
                    "ld50": "Real_LD50 (3.9, 7543.0)",
                },
                # Using list format since column name matches property name
                "constitutivePropertyMap": ["smiles"],
            }
        ],
    }

    location = {
        "path": "examples/pfas-generative-models/data/GM_Comparison/MolGX/Sample_0/PFAS_MolGX_test_SHORT_v0.csv"
    }

    return location, parameters


@pytest.fixture
def csv_sample_store_reference(
    csv_sample_store_parameters: tuple[
        dict[str, str],
        dict[str, str | list[str] | list[dict[str, str | dict[str, str]]]],
    ],
) -> SampleStoreReference:
    """Creates a SampleStoreReference for a CSV file"""

    location, parameters = csv_sample_store_parameters

    return SampleStoreReference(
        module=SampleStoreModuleConf(
            moduleClass="CSVSampleStore",
            moduleName="orchestrator.core.samplestore.csv",
        ),
        storageLocation=location,
        parameters=parameters,
    )


@pytest.fixture
def csv_sample_store(
    csv_sample_store_parameters: tuple[
        dict[str, str],
        dict[str, str | list[str] | list[dict[str, str | dict[str, str]]]],
    ],
) -> CSVSampleStore:
    location, parameters = csv_sample_store_parameters

    return CSVSampleStore(
        storageLocation=orchestrator.utilities.location.FilePathLocation.model_validate(
            location
        ),
        parameters=CSVSampleStoreDescription.model_validate(parameters),
    )


@pytest.fixture
def ml_multi_cloud_sample_store_configuration() -> SampleStoreConfiguration:
    with open("tests/resources/ml_multicloud_sample_store.yaml") as f:
        d = yaml.safe_load(f)

    return SampleStoreConfiguration.model_validate(d)


@pytest.fixture
def sample_store_configuration_smiles_yaml() -> dict[str, Any]:  # noqa: ANN401
    y = """
    copyFrom:
    - module:
        moduleClass: GT4SDTransformer
        moduleName: orchestrator.plugins.samplestores.gt4sd
      storageLocation:
        path: 'tests/test_generations.csv'
      parameters:
        source: 'tests/test_generations.csv'
        generatorIdentifier: 'gt4sd-pfas-transformer-model-one'
    """

    return yaml.safe_load(y)


@pytest.fixture
def sample_store_configuration_smiles(
    sample_store_configuration_smiles_yaml: dict[str, Any],
) -> orchestrator.core.samplestore.config.SampleStoreConfiguration:
    source_conf = (
        orchestrator.core.samplestore.config.SampleStoreConfiguration.model_validate(
            sample_store_configuration_smiles_yaml
        )
    )

    assert source_conf.copyFrom[0].module.moduleClass == "GT4SDTransformer"

    return source_conf


@pytest.fixture
def sample_store_resource(
    ml_multi_cloud_sample_store_configuration: SampleStoreConfiguration,
) -> SampleStoreResource:
    return SampleStoreResource(
        identifier="test_source",
        config=ml_multi_cloud_sample_store_configuration,
    )


valid_sample_store_configs = ["tests/resources/ml_multicloud_sample_store.yaml"]


@pytest.fixture(params=valid_sample_store_configs)
def valid_sample_store_config_file(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def test_sample_store_location() -> SQLStoreConfiguration:
    return SQLStoreConfiguration(
        scheme="mysql+pymysql",
        host="localhost",
        port=3306,
        password="somepass",
        database="sql_sample_store_aaa123",
        user="admin",
    )


@pytest.fixture(params=["sql", "csv"])
def sample_store_test_data(
    request: pytest.FixtureRequest,
) -> tuple[dict[str, Any], SQLStoreConfiguration | FilePathLocation]:
    c = None
    if request.param == "sql":
        c = (
            yaml.safe_load(sqlConfig),
            orchestrator.utilities.location.SQLStoreConfiguration,
        )
    elif request.param == "csv":
        c = (
            yaml.safe_load(csvConfig),
            orchestrator.utilities.location.FilePathLocation,
        )

    return c


csvConfig = """
module:
  moduleClass: CSVSampleStore
  moduleName: orchestrator.core.samplestore.csv
parameters:
  generatorIdentifier: 'gt4sd-pfas-molgx-model-one'
  identifierColumn: 'smiles'
  experiments:
    - experimentIdentifier: 'molgx-toxicity-inference-experiment'
      observedPropertyMap:
        pka: "Real_pKa (-0.83, 10.58)"
        logws: "Real_LogWS (-6.19, 1.13)"
        "biodegradation halflife": "Real_BioDeg (0.47, 2.66)"
        ld50: "Real_LD50 (3.9, 7543.0)"
      constitutivePropertyMap:
        smiles: 'smiles'
storageLocation:
  path: 'examples/pfas-generative-models/data/GM_Comparison/MolGX/Sample_0/PFAS_MolGX_test_SHORT_v0.csv'
"""
sqlConfig = """
module:
  moduleName: orchestrator.core.samplestore.sql
  moduleClass: SQLSampleStore
storageLocation:
  scheme: mysql+pymysql
  host: localhost
  port: 3306
  password: somepass
  user: michaelj
  database: temp
"""


@pytest.fixture
def sample_store_module_and_storage_location(
    request: pytest.FixtureRequest,
) -> tuple[SampleStoreModuleConf, SQLStoreConfiguration]:
    return SampleStoreModuleConf(
        moduleClass="SQLSampleStore",
        moduleName="orchestrator.core.samplestore.sql",
    ), SQLStoreConfiguration(
        scheme="mysql+pymysql",
        host="localhost",
        port=3306,
        password="somepass",
        database="mydb",
        user="admin",
    )


@pytest.fixture
def ado_sql_sample_store_with_storagelocation() -> (
    orchestrator.core.samplestore.config.SampleStoreConfiguration
):
    y = """
    copyFrom: []
    specification:
      module:
        moduleClass: SQLSampleStore
        moduleName: orchestrator.core.samplestore.sql
        modulePath: .
        moduleType: sample_store
      parameters: {}
      storageLocation:
        database: some-db
        host: some-host
        password: some-password
        path: some-path
        port: null
        scheme: mysql+pymysql
        sslVerify: false
        user: some-user
    """

    return orchestrator.core.samplestore.config.SampleStoreConfiguration.model_validate(
        yaml.safe_load(y)
    )


@pytest.fixture
def ado_sql_sample_store_no_storagelocation(
    ado_sql_sample_store_with_storagelocation: SampleStoreConfiguration,
) -> SampleStoreConfiguration:
    sample_store = copy.deepcopy(ado_sql_sample_store_with_storagelocation)
    sample_store.specification.storageLocation = None
    return sample_store
