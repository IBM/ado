# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import datetime
import typing
from typing import Any

import pytest
import yaml

import orchestrator.core
import orchestrator.utilities
import orchestrator.utilities.location
from orchestrator.core import SampleStoreResource
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
from orchestrator.core.samplestore.utils import initialize_sample_store_from_reference
from orchestrator.schema.observed_property import ObservedProperty
from orchestrator.schema.property import (
    AbstractPropertyDescriptor,
    ConstitutivePropertyDescriptor,
    Property,
)
from orchestrator.schema.property_value import (
    ConstitutivePropertyValue,
    ValueTypeEnum,
)
from orchestrator.schema.reference import ExperimentReference
from orchestrator.utilities.location import FilePathLocation, SQLStoreConfiguration

if typing.TYPE_CHECKING:
    from orchestrator.schema.entity import Entity


def test_state_identifier(
    csv_sample_store: CSVSampleStore,
    csv_sample_store_identifier: str,
) -> None:
    """Check the csv sample store id is what is expected"""

    assert csv_sample_store.identifier == csv_sample_store_identifier


def test_csv_sample_store_experiments(
    csv_sample_store: CSVSampleStore,
) -> None:
    """Check the experiments created in the state are what is expected"""

    catalog = csv_sample_store.experimentCatalog()
    assert len(catalog.experiments) == 1
    test_experiment = catalog.experiments[0]

    assert test_experiment.identifier == "molgx-toxicity-inference-experiment"
    expected_properties = ["pka", "logws", "biodegradation halflife", "ld50"]

    properties = [t.identifier for t in test_experiment.targetProperties]
    for p in properties:
        assert p in expected_properties


def test_csv_sample_store_entities(
    csv_sample_store: CSVSampleStore,
) -> None:
    """Tests if the first and last entities are what is expected"""

    assert csv_sample_store.entities[0].identifier == "[O-]SC1=C([O-])OCCC1"
    assert csv_sample_store.entities[-1].identifier == "O=S(=O)([O-])c1ccc([O-])c(O)c1"


def test_csv_sample_store_config(
    csv_sample_store: CSVSampleStore,
    csv_sample_store_parameters: tuple[
        dict[str, str],
        dict[str, str | list[str] | list[dict[str, str | dict[str, str]]]],
    ],
) -> None:
    location, parameters = csv_sample_store_parameters

    assert csv_sample_store.config == CSVSampleStoreDescription.model_validate(
        parameters
    )

    assert (
        csv_sample_store.location
        == orchestrator.utilities.location.FilePathLocation.model_validate(location)
    )


def test_csv_sample_store_description(
    csv_sample_store_parameters: tuple[
        dict[str, str],
        dict[str, str | list[str] | list[dict[str, str | dict[str, str]]]],
    ],
) -> None:
    _location, params = csv_sample_store_parameters
    desc = CSVSampleStoreDescription.model_validate(params)

    catalog = desc.catalog
    assert len(catalog.experiments) == 1
    assert catalog.experimentForReference(
        ExperimentReference(
            experimentIdentifier="molgx-toxicity-inference-experiment",
            actuatorIdentifier="replay",
        )
    )
    assert [op.targetProperty.identifier for op in desc.observedProperties] == [
        "pka",
        "logws",
        "biodegradation halflife",
        "ld50",
    ]
    assert sorted(desc.source_observed_property_identifiers) == sorted(
        [
            "Real_pKa (-0.83, 10.58)",
            "Real_LogWS (-6.19, 1.13)",
            "Real_BioDeg (0.47, 2.66)",
            "Real_LD50 (3.9, 7543.0)",
        ]
    )


def test_sample_store_resource(sample_store_resource: SampleStoreResource) -> None:
    assert sample_store_resource.identifier is not None
    assert sample_store_resource.identifier == "test_source"

    assert (
        sample_store_resource.kind
        == orchestrator.core.resources.CoreResourceKinds.SAMPLESTORE
    )

    assert sample_store_resource.created < datetime.datetime.now(datetime.timezone.utc)
    assert isinstance(sample_store_resource.metadata, dict)
    assert sample_store_resource.config is not None
    assert isinstance(sample_store_resource.config, SampleStoreConfiguration)
    assert sample_store_resource.config.specification.module is not None
    assert sample_store_resource.config.specification.parameters is not None
    assert len(sample_store_resource.status) == 1

    assert sample_store_resource.config.specification is not None
    assert sample_store_resource.config.specification is not None

    # Note: the resource location is optional in a SampleStoreSpecification to allow it to be set by external
    raw = sample_store_resource.model_dump()
    m = SampleStoreResource.model_validate(raw)

    if sample_store_resource.config.specification.storageLocation is not None:
        assert m.config.specification.storageLocation is not None
    else:
        # Test setting a storageLocation, dumping, and loading
        sample_store_resource.config.specification.storageLocation = (
            orchestrator.utilities.location.SQLStoreConfiguration(
                scheme="mysql+pymysql",
                host="localhost",
                port=3306,
                database="test",
                user="admin",
                password="password",
            )
        )
        raw = sample_store_resource.model_dump()
        m = SampleStoreResource.model_validate(raw)
        assert m.config.specification.storageLocation is not None


@pytest.fixture
def csv_sample_store_from_reference(
    csv_sample_store_reference: SampleStoreReference,
) -> orchestrator.core.samplestore.csv.CSVSampleStore:
    return initialize_sample_store_from_reference(reference=csv_sample_store_reference)


def test_csv_sample_store_from_reference(
    csv_sample_store_from_reference: CSVSampleStore,
    csv_sample_store_reference: SampleStoreReference,
) -> None:
    """Test creating a single sample store based on a description"""

    assert (
        csv_sample_store_from_reference.sourceDescription.generatorIdentifier
        == csv_sample_store_reference.parameters["generatorIdentifier"]
    )

    # sample_store is directly created in the fixture - we expect it to be passive as
    # is created from a CSV file
    assert csv_sample_store_from_reference.isPassive

    assert csv_sample_store_from_reference.numberOfEntities == 5000

    assert (
        csv_sample_store_from_reference.sourceDescription.model_dump()
        == csv_sample_store_reference.parameters
    )
    assert (
        csv_sample_store_from_reference.storageLocation
        == csv_sample_store_reference.storageLocation
    )


def test_sample_store_smiles(pfas_sample_store: SQLSampleStore) -> None:
    """Test creating a single sample store based on a description"""

    # sample_store is directly created in the fixture - we expect it to be passive as
    # is created from a CSV file
    assert not pfas_sample_store.isPassive

    assert pfas_sample_store.numberOfEntities == 101


def test_sample_store_config_file_valid(valid_sample_store_config_file: str) -> None:
    import pathlib

    valid_sample_store_config_file = pathlib.Path(valid_sample_store_config_file)
    orchestrator.core.samplestore.config.SampleStoreConfiguration.model_validate(
        yaml.safe_load(valid_sample_store_config_file.read_text())
    )


def test_sample_store_specification(
    sample_store_module_and_storage_location: tuple[
        SampleStoreModuleConf, SQLStoreConfiguration
    ],
) -> None:
    """Test we can create, dump and load a SampleStoreSpecification"""

    module, location = sample_store_module_and_storage_location

    sample_store = orchestrator.core.samplestore.config.SampleStoreSpecification(
        module=module,
        storageLocation=location,
    )

    assert sample_store.storageLocation is not None
    assert sample_store.module is not None

    raw = sample_store.model_dump()

    assert raw.get("storageLocation") is not None
    assert raw.get("module") is not None

    # This test is to ensure the serialization of the storageLocation field of SampleStoreSpecification
    # is preserving fields of ResourceLocation subclasses
    if isinstance(
        sample_store.storageLocation,
        orchestrator.utilities.location.SQLStoreConfiguration,
    ):
        assert sample_store.storageLocation.database is not None
        assert raw["storageLocation"].get("database") is not None

    m = orchestrator.core.samplestore.config.SampleStoreSpecification.model_validate(
        raw
    )
    assert m.storageLocation is not None

    # Test storageLocation can be optional
    # First, without passing storageLocation
    sample_store = orchestrator.core.samplestore.config.SampleStoreSpecification(
        module=module
    )
    # Next, passing storageLocation = None - this triggers different pydantic path
    sample_store = orchestrator.core.samplestore.config.SampleStoreSpecification(
        module=module, storageLocation=None
    )

    raw = sample_store.model_dump()
    print(raw)
    m = orchestrator.core.samplestore.config.SampleStoreSpecification.model_validate(
        raw
    )

    assert m.storageLocation is None


def test_sample_store_correct_class(
    sample_store_test_data: tuple[
        dict[str, Any], SQLStoreConfiguration | FilePathLocation
    ],
) -> None:
    config, expectedClass = sample_store_test_data

    sample_store = (
        orchestrator.core.samplestore.config.SampleStoreSpecification.model_validate(
            config
        )
    )
    assert isinstance(sample_store.storageLocation, expectedClass)


def test_base_entity_with_constitutive_property_values(
    ml_multi_cloud_csv_sample_store: CSVSampleStore,
) -> None:
    ents = ml_multi_cloud_csv_sample_store.entitiesWithConstitutivePropertyValues(
        [
            ConstitutivePropertyValue(
                value=0,
                property=ConstitutivePropertyDescriptor(identifier="cpu_family"),
            ),
            ConstitutivePropertyValue(
                value=3, property=ConstitutivePropertyDescriptor(identifier="nodes")
            ),
        ]
    )

    assert ents
    assert len(ents) == 5


def test_csv_sample_store_type_parsing(
    ml_multi_cloud_csv_sample_store: CSVSampleStore,
) -> None:
    entity: Entity = ml_multi_cloud_csv_sample_store.entities[0]
    for prop_id in ["cpu_family", "vcpu_size", "nodes", "provider"]:
        assert entity.valueForConstitutivePropertyIdentifier(
            prop_id
        ), f"Expected the entity to have a constitutive property {prop_id}"

    for prop_id in ["wallClockRuntime", "status"]:
        op = ObservedProperty(
            experimentReference=ExperimentReference(
                experimentIdentifier="benchmark_performance",
                actuatorIdentifier="replay",
            ),
            targetProperty=AbstractPropertyDescriptor(identifier=prop_id),
        )
        assert entity.valuesForObservedPropertyIdentifier(
            op.identifier
        ), f"Expected the entity to have an observed property called {op.identifier}"

    assert (
        entity.valueForConstitutivePropertyIdentifier("cpu_family").valueType
        == ValueTypeEnum.NUMERIC_VALUE_TYPE
    )
    assert (
        entity.valueForConstitutivePropertyIdentifier("vcpu_size").valueType
        == ValueTypeEnum.NUMERIC_VALUE_TYPE
    )
    assert (
        entity.valueForConstitutivePropertyIdentifier("nodes").valueType
        == ValueTypeEnum.NUMERIC_VALUE_TYPE
    )
    assert (
        entity.valueForConstitutivePropertyIdentifier("provider").valueType
        == ValueTypeEnum.STRING_VALUE_TYPE
    )

    values = entity.valuesForTargetProperty(Property(identifier="wallClockRuntime"))
    for v in values:
        assert v.valueType == ValueTypeEnum.NUMERIC_VALUE_TYPE

    values = entity.valuesForTargetProperty(Property(identifier="status"))
    for v in values:
        assert v.valueType == ValueTypeEnum.STRING_VALUE_TYPE


def test_csv_store_backward_compatibility_no_actuator_field(
    csv_sample_store_parameters: tuple[
        dict[str, str],
        dict[str, str | list[str] | list[dict[str, str | dict[str, str]]]],
    ],
) -> None:
    """Test that existing CSV configs without actuatorIdentifier field still work"""
    _location, params = csv_sample_store_parameters

    # Verify the fixture doesn't have actuatorIdentifier (simulating legacy config)
    for exp in params.get("experiments", []):
        assert "actuatorIdentifier" not in exp

    # Should create successfully with validation enabled
    # (replay actuator skips experiment validation)
    desc = CSVSampleStoreDescription.model_validate(params)

    # Should default to replay actuator
    catalog = desc.catalog
    for exp in catalog.experiments:
        assert exp.actuatorIdentifier == "replay"


def test_csv_from_csv_method_backward_compatibility() -> None:
    """Test that from_csv class method works with and without new parameters"""
    import os
    import tempfile

    import pandas as pd

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(
            {"id": ["entity1", "entity2"], "param1": [1, 2], "metric1": [100, 200]}
        )
        df.to_csv(f.name, index=False)
        csv_path = f.name

    try:
        # Test backward compatible usage
        # Uses replay actuator by default
        store1 = CSVSampleStore.from_csv(
            csvPath=csv_path,
            idColumn="id",
            experimentIdentifier="test_exp",
            observedPropertyColumns=["metric1"],
            constitutivePropertyColumns=["param1"],
        )
        assert store1 is not None
        assert len(store1.entities) == 2

        # Verify the actuator was set to replay by default
        catalog = store1.experimentCatalog()
        assert catalog.experiments[0].actuatorIdentifier == "replay"
    finally:
        # Cleanup
        if os.path.exists(csv_path):
            os.unlink(csv_path)


def test_csv_property_map_list_format() -> None:
    """Test that property maps can be specified as lists when column names match property names"""

    # Test with list format (simplified)
    params_list = {
        "generatorIdentifier": "test-gen",
        "identifierColumn": "id",
        "experiments": [
            {
                "experimentIdentifier": "test_exp",
                "observedPropertyMap": ["metric1", "metric2"],
                "constitutivePropertyMap": ["param1", "param2"],
            }
        ],
    }

    desc_list = CSVSampleStoreDescription.model_validate(params_list)
    exp_list = desc_list.experiments[0]

    # Verify list was converted to dict with matching keys and values
    assert exp_list.observedPropertyMap == {
        "metric1": "metric1",
        "metric2": "metric2",
    }
    assert exp_list.constitutivePropertyMap == {
        "param1": "param1",
        "param2": "param2",
    }

    # Test with dict format (traditional)
    params_dict = {
        "generatorIdentifier": "test-gen",
        "identifierColumn": "id",
        "experiments": [
            {
                "experimentIdentifier": "test_exp",
                "observedPropertyMap": {"metric1": "col1", "metric2": "col2"},
                "constitutivePropertyMap": {"param1": "p1", "param2": "p2"},
            }
        ],
    }

    desc_dict = CSVSampleStoreDescription.model_validate(params_dict)
    exp_dict = desc_dict.experiments[0]

    # Verify dict format is preserved
    assert exp_dict.observedPropertyMap == {"metric1": "col1", "metric2": "col2"}
    assert exp_dict.constitutivePropertyMap == {"param1": "p1", "param2": "p2"}


def test_csv_old_format_migration() -> None:
    """Test that old CSVSampleStoreDescription format is migrated to new format

    Old format (the only one stored before these changes):
    - constitutivePropertyColumns at top level
    - experiments list with propertyMap (not observedPropertyMap)
    - No constitutivePropertyMap in experiment descriptions
    """
    # Old format matching the stored format
    old_format = {
        "identifierColumn": "config",
        "generatorIdentifier": "multi-cloud-ml",
        "constitutivePropertyColumns": ["cpu_family", "vcpu_size", "nodes", "provider"],
        "experiments": [
            {
                "experimentIdentifier": "benchmark_performance",
                "propertyMap": {
                    "wallClockRuntime": "wallClockRuntime",
                    "status": "status",
                },
            }
        ],
    }

    desc = CSVSampleStoreDescription.model_validate(old_format)
    assert len(desc.experiments) == 1
    exp = desc.experiments[0]

    # Verify migration: propertyMap -> observedPropertyMap
    assert exp.observedPropertyMap == {
        "wallClockRuntime": "wallClockRuntime",
        "status": "status",
    }

    # Verify migration: constitutivePropertyColumns -> constitutivePropertyMap
    assert exp.constitutivePropertyMap == {
        "cpu_family": "cpu_family",
        "vcpu_size": "vcpu_size",
        "nodes": "nodes",
        "provider": "provider",
    }

    # Verify it's an ExternalExperimentDescription (replay actuator)
    assert exp.actuatorIdentifier == "replay"
    assert exp.experimentIdentifier == "benchmark_performance"

    # Verify constitutivePropertyColumns was removed from top level
    assert "constitutivePropertyColumns" not in desc.model_dump()
