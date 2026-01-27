# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""Tests for CSV actuator validation functionality"""

import os
import tempfile
from collections.abc import Generator

import pandas as pd
import pytest

from orchestrator.core.samplestore.csv import CSVSampleStore, CSVSampleStoreDescription
from orchestrator.modules.actuators.catalog import ExperimentCatalog
from orchestrator.modules.actuators.registry import (
    ActuatorRegistry,
    UnknownActuatorError,
    UnknownExperimentError,
)
from orchestrator.schema.experiment import Experiment
from orchestrator.utilities.location import FilePathLocation


@pytest.fixture
def mock_actuator_catalog() -> ExperimentCatalog:
    """Creates a mock actuator catalog for testing"""
    experiment = Experiment.experimentWithAbstractPropertyIdentifiers(
        identifier="test_experiment",
        actuatorIdentifier="test_actuator",
        targetProperties=["output_metric"],
        requiredConstitutiveProperties=["input_param"],
    )

    return ExperimentCatalog(
        experiments={experiment.identifier: experiment},
        catalogIdentifier="test_actuator",
    )


@pytest.fixture
def setup_test_actuator(
    mock_actuator_catalog: ExperimentCatalog,
) -> Generator[ActuatorRegistry, None, None]:
    """Registers a test actuator in the global registry"""
    registry = ActuatorRegistry.globalRegistry()

    # Store original state
    original_catalog_map = registry.catalogIdentifierMap.copy()

    # Register test actuator catalog directly in the catalogIdentifierMap
    registry.catalogIdentifierMap["test_actuator"] = mock_actuator_catalog

    # Also need to add to actuatorIdentifierMap to avoid errors
    # Create a minimal mock actuator object with the required identifier attribute
    from unittest.mock import Mock

    mock_actuator = Mock()
    mock_actuator.identifier = "test_actuator"

    original_actuator_map = registry.actuatorIdentifierMap.copy()
    registry.actuatorIdentifierMap["test_actuator"] = mock_actuator

    yield registry

    # Restore original state
    registry.catalogIdentifierMap = original_catalog_map
    registry.actuatorIdentifierMap = original_actuator_map


class TestCSVActuatorValidation:
    """Test suite for CSV actuator validation"""

    def test_default_actuator_is_replay(self) -> None:
        """Test that actuator defaults to 'replay' when not specified"""
        desc = CSVSampleStoreDescription.model_validate(
            {
                "identifierColumn": "id",
                "constitutivePropertyColumns": ["input_param"],
                "experiments": [
                    {
                        "experimentIdentifier": "test_exp",
                        "propertyMap": {"output": "output_col"},
                    }
                ],
            }
        )

        catalog = desc.catalog
        exp = catalog.experiments[0]
        assert exp.actuatorIdentifier == "replay"

    def test_actuator_specified_uses_provided_value(self) -> None:
        """Test that specified actuator is used"""
        # Use replay actuator since we don't have custom_actuator registered
        # This test verifies that the actuatorIdentifier field is respected
        desc = CSVSampleStoreDescription.model_validate(
            {
                "identifierColumn": "id",
                "experiments": [
                    {
                        "experimentIdentifier": "test_exp",
                        "actuatorIdentifier": "replay",
                        "observedPropertyMap": {"output": "output_col"},
                        "constitutivePropertyMap": ["input_param"],
                    }
                ],
            }
        )

        catalog = desc.catalog
        exp = catalog.experiments[0]
        assert exp.actuatorIdentifier == "replay"

    def test_validation_enabled_by_default(self) -> None:
        """Test that validation is enabled by default"""
        # Should raise an error because validation is enabled by default
        # and the actuator doesn't exist
        with pytest.raises(
            UnknownActuatorError, match="No actuator called nonexistent_actuator"
        ):
            CSVSampleStoreDescription.model_validate(
                {
                    "identifierColumn": "id",
                    "experiments": [
                        {
                            "experimentIdentifier": "nonexistent_exp",
                            "actuatorIdentifier": "nonexistent_actuator",
                            "observedPropertyMap": {"output": "output_col"},
                            "constitutivePropertyMap": ["input_param"],
                        }
                    ],
                }
            )

    def test_validation_can_be_disabled(self) -> None:
        """Test that validation can be explicitly disabled"""
        # Note: Validation is now automatic based on actuator type
        # External (replay) actuators skip validation, internal actuators validate
        # This test is kept for backward compatibility but behavior has changed
        desc = CSVSampleStoreDescription.model_validate(
            {
                "identifierColumn": "id",
                "experiments": [
                    {
                        "experimentIdentifier": "nonexistent_exp",
                        "actuatorIdentifier": "replay",  # Replay skips validation
                        "observedPropertyMap": {"output": "output_col"},
                        "constitutivePropertyMap": ["input_param"],
                    }
                ],
            }
        )

        # Should not raise an error because replay actuator skips validation
        assert desc.experiments[0].actuatorIdentifier == "replay"

    def test_validation_with_valid_actuator_and_experiment(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test successful validation with valid actuator and experiment (validation on by default)"""
        desc = CSVSampleStoreDescription.model_validate(
            {
                "identifierColumn": "id",
                "experiments": [
                    {
                        "experimentIdentifier": "test_experiment",
                        "actuatorIdentifier": "test_actuator",
                        "observedPropertyMap": {"output_metric": "output_col"},
                        "constitutivePropertyMap": ["input_param"],
                    }
                ],
            }
        )

        # Should not raise an error (validation happens automatically for internal actuators)
        assert desc.experiments[0].actuatorIdentifier == "test_actuator"

    def test_validation_fails_with_unknown_actuator(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test validation fails when actuator doesn't exist (validation on by default)"""
        with pytest.raises(
            UnknownActuatorError, match="No actuator called unknown_actuator"
        ):
            CSVSampleStoreDescription.model_validate(
                {
                    "identifierColumn": "id",
                    "experiments": [
                        {
                            "experimentIdentifier": "test_experiment",
                            "actuatorIdentifier": "unknown_actuator",
                            "observedPropertyMap": {"output_metric": "output_col"},
                            "constitutivePropertyMap": ["input_param"],
                        }
                    ],
                }
            )

    def test_validation_fails_with_unknown_experiment(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test validation fails when experiment doesn't exist in actuator (validation on by default)"""
        with pytest.raises(UnknownExperimentError, match=r"test_actuator\.unknown_exp"):
            CSVSampleStoreDescription.model_validate(
                {
                    "identifierColumn": "id",
                    "experiments": [
                        {
                            "experimentIdentifier": "unknown_exp",
                            "actuatorIdentifier": "test_actuator",
                            "observedPropertyMap": {"output_metric": "output_col"},
                            "constitutivePropertyMap": ["input_param"],
                        }
                    ],
                }
            )

    def test_validation_fails_with_missing_required_properties(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test validation fails when required constitutive properties are missing (validation on by default)"""
        import pydantic

        with pytest.raises(
            pydantic.ValidationError,
            match="constitutivePropertyMap contains invalid constitutive property identifiers",
        ):
            CSVSampleStoreDescription.model_validate(
                {
                    "identifierColumn": "id",
                    "experiments": [
                        {
                            "experimentIdentifier": "test_experiment",
                            "actuatorIdentifier": "test_actuator",
                            "observedPropertyMap": {"output_metric": "output_col"},
                            "constitutivePropertyMap": [
                                "wrong_param"
                            ],  # Invalid - missing 'input_param'
                        }
                    ],
                }
            )

    def test_validation_fails_with_parameterized_experiment(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test validation fails when experiment identifier is parameterized

        Parameterized experiments will naturally fail because experimentForReference
        won't find them in the catalog (they need special handling in the future).
        """
        with pytest.raises(
            UnknownExperimentError,
            match=r"test_actuator.test_experiment-param1.value1-param2.value2",
        ):
            CSVSampleStoreDescription.model_validate(
                {
                    "identifierColumn": "id",
                    "experiments": [
                        {
                            "experimentIdentifier": "test_experiment-param1.value1-param2.value2",
                            "actuatorIdentifier": "test_actuator",
                            "observedPropertyMap": {"output_metric": "output_col"},
                            "constitutivePropertyMap": ["input_param"],
                        }
                    ],
                }
            )

    def test_backward_compatibility_no_validation(self) -> None:
        """Test that existing configs without actuatorIdentifier still work (migrated format)"""
        # Test old format migration
        desc = CSVSampleStoreDescription.model_validate(
            {
                "identifierColumn": "id",
                "generatorIdentifier": "test_gen",
                "constitutivePropertyColumns": ["param1", "param2"],
                "experiments": [
                    {
                        "experimentIdentifier": "legacy_exp",
                        "propertyMap": {"metric1": "col1", "metric2": "col2"},
                    }
                ],
            }
        )

        catalog = desc.catalog
        assert len(catalog.experiments) == 1
        assert catalog.experiments[0].actuatorIdentifier == "replay"
        assert catalog.experiments[0].identifier == "legacy_exp"

    def test_replay_actuator_skips_experiment_validation(self) -> None:
        """Test that replay actuator skips experiment validation"""
        desc = CSVSampleStoreDescription.model_validate(
            {
                "identifierColumn": "id",
                "generatorIdentifier": "test_gen",
                "experiments": [
                    {
                        "experimentIdentifier": "nonexistent_experiment",
                        "actuatorIdentifier": "replay",  # Explicitly set to replay
                        "observedPropertyMap": {"metric1": "col1", "metric2": "col2"},
                        "constitutivePropertyMap": ["param1", "param2"],
                    }
                ],
            }
        )

        # Should not raise an error - replay actuator skips experiment validation
        catalog = desc.catalog
        assert len(catalog.experiments) == 1
        assert catalog.experiments[0].actuatorIdentifier == "replay"
        assert catalog.experiments[0].identifier == "nonexistent_experiment"

    def test_mixed_actuators_in_experiments(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test that different experiments can have different actuators"""
        desc = CSVSampleStoreDescription.model_validate(
            {
                "identifierColumn": "id",
                "experiments": [
                    {
                        "experimentIdentifier": "test_experiment",
                        "actuatorIdentifier": "test_actuator",
                        "observedPropertyMap": {"output_metric": "col1"},
                        "constitutivePropertyMap": ["input_param"],
                    },
                    {
                        "experimentIdentifier": "legacy_experiment",
                        "actuatorIdentifier": "replay",  # Explicitly set to replay
                        "observedPropertyMap": {"other_metric": "col2"},
                        "constitutivePropertyMap": ["input_param"],
                    },
                ],
            }
        )

        catalog = desc.catalog
        assert len(catalog.experiments) == 2

        # Find experiments by identifier
        exps_by_id = {e.identifier: e for e in catalog.experiments}
        assert exps_by_id["test_experiment"].actuatorIdentifier == "test_actuator"
        assert exps_by_id["legacy_experiment"].actuatorIdentifier == "replay"

    def test_auto_infer_constitutive_properties(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test that constitutive properties are auto-inferred from experiment definition"""
        desc = CSVSampleStoreDescription.model_validate(
            {
                "identifierColumn": "id",
                "generatorIdentifier": "test_gen",
                # constitutivePropertyMap not specified - should be inferred
                "experiments": [
                    {
                        "experimentIdentifier": "test_experiment",
                        "actuatorIdentifier": "test_actuator",
                        "observedPropertyMap": {"output_metric": "output_col"},
                    },
                ],
            }
        )

        # Should have auto-inferred constitutive properties from experiment
        constitutive_prop_ids = [
            prop.identifier for prop in desc.constitutiveProperties
        ]
        assert "input_param" in constitutive_prop_ids

    def test_auto_infer_property_map_target_format(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test that property map is auto-inferred using target properties"""
        desc = CSVSampleStoreDescription.model_validate(
            {
                "identifierColumn": "id",
                "generatorIdentifier": "test_gen",
                "experiments": [
                    {
                        "experimentIdentifier": "test_experiment",
                        "actuatorIdentifier": "test_actuator",
                        "constitutivePropertyMap": ["input_param"],
                        # observedPropertyMap not specified - should be inferred
                        "propertyFormat": "target",  # Use target properties
                    },
                ],
            }
        )

        # Should have auto-inferred property map from target properties
        exp_desc = desc.experiments[0]
        assert "output_metric" in exp_desc.observedPropertyMap
        # For target format, property maps to itself
        assert exp_desc.observedPropertyMap["output_metric"] == "output_metric"

    def test_auto_infer_property_map_observed_format(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test that property map is auto-inferred using observed properties"""
        desc = CSVSampleStoreDescription.model_validate(
            {
                "identifierColumn": "id",
                "generatorIdentifier": "test_gen",
                "experiments": [
                    {
                        "experimentIdentifier": "test_experiment",
                        "actuatorIdentifier": "test_actuator",
                        "constitutivePropertyMap": ["input_param"],
                        # observedPropertyMap not specified - should be inferred
                        "propertyFormat": "observed",  # Use observed properties
                    },
                ],
            }
        )

        # Should have auto-inferred property map from observed properties
        exp_desc = desc.experiments[0]
        # Keys are target property identifiers, values are observed property identifiers
        assert "output_metric" in exp_desc.observedPropertyMap
        # Observed property identifier format: experiment_id-target_property_id
        expected_obs_prop_id = "test_experiment-output_metric"
        assert exp_desc.observedPropertyMap["output_metric"] == expected_obs_prop_id

    def test_partial_specification_with_inference(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test that explicit specifications take precedence over inference"""
        desc = CSVSampleStoreDescription.model_validate(
            {
                "identifierColumn": "id",
                "generatorIdentifier": "test_gen",
                "experiments": [
                    {
                        "experimentIdentifier": "test_experiment",
                        "actuatorIdentifier": "test_actuator",
                        "constitutivePropertyMap": [
                            "input_param"
                        ],  # Explicit - must match experiment requirements
                        "observedPropertyMap": {
                            "output_metric": "custom_col"
                        },  # Explicit custom mapping
                    },
                ],
            }
        )

        # Should use explicit specifications, not inferred ones
        constitutive_prop_ids = [
            prop.identifier for prop in desc.constitutiveProperties
        ]
        assert "input_param" in constitutive_prop_ids
        # observedPropertyMap was explicitly set to use custom column name
        assert desc.experiments[0].observedPropertyMap == {
            "output_metric": "custom_col"
        }

    def test_csv_column_validation_with_auto_inference(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test that CSV column validation works with auto-inferred properties"""
        # Create a CSV file with missing columns
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            df = pd.DataFrame(
                {
                    "id": ["entity1", "entity2"],
                    "input_param": [1, 2],
                    # Missing "output_metric" column
                }
            )
            df.to_csv(temp_file.name, index=False)
            csv_path = temp_file.name

        try:
            # Create description with auto-inference
            desc = CSVSampleStoreDescription.model_validate(
                {
                    "identifierColumn": "id",
                    "generatorIdentifier": "test_gen",
                    # Auto-infer both constitutive and property map
                    "experiments": [
                        {
                            "experimentIdentifier": "test_experiment",
                            "actuatorIdentifier": "test_actuator",
                        },
                    ],
                }
            )

            # Should raise error when creating CSVSampleStore due to missing column
            with pytest.raises(ValueError, match="missing required columns"):
                CSVSampleStore(
                    storageLocation=FilePathLocation(path=csv_path), parameters=desc
                )
        finally:
            # Clean up
            os.unlink(csv_path)

    def test_csv_column_validation_success_with_auto_inference(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test that CSV loads successfully when all required columns are present"""
        # Create a CSV file with all required columns
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            df = pd.DataFrame(
                {
                    "id": ["entity1", "entity2"],
                    "input_param": [1, 2],
                    "output_metric": [100, 200],
                }
            )
            df.to_csv(temp_file.name, index=False)
            csv_path = temp_file.name

        try:
            # Create description with auto-inference
            desc = CSVSampleStoreDescription.model_validate(
                {
                    "identifierColumn": "id",
                    "generatorIdentifier": "test_gen",
                    # Auto-infer both constitutive and property map
                    "experiments": [
                        {
                            "experimentIdentifier": "test_experiment",
                            "actuatorIdentifier": "test_actuator",
                        },
                    ],
                }
            )

            # Should create successfully
            store = CSVSampleStore(
                storageLocation=FilePathLocation(path=csv_path), parameters=desc
            )
            assert store is not None
            assert len(store.entities) == 2
        finally:
            # Clean up
            os.unlink(csv_path)

    def test_replay_actuator_requires_explicit_constitutive_properties(self) -> None:
        """Test that replay actuator requires explicit constitutivePropertyMap specification"""
        import pydantic

        with pytest.raises(
            pydantic.ValidationError,
            match="constitutivePropertyMap",
        ):
            CSVSampleStoreDescription.model_validate(
                {
                    "identifierColumn": "id",
                    "generatorIdentifier": "test_gen",
                    # No constitutivePropertyMap specified
                    "experiments": [
                        {
                            "experimentIdentifier": "test_experiment",
                            "actuatorIdentifier": "replay",  # Uses replay
                            "observedPropertyMap": {"output": "output_col"},
                        },
                    ],
                }
            )

    def test_non_replay_actuator_validates_columns_exist_in_csv(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test that non-replay actuators validate column names exist in CSV when using from_csv()"""
        # Create a CSV file with some columns
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            df = pd.DataFrame(
                {
                    "id": ["entity1", "entity2"],
                    "input_param": [1, 2],
                    "output_metric": [100, 200],
                }
            )
            df.to_csv(temp_file.name, index=False)
            csv_path = temp_file.name

        try:
            # Test 1: Valid columns - should succeed
            store1 = CSVSampleStore.from_csv(
                csvPath=csv_path,
                idColumn="id",
                experimentIdentifier="test_experiment",
                actuatorIdentifier="test_actuator",
                observedPropertyColumns=["output_metric"],
                constitutivePropertyColumns=["input_param"],
            )
            assert store1 is not None
            assert len(store1.entities) == 2

            # Test 2: Missing observed property column - should fail
            with pytest.raises(
                ValueError,
                match="observedPropertyMap contains invalid target property identifiers",
            ):
                CSVSampleStore.from_csv(
                    csvPath=csv_path,
                    idColumn="id",
                    experimentIdentifier="test_experiment",
                    actuatorIdentifier="test_actuator",
                    observedPropertyColumns=["nonexistent_column"],
                    constitutivePropertyColumns=["input_param"],
                )

            # Test 3: Missing constitutive property column - should fail
            with pytest.raises(
                ValueError,
                match="constitutivePropertyMap contains invalid constitutive property identifiers",
            ):
                CSVSampleStore.from_csv(
                    csvPath=csv_path,
                    idColumn="id",
                    experimentIdentifier="test_experiment",
                    actuatorIdentifier="test_actuator",
                    observedPropertyColumns=["output_metric"],
                    constitutivePropertyColumns=["nonexistent_column"],
                )

            # Test 4: Missing id column - should fail
            with pytest.raises(
                ValueError,
                match="is missing required columns:",
            ):
                CSVSampleStore.from_csv(
                    csvPath=csv_path,
                    idColumn="nonexistent_id",
                    experimentIdentifier="test_experiment",
                    actuatorIdentifier="test_actuator",
                    observedPropertyColumns=["output_metric"],
                    constitutivePropertyColumns=["input_param"],
                )
        finally:
            # Clean up
            os.unlink(csv_path)

    def test_non_replay_actuator_column_validation_with_auto_inference(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test that column validation works when property maps are auto-inferred for non-replay actuators"""
        # Create a CSV file with columns matching experiment property identifiers
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            df = pd.DataFrame(
                {
                    "id": ["entity1", "entity2"],
                    "input_param": [1, 2],
                    "output_metric": [100, 200],  # Matches target property identifier
                }
            )
            df.to_csv(temp_file.name, index=False)
            csv_path = temp_file.name

        try:
            # Test with propertyFormat="target" (default) - column names match target property identifiers
            store1 = CSVSampleStore.from_csv(
                csvPath=csv_path,
                idColumn="id",
                experimentIdentifier="test_experiment",
                actuatorIdentifier="test_actuator",
                propertyFormat="target",
                # Property maps will be auto-inferred
            )
            assert store1 is not None
            assert len(store1.entities) == 2

            # Test with propertyFormat="observed" - column names should match observed property identifiers
            # Observed property identifier format: experiment_id-target_property_id
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as temp_file2:
                df2 = pd.DataFrame(
                    {
                        "id": ["entity1", "entity2"],
                        "input_param": [1, 2],
                        "test_experiment-output_metric": [
                            100,
                            200,
                        ],  # Matches observed property identifier
                    }
                )
                df2.to_csv(temp_file2.name, index=False)
                csv_path2 = temp_file2.name

            try:
                store2 = CSVSampleStore.from_csv(
                    csvPath=csv_path2,
                    idColumn="id",
                    experimentIdentifier="test_experiment",
                    actuatorIdentifier="test_actuator",
                    propertyFormat="observed",
                    # Property maps will be auto-inferred
                )
                assert store2 is not None
                assert len(store2.entities) == 2
            finally:
                os.unlink(csv_path2)

            # Test with wrong column name for observed format - should fail
            with pytest.raises(ValueError, match="missing required columns"):
                CSVSampleStore.from_csv(
                    csvPath=csv_path,  # Uses output_metric, not test_experiment-output_metric
                    idColumn="id",
                    experimentIdentifier="test_experiment",
                    actuatorIdentifier="test_actuator",
                    propertyFormat="observed",
                )
        finally:
            # Clean up
            os.unlink(csv_path)

    def test_sqlsamplestore_from_csv_with_non_replay_actuator(
        self, setup_test_actuator: ActuatorRegistry
    ) -> None:
        """Test that SQLSampleStore.from_csv() properly passes through actuatorIdentifier and propertyFormat"""
        from orchestrator.core.samplestore.sql import SQLSampleStore
        from orchestrator.utilities.location import SQLiteStoreConfiguration

        # Create a CSV file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            df = pd.DataFrame(
                {
                    "id": ["entity1", "entity2"],
                    "input_param": [1, 2],
                    "output_metric": [100, 200],
                }
            )
            df.to_csv(temp_file.name, index=False)
            csv_path = temp_file.name

        # Create a temporary SQLite database file
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db_file:
            db_path = temp_db_file.name

        try:
            # Create a SQL store configuration using temporary file
            store_config = SQLiteStoreConfiguration(scheme="sqlite", path=db_path)

            # Test that parameters are passed through correctly
            # The CSV store creation should validate columns exist
            # If columns are missing, we should get a ValueError from CSV parsing
            sql_store = SQLSampleStore.from_csv(
                csvPath=csv_path,
                idColumn="id",
                storeConfiguration=store_config,
                experimentIdentifier="test_experiment",
                actuatorIdentifier="test_actuator",
                observedPropertyColumns=["output_metric"],
                constitutivePropertyColumns=["input_param"],
                propertyFormat="target",
            )
            # If we get here, the CSV store was created successfully with correct parameters
            assert sql_store is not None

            # Test with propertyFormat="observed" and matching column names
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as temp_file2:
                df2 = pd.DataFrame(
                    {
                        "id": ["entity1", "entity2"],
                        "input_param": [1, 2],
                        "test_experiment-output_metric": [100, 200],
                    }
                )
                df2.to_csv(temp_file2.name, index=False)
                csv_path2 = temp_file2.name

            try:
                sql_store2 = SQLSampleStore.from_csv(
                    csvPath=csv_path2,
                    idColumn="id",
                    storeConfiguration=store_config,
                    experimentIdentifier="test_experiment",
                    actuatorIdentifier="test_actuator",
                    propertyFormat="observed",
                )
                assert sql_store2 is not None
            finally:
                os.unlink(csv_path2)

            # Test that missing columns are caught (column validation happens during CSV store creation)
            with pytest.raises(
                ValueError,
                match=r"is missing required columns: \['test_experiment-output_metric'\]",
            ):
                SQLSampleStore.from_csv(
                    csvPath=csv_path,  # Missing test_experiment-output_metric column
                    idColumn="id",
                    storeConfiguration=store_config,
                    experimentIdentifier="test_experiment",
                    actuatorIdentifier="test_actuator",
                    propertyFormat="observed",
                )

        finally:
            # Clean up CSV file and database file
            if os.path.exists(csv_path):
                os.unlink(csv_path)
            if os.path.exists(db_path):
                os.unlink(db_path)
