# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for GT4SDTransformer to CSVSampleStore migration validator"""

from orchestrator.core.legacy.registry import LegacyValidatorRegistry


class TestGT4SDTransformerMigration:
    """Test migrate_gt4sd_transformer_to_csv validator"""

    def test_migrates_gt4sd_transformer_to_csv_sample_store(
        self, legacy_validators_loaded: None
    ) -> None:
        """Test that GT4SDTransformer is migrated to CSVSampleStore with explicit parameters"""
        validator = LegacyValidatorRegistry.get_validator(
            "samplestore_gt4sd_transformer_to_csv"
        )
        assert validator is not None

        data = {
            "config": {
                "copyFrom": [
                    {
                        "module": {
                            "moduleClass": "GT4SDTransformer",
                            "moduleName": "orchestrator.plugins.samplestores.gt4sd",
                        },
                        "storageLocation": {
                            "path": "data/GM_Comparison/Transfromer/Sample_0/test_generations.csv"
                        },
                        "parameters": {
                            "generatorIdentifier": "gt4sd-pfas-transformer-model-one"
                        },
                    }
                ]
            }
        }

        result = validator.validator_function(data)

        # Check module class and name were updated
        copy_from = result["config"]["copyFrom"][0]
        assert copy_from["module"]["moduleClass"] == "CSVSampleStore"
        assert copy_from["module"]["moduleName"] == "orchestrator.core.samplestore.csv"

        # Check identifierColumn was added
        assert copy_from["parameters"]["identifierColumn"] == "smiles"

        # Check experiments configuration was added
        assert "experiments" in copy_from["parameters"]
        experiments = copy_from["parameters"]["experiments"]
        assert len(experiments) == 1
        assert (
            experiments[0]["experimentIdentifier"]
            == "transformer-toxicity-inference-experiment"
        )
        assert "observedPropertyMap" in experiments[0]
        assert "constitutivePropertyMap" in experiments[0]
        assert experiments[0]["constitutivePropertyMap"] == ["smiles"]

        # Check property map was correctly added
        property_map = experiments[0]["observedPropertyMap"]
        assert property_map["logws"] == "GenLogws"
        assert property_map["logd"] == "GenLogd"
        assert property_map["loghl"] == "GenLoghl"
        assert property_map["pka"] == "GenPka"
        assert property_map["biodegradation halflife"] == "GenBiodeg"
        assert property_map["bcf"] == "GenBcf"
        assert property_map["ld50"] == "GenLd50"
        assert property_map["scscore"] == "GenScscore"

        # Check original generatorIdentifier was preserved
        assert (
            copy_from["parameters"]["generatorIdentifier"]
            == "gt4sd-pfas-transformer-model-one"
        )

    def test_preserves_existing_identifier_column(
        self, legacy_validators_loaded: None
    ) -> None:
        """Test that existing identifierColumn is not overwritten"""
        validator = LegacyValidatorRegistry.get_validator(
            "samplestore_gt4sd_transformer_to_csv"
        )
        assert validator is not None

        data = {
            "config": {
                "copyFrom": [
                    {
                        "module": {
                            "moduleClass": "GT4SDTransformer",
                            "moduleName": "orchestrator.plugins.samplestores.gt4sd",
                        },
                        "parameters": {
                            "generatorIdentifier": "gt4sd-pfas-transformer-model-one",
                            "identifierColumn": "custom_id",
                        },
                    }
                ]
            }
        }

        result = validator.validator_function(data)

        # Check that custom identifierColumn was preserved
        assert (
            result["config"]["copyFrom"][0]["parameters"]["identifierColumn"]
            == "custom_id"
        )

    def test_preserves_existing_experiments(
        self, legacy_validators_loaded: None
    ) -> None:
        """Test that existing experiments configuration is not overwritten"""
        validator = LegacyValidatorRegistry.get_validator(
            "samplestore_gt4sd_transformer_to_csv"
        )
        assert validator is not None

        custom_experiments = [
            {
                "experimentIdentifier": "custom-experiment",
                "observedPropertyMap": {"prop1": "Prop1"},
                "constitutivePropertyMap": ["id"],
            }
        ]

        data = {
            "config": {
                "copyFrom": [
                    {
                        "module": {
                            "moduleClass": "GT4SDTransformer",
                            "moduleName": "orchestrator.plugins.samplestores.gt4sd",
                        },
                        "parameters": {
                            "generatorIdentifier": "gt4sd-pfas-transformer-model-one",
                            "experiments": custom_experiments,
                        },
                    }
                ]
            }
        }

        result = validator.validator_function(data)

        # Check that custom experiments were preserved
        assert (
            result["config"]["copyFrom"][0]["parameters"]["experiments"]
            == custom_experiments
        )

    def test_does_not_modify_other_module_classes(
        self, legacy_validators_loaded: None
    ) -> None:
        """Test that other module classes are not modified"""
        validator = LegacyValidatorRegistry.get_validator(
            "samplestore_gt4sd_transformer_to_csv"
        )
        assert validator is not None

        data = {
            "config": {
                "copyFrom": [
                    {
                        "module": {
                            "moduleClass": "CSVSampleStore",
                            "moduleName": "orchestrator.core.samplestore.csv",
                        },
                        "parameters": {"identifierColumn": "id"},
                    }
                ]
            }
        }

        result = validator.validator_function(data)

        # Check that nothing was changed
        copy_from = result["config"]["copyFrom"][0]
        assert copy_from["module"]["moduleClass"] == "CSVSampleStore"
        assert copy_from["module"]["moduleName"] == "orchestrator.core.samplestore.csv"
        assert copy_from["parameters"] == {"identifierColumn": "id"}

    def test_handles_multiple_copy_from_entries(
        self, legacy_validators_loaded: None
    ) -> None:
        """Test that validator handles multiple copyFrom entries correctly"""
        validator = LegacyValidatorRegistry.get_validator(
            "samplestore_gt4sd_transformer_to_csv"
        )
        assert validator is not None

        data = {
            "config": {
                "copyFrom": [
                    {
                        "module": {
                            "moduleClass": "GT4SDTransformer",
                            "moduleName": "orchestrator.plugins.samplestores.gt4sd",
                        },
                        "parameters": {"generatorIdentifier": "model-one"},
                    },
                    {
                        "module": {
                            "moduleClass": "CSVSampleStore",
                            "moduleName": "orchestrator.core.samplestore.csv",
                        },
                        "parameters": {"identifierColumn": "id"},
                    },
                ]
            }
        }

        result = validator.validator_function(data)

        # Check first entry was migrated
        first_entry = result["config"]["copyFrom"][0]
        assert first_entry["module"]["moduleClass"] == "CSVSampleStore"
        assert (
            first_entry["module"]["moduleName"] == "orchestrator.core.samplestore.csv"
        )
        assert "identifierColumn" in first_entry["parameters"]
        assert "experiments" in first_entry["parameters"]

        # Check second entry was not modified
        second_entry = result["config"]["copyFrom"][1]
        assert second_entry["module"]["moduleClass"] == "CSVSampleStore"
        assert second_entry["parameters"] == {"identifierColumn": "id"}

    def test_handles_missing_copy_from(self, legacy_validators_loaded: None) -> None:
        """Test that validator handles missing copyFrom field gracefully"""
        validator = LegacyValidatorRegistry.get_validator(
            "samplestore_gt4sd_transformer_to_csv"
        )
        assert validator is not None

        data = {"config": {"specification": {"module": {}}}}

        result = validator.validator_function(data)

        # Check that data was not modified
        assert result == data

    def test_handles_empty_copy_from(self, legacy_validators_loaded: None) -> None:
        """Test that validator handles empty copyFrom array gracefully"""
        validator = LegacyValidatorRegistry.get_validator(
            "samplestore_gt4sd_transformer_to_csv"
        )
        assert validator is not None

        data = {"config": {"copyFrom": []}}

        result = validator.validator_function(data)

        # Check that data was not modified
        assert result == data

    def test_handles_missing_parameters(self, legacy_validators_loaded: None) -> None:
        """Test that validator adds parameters if missing"""
        validator = LegacyValidatorRegistry.get_validator(
            "samplestore_gt4sd_transformer_to_csv"
        )
        assert validator is not None

        data = {
            "config": {
                "copyFrom": [
                    {
                        "module": {
                            "moduleClass": "GT4SDTransformer",
                            "moduleName": "orchestrator.plugins.samplestores.gt4sd",
                        },
                    }
                ]
            }
        }

        result = validator.validator_function(data)

        # Check that parameters were added
        copy_from = result["config"]["copyFrom"][0]
        assert "parameters" in copy_from
        assert "identifierColumn" in copy_from["parameters"]
        assert "experiments" in copy_from["parameters"]


# Made with Bob
