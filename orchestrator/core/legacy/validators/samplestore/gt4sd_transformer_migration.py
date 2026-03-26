# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validator for migrating GT4SDTransformer to CSVSampleStore"""

from orchestrator.core.legacy.registry import legacy_validator
from orchestrator.core.legacy.utils import (
    get_nested_value,
    has_nested_field,
    set_nested_value,
)
from orchestrator.core.resources import CoreResourceKinds


@legacy_validator(
    identifier="samplestore_gt4sd_transformer_to_csv",
    resource_type=CoreResourceKinds.SAMPLESTORE,
    deprecated_field_paths=[
        "config.copyFrom.0.module.moduleClass",
        "config.copyFrom.0.module.moduleName",
    ],
    deprecated_from_version="1.3.5",
    removed_from_version="1.6.0",
    description="Converts GT4SDTransformer plugin to CSVSampleStore with explicit parameters",
)
def migrate_gt4sd_transformer_to_csv(data: dict) -> dict:
    """Migrate GT4SDTransformer plugin usage to CSVSampleStore with explicit parameters

    The GT4SDTransformer class was a thin wrapper around CSVSampleStore that
    automatically filled in parameters. This validator converts configurations
    using GT4SDTransformer to use CSVSampleStore directly with explicit parameters.

    Old format:
        config:
            copyFrom:
                - module:
                    moduleClass: GT4SDTransformer
                    moduleName: orchestrator.plugins.samplestores.gt4sd
                  parameters:
                    generatorIdentifier: 'gt4sd-pfas-transformer-model-one'

    New format:
        config:
            copyFrom:
                - module:
                    moduleClass: CSVSampleStore
                    moduleName: orchestrator.core.samplestore.csv
                  parameters:
                    generatorIdentifier: 'gt4sd-pfas-transformer-model-one'
                    identifierColumn: 'smiles'
                    experiments:
                      - experimentIdentifier: 'transformer-toxicity-inference-experiment'
                        observedPropertyMap:
                          logws: GenLogws
                          logd: GenLogd
                          loghl: GenLoghl
                          pka: GenPka
                          "biodegradation halflife": GenBiodeg
                          bcf: GenBcf
                          ld50: GenLd50
                          scscore: GenScscore
                        constitutivePropertyMap: [smiles]

    Args:
        data: The resource data dictionary

    Returns:
        The migrated resource data dictionary
    """

    if not isinstance(data, dict):
        return data

    # Property map from the old GT4SDTransformer class
    property_map = {
        "logws": "GenLogws",
        "logd": "GenLogd",
        "loghl": "GenLoghl",
        "pka": "GenPka",
        "biodegradation halflife": "GenBiodeg",
        "bcf": "GenBcf",
        "ld50": "GenLd50",
        "scscore": "GenScscore",
    }

    # Check config.copyFrom array for GT4SDTransformer usage
    if has_nested_field(data, "config.copyFrom"):
        copy_from = get_nested_value(data, "config.copyFrom")
        if isinstance(copy_from, list):
            for item in copy_from:
                if not isinstance(item, dict):
                    continue

                # Check if this is a GT4SDTransformer module
                module_class = get_nested_value(item, "module.moduleClass")
                module_name = get_nested_value(item, "module.moduleName")

                if (
                    module_class == "GT4SDTransformer"
                    and module_name == "orchestrator.plugins.samplestores.gt4sd"
                ):
                    # Update module class and name
                    set_nested_value(item, "module.moduleClass", "CSVSampleStore")
                    set_nested_value(
                        item, "module.moduleName", "orchestrator.core.samplestore.csv"
                    )

                    # Add explicit parameters that GT4SDTransformer provided automatically
                    if not has_nested_field(item, "parameters"):
                        set_nested_value(item, "parameters", {})

                    parameters = get_nested_value(item, "parameters")

                    # Add identifierColumn if not present
                    if (
                        isinstance(parameters, dict)
                        and "identifierColumn" not in parameters
                    ):
                        set_nested_value(item, "parameters.identifierColumn", "smiles")

                    # Add experiments configuration if not present
                    if isinstance(parameters, dict) and "experiments" not in parameters:
                        experiment_config = {
                            "experimentIdentifier": "transformer-toxicity-inference-experiment",
                            "observedPropertyMap": property_map,
                            "constitutivePropertyMap": ["smiles"],
                        }
                        set_nested_value(
                            item, "parameters.experiments", [experiment_config]
                        )

    return data


# Made with Bob
