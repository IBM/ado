# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
from types import MappingProxyType

import orchestrator.utilities.location
from orchestrator.core.samplestore.base import ExternalExperimentDescription
from orchestrator.core.samplestore.csv import (
    CSVSampleStore,
    CSVSampleStoreDescription,
)


def fill_gt4sd_transformer_csv_parameters(parameters: dict) -> dict:

    experimentDescription = ExternalExperimentDescription(
        experimentIdentifier="transformer-toxicity-inference-experiment",
        observedPropertyMap=dict(GT4SDTransformer.propertyMap),
        constitutivePropertyMap=["smiles"],
    )

    parameters["experiments"] = [experimentDescription]
    parameters["identifierColumn"] = "smiles"

    return parameters


class GT4SDTransformer(CSVSampleStore):
    propertyMap = MappingProxyType(
        {
            "logws": "GenLogws",
            "logd": "GenLogd",
            "loghl": "GenLoghl",
            "pka": "GenPka",
            "biodegradation halflife": "GenBiodeg",
            "bcf": "GenBcf",
            "ld50": "GenLd50",
            "scscore": "GenScscore",
        }
    )

    @staticmethod
    def validate_parameters(
        parameters: dict,
    ) -> CSVSampleStoreDescription:
        parameters = fill_gt4sd_transformer_csv_parameters(parameters)
        log = logging.getLogger("GT4SDTransformerSampleStore")
        log.debug(
            f"Creating GT4SDTransformer sample store with parameters {parameters}"
        )

        return CSVSampleStoreDescription.model_validate(parameters)

    def __init__(
        self,
        storageLocation: orchestrator.utilities.location.FilePathLocation,
        parameters: CSVSampleStoreDescription,
    ) -> None:
        """

        :param parameters: A dictionary the fields of CSVSampleStoreDescription
        """

        super().__init__(
            storageLocation=storageLocation,
            parameters=parameters,
        )
        self.log = logging.getLogger("GT4SDTransformerSampleStore")
