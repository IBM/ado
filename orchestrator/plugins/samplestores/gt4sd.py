# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
from types import MappingProxyType

import orchestrator.utilities.location
from orchestrator.core.samplestore.base import ExperimentDescription
from orchestrator.core.samplestore.csv import (
    CSVSampleStore,
    CSVSampleStoreDescription,
)


def fill_gt4sd_transformer_csv_parameters(parameters):
    parameters = {} if parameters is None else parameters

    experimentDescription = ExperimentDescription(
        experimentIdentifier="transformer-toxicity-inference-experiment",
        propertyMap=GT4SDTransformer.propertyMap,
    )

    parameters["experiments"] = [experimentDescription]
    parameters["identifierColumn"] = "smiles"
    parameters["constitutivePropertyColumns"] = ["smiles"]

    return parameters


class GT4SDTransformer(CSVSampleStore):
    propertyMap = MappingProxyType(
        {
            "logws": "genlogws",
            "logd": "genlogd",
            "loghl": "genloghl",
            "pka": "genpka",
            "biodegradation halflife": "genbiodeg",
            "bcf": "genbcf",
            "ld50": "genld50",
            "scscore": "genscscore",
        }
    )

    @staticmethod
    def validate_parameters(parameters=None):
        # AP: parameters are used to instantiate a CSVSampleStoreDescription
        if parameters is None:
            raise ValueError("parameters cannot be None for GT4SDTransformer")

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
    ):
        """

        :param parameters: A dictionary the fields of CSVSampleStoreDescription
        """

        super().__init__(
            storageLocation=storageLocation,
            parameters=parameters,
        )
        self.log = logging.getLogger("GT4SDTransformerSampleStore")
