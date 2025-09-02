# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging

import orchestrator.utilities.location
from orchestrator.core.samplestore.base import ExperimentDescription
from orchestrator.core.samplestore.csv import (
    CSVSampleStore,
    CSVSampleStoreDescription,
)


class HOPV(CSVSampleStore):

    @staticmethod
    def validate_parameters(parameters=None):

        properties = ["homo", "lumo", "pce", "voc", "jsc"]

        insilico = ExperimentDescription(
            experimentIdentifier="insilico-pv-property-exp",
            propertyMap={p: f"{p}_calc" for p in properties},
        )

        exp = ExperimentDescription(
            experimentIdentifier="real-pv-property-exp",
            propertyMap={p: f"{p}_exp" for p in properties},
        )

        parameters["experiments"] = [insilico, exp]
        parameters["identifierColumn"] = "smiles"
        parameters["constitutivePropertyColumns"] = ["smiles"]

        return CSVSampleStoreDescription.model_validate(parameters)

    def __init__(
        self,
        storageLocation: orchestrator.utilities.location.FilePathLocation,
        parameters: CSVSampleStoreDescription,
    ):
        """

        :param parameters: A dictionary containing one field "data-file" which is the location of the HOPV CSV
        """

        super().__init__(storageLocation, parameters)
        self.log = logging.getLogger("HOPVSampleStore")
