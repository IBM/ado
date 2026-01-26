# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging

import orchestrator.utilities.location
from orchestrator.core.samplestore.base import ExternalExperimentDescription
from orchestrator.core.samplestore.csv import (
    CSVSampleStore,
    CSVSampleStoreDescription,
)


class HOPV(CSVSampleStore):

    @staticmethod
    def validate_parameters(
        parameters: dict,
    ) -> CSVSampleStoreDescription:

        properties = ["homo", "lumo", "pce", "voc", "jsc"]

        insilico = ExternalExperimentDescription(
            experimentIdentifier="insilico-pv-property-exp",
            observedPropertyMap={p: f"{p}_calc" for p in properties},
            constitutivePropertyMap=["smiles"],
        )

        exp = ExternalExperimentDescription(
            experimentIdentifier="real-pv-property-exp",
            observedPropertyMap={p: f"{p}_exp" for p in properties},
            constitutivePropertyMap=["smiles"],
        )

        parameters["experiments"] = [insilico, exp]
        parameters["identifierColumn"] = "smiles"

        return CSVSampleStoreDescription.model_validate(parameters)

    def __init__(
        self,
        storageLocation: orchestrator.utilities.location.FilePathLocation,
        parameters: CSVSampleStoreDescription,
    ) -> None:
        """

        :param parameters: A dictionary containing one field "data-file" which is the location of the HOPV CSV
        """

        super().__init__(storageLocation, parameters)
        self.log = logging.getLogger("HOPVSampleStore")
