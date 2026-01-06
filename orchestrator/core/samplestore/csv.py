# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
from typing import TYPE_CHECKING

import pydantic

if TYPE_CHECKING:
    import pandas as pd

import orchestrator.core.samplestore.config
import orchestrator.utilities.location
from orchestrator.core.samplestore.base import (
    ExperimentDescription,
    PassiveSampleStore,
    SampleStoreDescription,
)
from orchestrator.modules.actuators.catalog import ExperimentCatalog
from orchestrator.schema.entity import Entity
from orchestrator.schema.observed_property import ObservedPropertyValue
from orchestrator.schema.property import (
    ConstitutivePropertyDescriptor,
)
from orchestrator.schema.property_value import ConstitutivePropertyValue
from orchestrator.schema.result import ValidMeasurementResult


class CSVSampleStoreDescription(SampleStoreDescription):
    identifierColumn: str = pydantic.Field(
        description="The header of the column that contains the entity ids"
    )

    generatorIdentifier: str | None = pydantic.Field(
        default=None,
        validate_default=True,
        description="The id of the entity generator",
    )
    constitutivePropertyColumns: list[str] = pydantic.Field(
        description="List of headers of columns containing constitutive properties",
    )

    @pydantic.field_validator("identifierColumn")
    def identifier_is_lowercase(cls, value):
        return value.lower()

    @property
    def constitutiveProperties(self) -> list[ConstitutivePropertyDescriptor]:

        # sourceDescription.constitutivePropertyColumns may be mixed-case - convert to  lowercase
        return [
            ConstitutivePropertyDescriptor(identifier=column_name.lower())
            for column_name in self.constitutivePropertyColumns
        ]

    @property
    def observedPropertyColumns(self):
        """Returns the headers of the columns containing the observed properties"""

        columns = []

        for op in self.observedProperties:
            expDescription = self.experimentDescriptionMap.get(
                op.experimentReference.experimentIdentifier
            )
            if expDescription is not None:
                columnHeader = expDescription.propertyMap[op.targetProperty.identifier]
                columns.append(columnHeader)

        return columns


class CSVSampleStore(PassiveSampleStore):
    """Reads entities and properties from a CSV file

    Entities are assumed to be in rows, properties are column headers

    """

    @staticmethod
    def validate_parameters(parameters=None):
        # AP: parameters are used to instantiate a CSVSampleStoreDescription
        if parameters is None:
            raise ValueError("parameters cannot be None for CSVSampleStore")

        return CSVSampleStoreDescription.model_validate(parameters)

    @staticmethod
    def storage_location_class():

        return orchestrator.utilities.location.FilePathLocation

    @classmethod
    def experimentCatalogFromReference(
        cls,
        reference: orchestrator.core.samplestore.config.SampleStoreReference = None,
    ) -> ExperimentCatalog:
        """
        :param reference: A SampleStoreReference instance
        """

        if reference.parameters is None:
            raise ValueError("CSVSampleStore.experimentCatalog requires parameters")

        return reference.parameters.catalog

    def experimentCatalog(self):

        return self.sourceDescription.catalog

    @classmethod
    def from_csv(
        cls,
        csvPath: str,
        idColumn: str,
        generatorIdentifier: str | None = None,
        experimentIdentifier: str | None = None,
        observedPropertyColumns: list[str] | None = None,
        constitutivePropertyColumns: list[str] | None = None,
    ):

        # Create a schema of the contents of the CSV file
        # This is used to convert its contents to entities
        experiments = []
        if observedPropertyColumns:
            experimentDescriptor = ExperimentDescription(
                experimentIdentifier=experimentIdentifier,
                propertyMap={k: k for k in observedPropertyColumns},
            )

            experiments.append(experimentDescriptor)

        if not constitutivePropertyColumns:
            # check the file
            import pandas as pd

            headers = pd.read_csv(csvPath, nrows=0).columns.tolist()
            constitutivePropertyColumns = [
                h for h in headers if h not in [*observedPropertyColumns, idColumn]
            ]

        csvDescription = CSVSampleStoreDescription(
            identifierColumn=idColumn,
            generatorIdentifier=generatorIdentifier,
            experiments=experiments,
            constitutivePropertyColumns=constitutivePropertyColumns,
        )

        return CSVSampleStore(
            storageLocation=orchestrator.utilities.location.FilePathLocation(
                path=csvPath
            ),
            parameters=csvDescription,
        )

    def __init__(
        self,
        storageLocation: orchestrator.utilities.location.FilePathLocation,
        parameters: CSVSampleStoreDescription,
    ):
        """

        :param parameters: A dictionary that describes how parse the CSV file. It contains the following keys
            - "identifierColumn" the column containing the entity identifier
            - "experiments" An list of dicts. Each describing an "Experiment" defined by the CSV File contents.
                    Each dict has two keys - experimentIdentifier and propertyMap
            - "generatorIdentifier" an identifier for the source of the entities

        If generatorIdentifier is not in parameters then the value of storage_location.file_hash is used
        """

        self.log = logging.getLogger("CSVSampleStore")
        self.sourceDescription = parameters
        self.storageLocation = storageLocation

        if self.sourceDescription.generatorIdentifier is None:
            self.sourceDescription.generatorIdentifier = (
                self.storageLocation.hash_identifier
            )

        import pandas as pd

        self._data = pd.read_csv(self.storageLocation.path)
        # Make column headers lowercase
        self._data.columns = self._data.columns.str.lower().str.strip()
        self._observedProperties = self.sourceDescription.observedProperties

        # TODO: necessary to merge entities...
        self._entities = []
        self._ent_by_id: dict[str, Entity] = {}
        # TODO: improve
        for _i, row in self._data.T.items():  # noqa: PERF102
            entity_id = row[self.sourceDescription.identifierColumn]
            try:
                # Check if entity already exists
                self._ent_by_id[entity_id]
            except KeyError:
                # No - Create a new entity
                try:
                    ne = self._entity_from_csv_entry(row)
                except pydantic.ValidationError as error:
                    self.log.debug(f"Error processing row {row}. {error}")
                else:
                    self._entities.append(ne)
                    self._ent_by_id[ne.identifier] = ne
            else:
                # Yes - Add the additional observed properties to existing entity
                observed_properties, _ = self._observed_property_values_from_row(row)
                experiments_in_properties = {
                    op.property.experimentReference for op in observed_properties
                }
                for experiment in experiments_in_properties:
                    property_values_for_experiment = [
                        op
                        for op in observed_properties
                        if op.property.experimentReference == experiment
                    ]
                    self._ent_by_id[entity_id].add_measurement_result(
                        ValidMeasurementResult(
                            entityIdentifier=entity_id,
                            measurements=property_values_for_experiment,
                        )
                    )

        self._entity_ids = [e.identifier for e in self.entities]

    @property
    def config(self) -> CSVSampleStoreDescription:

        return self.sourceDescription.model_copy()

    @property
    def location(self) -> orchestrator.utilities.location.ResourceLocation:

        return self.storageLocation.model_copy()

    def _observed_property_values_from_row(self, row: "pd.Series") -> tuple[list, list]:

        observedCalcValue = []
        experimentDescriptionMap = self.sourceDescription.experimentDescriptionMap

        columns_already_processed = [self.sourceDescription.identifierColumn]
        # Add values for experiments which are defined in this sample store
        for op in self._observedProperties:
            # See if the experiment that provides this property is defined by this CSV file
            expDescription = experimentDescriptionMap.get(
                op.experimentReference.experimentIdentifier
            )
            if expDescription is not None:
                columnHeader = expDescription.propertyMap[
                    op.targetProperty.identifier
                ].lower()
                opv = ObservedPropertyValue(property=op, value=row[columnHeader])
                observedCalcValue.append(opv)
                columns_already_processed.append(columnHeader)
            else:
                self.log.debug(
                    f"Experiment that provides {op}, {op.experimentReference.experimentIdentifier}, is not provided by this CSV file"
                )

        return observedCalcValue, columns_already_processed

    def _entity_from_csv_entry(self, row: "pd.Series") -> Entity:
        """Creates an entity from pandas Series

        :param row: A Series

        Raises:
            Raise a KeyError if there is no entry in row related to a property in propertyNames
        """

        entity_id = row[self.sourceDescription.identifierColumn]
        observedCalcValue, _ = self._observed_property_values_from_row(row)

        constitutive_property_values = []
        for cp in self.sourceDescription.constitutiveProperties:
            value = row[cp.identifier]
            # PropertyValue will handle converting value to the most appropriate type from string
            constitutive_property_values.append(
                ConstitutivePropertyValue(property=cp, value=value)
            )

        try:
            entity = Entity(
                identifier=entity_id,
                generatorid=self.sourceDescription.generatorIdentifier,
                constitutive_property_values=tuple(constitutive_property_values),
            )
        except pydantic.ValidationError as error:
            self.log.warning(f"Unable to create entity from row {row}: {error}")
            raise

        experiments_in_properties = {
            op.property.experimentReference for op in observedCalcValue
        }
        for experiment in experiments_in_properties:
            property_values_for_experiment = [
                op
                for op in observedCalcValue
                if op.property.experimentReference == experiment
            ]
            entity.add_measurement_result(
                ValidMeasurementResult(
                    entityIdentifier=entity_id,
                    measurements=property_values_for_experiment,
                )
            )
        return entity

    @property
    def csvDescription(self) -> CSVSampleStoreDescription:

        return self.sourceDescription

    @property
    def entities(self):

        return self._entities

    @property
    def numberOfEntities(self):

        return len(self._entities)

    def containsEntityWithIdentifier(self, entity_id):

        return entity_id in self._entity_ids

    @property
    def identifier(self):

        # hash file
        import hashlib

        # hash experiment/properties
        h = (
            hashlib.md5()
        )  # Construct a hash object using our selected hashing algorithm
        for op in self._observedProperties:
            h.update(
                op.identifier.encode("utf-8")
            )  # Update the hash using a bytes object

        generator_id = self.sourceDescription.generatorIdentifier
        return f"{generator_id}-{h.hexdigest()}"
