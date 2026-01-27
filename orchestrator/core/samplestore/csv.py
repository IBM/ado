# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
from typing import TYPE_CHECKING, Annotated, Literal

import pydantic

if TYPE_CHECKING:
    import pandas as pd

import orchestrator.core.samplestore.config
import orchestrator.utilities.location
from orchestrator.core.samplestore.base import (
    PassiveSampleStore,
    SampleStoreDescription,
)
from orchestrator.modules.actuators.catalog import ExperimentCatalog
from orchestrator.schema.entity import Entity
from orchestrator.schema.observed_property import ObservedPropertyValue
from orchestrator.schema.property_value import ConstitutivePropertyValue
from orchestrator.schema.result import ValidMeasurementResult


def warn_deprecated_csv_sample_store_model_in_use(
    deprecated_from_version: str = "1.3.5",
    removed_from_version: str = "1.6.0",
    deprecated_fields: str | list[str] | None = None,
    latest_format_documentation_url: str | None = None,
) -> None:
    """Warn that a deprecated CSV sample store model format is being auto-upgraded

    Similar to warn_deprecated_actuator_parameters_model_in_use but for sample stores.
    """
    from rich.console import Console

    resource_name = "samplestore"
    doc_url = (
        f": {latest_format_documentation_url}"
        if latest_format_documentation_url
        else ""
    )

    if deprecated_fields:
        fields_causing_issues = f"fields [b magenta]{deprecated_fields}[/b magenta]"
        if isinstance(deprecated_fields, str):
            fields_causing_issues = f"field [b magenta]{deprecated_fields}[/b magenta]"
        elif isinstance(deprecated_fields, list) and len(deprecated_fields) == 1:
            fields_causing_issues = (
                f"field [b magenta]{deprecated_fields[0]}[/b magenta]"
            )

        warning_preamble = (
            f"The use of {fields_causing_issues} in the {resource_name} configuration "
            f"is deprecated as of ADO [b cyan]{deprecated_from_version}[/b cyan]."
        )
    else:
        warning_preamble = (
            f"The {resource_name} configuration format has been updated "
            f"as of ADO [b cyan]{deprecated_from_version}[/b cyan]."
        )

    autoupgrade_notice = "It is being temporarily auto-upgraded to the latest version."
    autoupgrade_removal_warning = (
        f"[b]This behavior will be removed with ADO "
        f"[b cyan]{removed_from_version}[/b cyan][/b]."
    )
    manual_upgrade_hint = (
        f"Run [b cyan]ado upgrade {resource_name}s[/b cyan] to upgrade the stored {resource_name}s.\n\t"
        f"Update your {resource_name} YAML files to use the latest format{doc_url}."
    )

    Console(stderr=True).print(
        f"[b yellow]WARN[/b yellow]:\t{warning_preamble}\n\t"
        f"{autoupgrade_notice}\n\t{autoupgrade_removal_warning}\n"
        f"[b magenta]HINT[/b magenta]:\t{manual_upgrade_hint}",
        overflow="ignore",
        crop=False,
    )


class CSVSampleStoreDescription(SampleStoreDescription):
    identifierColumn: Annotated[
        str,
        pydantic.Field(
            description="The header of the column that contains the entity ids"
        ),
    ]
    generatorIdentifier: Annotated[
        str | None,
        pydantic.Field(
            validate_default=True,
            description="The id of the entity generator",
        ),
    ] = None

    model_config = pydantic.ConfigDict(extra="forbid")

    @pydantic.model_validator(mode="before")
    @classmethod
    def migrate_old_format(cls, data: dict) -> dict:
        """Migrate old CSVSampleStoreDescription format to new format

        Old format (the only one stored before these changes):
        - constitutivePropertyColumns at top level (list)
        - experiments list with propertyMap (not observedPropertyMap)
        - No constitutivePropertyMap in experiment descriptions

        Migration:
        - Remove constitutivePropertyColumns from top level
        - For each experiment desc dict, add constitutivePropertyMap = constitutivePropertyColumns
        - For each experiment desc dict, rename propertyMap to observedPropertyMap
        """

        if not isinstance(data, dict):
            return data

        # Check if this is old format (has constitutivePropertyColumns at top level)
        if "constitutivePropertyColumns" not in data:
            return data

        # Old format detected - emit warning and perform migration
        warn_deprecated_csv_sample_store_model_in_use(
            deprecated_from_version="1.3.5",
            removed_from_version="1.6.0",
            deprecated_fields=["constitutivePropertyColumns", "propertyMap"],
        )

        constitutive_columns = data.pop("constitutivePropertyColumns")

        # Migrate experiments if present
        if "experiments" in data and isinstance(data["experiments"], list):
            for exp in data["experiments"]:
                if isinstance(exp, dict):
                    # Rename propertyMap to observedPropertyMap
                    if "propertyMap" in exp:
                        exp["observedPropertyMap"] = exp.pop("propertyMap")
                    # Add constitutivePropertyMap from top-level constitutivePropertyColumns
                    exp["constitutivePropertyMap"] = constitutive_columns

        return data


class CSVSampleStore(PassiveSampleStore):
    """Reads entities and properties from a CSV file

    Entities are assumed to be in rows, properties are column headers

    """

    @staticmethod
    def validate_parameters(parameters: dict) -> CSVSampleStoreDescription:
        return CSVSampleStoreDescription.model_validate(parameters)

    @staticmethod
    def storage_location_class() -> (
        type[orchestrator.utilities.location.FilePathLocation]
    ):
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

    def experimentCatalog(self) -> ExperimentCatalog:
        return self.sourceDescription.catalog

    @classmethod
    def from_csv(
        cls,
        csvPath: str,
        idColumn: str,
        generatorIdentifier: str | None = None,
        experimentIdentifier: str | None = None,
        actuatorIdentifier: str = "replay",
        observedPropertyColumns: list[str] | None = None,
        constitutivePropertyColumns: list[str] | None = None,
        propertyFormat: Literal["target", "observed"] = "target",
    ) -> "CSVSampleStore":
        """Create a CSVSampleStore from a CSV file

        Args:
            csvPath: Path to the CSV file
            idColumn: Column containing entity identifiers
            generatorIdentifier: Optional identifier for the entity generator
            experimentIdentifier: Experiment identifier
            actuatorIdentifier: Actuator identifier (defaults to 'replay' if not provided)
            observedPropertyColumns: List of columns containing observed properties
            constitutivePropertyColumns: List of columns containing constitutive properties
            propertyFormat: The naming format used for the observed property columns.
                Only relevant when actuatorIdentifier is not 'replay'
        """
        import pandas as pd

        from orchestrator.core.samplestore.base import (
            ExternalExperimentDescription,
            InternalExperimentDescription,
        )

        if actuatorIdentifier != "replay":
            experimentDescriptor = InternalExperimentDescription(
                actuatorIdentifier=actuatorIdentifier,
                experimentIdentifier=experimentIdentifier,
                observedPropertyMap=observedPropertyColumns,
                constitutivePropertyMap=constitutivePropertyColumns,
                propertyFormat=propertyFormat,
            )
        else:

            # Read CSV to get headers
            headers = pd.read_csv(csvPath, nrows=0).columns.tolist()
            headers = [h.strip() for h in headers]

            # Determine constitutive property columns if not provided
            if not constitutivePropertyColumns:
                # Exclude id column and observed property columns
                excluded = [idColumn]
                if observedPropertyColumns:
                    excluded.extend(observedPropertyColumns)

                constitutivePropertyColumns = [h for h in headers if h not in excluded]

            experimentDescriptor = ExternalExperimentDescription(
                experimentIdentifier=experimentIdentifier,
                observedPropertyMap=observedPropertyColumns,
                constitutivePropertyMap=constitutivePropertyColumns,
            )

        csvDescription = CSVSampleStoreDescription(
            identifierColumn=idColumn,
            generatorIdentifier=generatorIdentifier,
            experiments=[experimentDescriptor],
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
    ) -> None:
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
        # Strip whitespace from column headers
        self._data.columns = self._data.columns.str.strip()

        # Validate that all required columns exist in the CSV
        self._validate_required_columns()

        # TODO: necessary to merge entities...
        self._entities: list[Entity] = []
        self._ent_by_id: dict[str, Entity] = {}
        # TODO: improve
        for _i, row in self._data.T.items():  # noqa: PERF102
            entity_id = row[self.sourceDescription.identifierColumn]
            entity = None
            try:
                # Check if entity already exists
                entity = self._ent_by_id[entity_id]
            except KeyError:
                # No - Create a new entity
                try:
                    entity = self._entity_from_csv_entry(row)
                except pydantic.ValidationError as error:
                    self.log.debug(f"Error processing row {row}. {error}")
                else:
                    self._entities.append(entity)
                    self._ent_by_id[entity.identifier] = entity
            finally:
                # Add measurement results if entity exists (either new or existing)
                if entity is not None:
                    measurement_results = self._measurement_results_from_row(
                        row, entity_id
                    )
                    for result in measurement_results:
                        entity.add_measurement_result(result)

        self._entity_ids = [e.identifier for e in self.entities]

    @property
    def config(self) -> CSVSampleStoreDescription:
        return self.sourceDescription.model_copy()

    @property
    def location(self) -> orchestrator.utilities.location.ResourceLocation:
        return self.storageLocation.model_copy()

    def _validate_required_columns(self) -> None:
        """Validates that all required columns exist in the CSV file"""
        # Collect all required columns
        required_columns = set()

        # Identifier column is required
        required_columns.add(self.sourceDescription.identifierColumn)

        # Constitutive property columns are required
        for col in self.sourceDescription.source_constitutive_property_identifiers:
            required_columns.add(col)

        # Observed property columns are required
        for col in self.sourceDescription.source_observed_property_identifiers:
            required_columns.add(col)

        # Check which required columns are missing
        available_columns = set(self._data.columns)
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                f"CSV file '{self.storageLocation.path}' is missing required columns: "
                f"{sorted(missing_columns)}. Available columns: {sorted(available_columns)}"
            )

    def _measurement_results_from_row(
        self, row: "pd.Series", entity_id: str
    ) -> list[ValidMeasurementResult]:
        """Creates measurement results from a row of the CSV

        Args:
            row: A pandas Series representing a row of the CSV
            entity_id: The entity identifier for this row

        Returns:
            List of ValidMeasurementResult instances
        """
        measurement_results = []

        # Iterate over experiment descriptions directly
        for exp_desc in self.sourceDescription.experiments:
            observed_property_values = []
            experiment = exp_desc.experiment

            # Create a set of target property IDs from the observed property map
            target_prop_ids = set(exp_desc.observedPropertyMap.keys())

            # Filter observed properties to only those in the map
            filtered_obs_props = [
                obs_prop
                for obs_prop in experiment.observedProperties
                if obs_prop.targetProperty.identifier in target_prop_ids
            ]

            # Iterate filtered observed properties and get column names from the map
            for obs_prop in filtered_obs_props:
                column_name = exp_desc.observedPropertyMap[
                    obs_prop.targetProperty.identifier
                ]

                # Skip if the column is not in the row
                if column_name not in row.index:
                    continue

                opv = ObservedPropertyValue(property=obs_prop, value=row[column_name])
                observed_property_values.append(opv)

            # Create measurement result if we have any observed properties
            if observed_property_values:
                measurement_results.append(
                    ValidMeasurementResult(
                        entityIdentifier=entity_id,
                        measurements=observed_property_values,
                    )
                )

        return measurement_results

    def _entity_from_csv_entry(self, row: "pd.Series") -> Entity:
        """Creates an entity from pandas Series (constitutive properties only)

        :param row: A Series

        Raises:
            Raise a KeyError if there is no entry in row related to a property in propertyNames
        """

        entity_id = row[self.sourceDescription.identifierColumn]

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

        return entity

    @property
    def csvDescription(self) -> CSVSampleStoreDescription:
        return self.sourceDescription

    @property
    def entities(self) -> list[Entity]:
        return self._entities

    @property
    def numberOfEntities(self) -> int:
        return len(self._entities)

    def containsEntityWithIdentifier(self, entity_id: str) -> bool:
        return entity_id in self._entity_ids

    @property
    def identifier(self) -> str:
        # hash file
        import hashlib

        # hash experiment/properties
        h = hashlib.md5(
            usedforsecurity=False
        )  # Construct a hash object using our selected hashing algorithm
        for op in self.sourceDescription.observedProperties:
            h.update(
                op.identifier.encode("utf-8")
            )  # Update the hash using a bytes object

        generator_id = self.sourceDescription.generatorIdentifier
        return f"{generator_id}-{h.hexdigest()}"
