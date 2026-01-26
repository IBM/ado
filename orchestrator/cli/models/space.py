# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import math
import typing
from io import StringIO

import rich.table

import orchestrator.schema.property
from orchestrator.cli.utils.output.prints import console_print
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.schema.entityspace import EntitySpaceRepresentation

if typing.TYPE_CHECKING:
    import pandas as pd

    from orchestrator.core.discoveryspace.resource import DiscoverySpaceResource
    from orchestrator.metastore.project import ProjectContext


class SpaceDetails:

    def __init__(
        self,
        entities_sampled_from_space_with_all_measurements_applied: int,
        entities_sampled_from_space_with_partial_measurements_applied: int,
        entities_yet_to_be_sampled_and_measured_from_space: int,
        entities_matching_the_space: int,
        matching_entities_in_sample_store_with_measurement_space_applied: int,
        size_of_entity_space: int,
    ) -> None:
        # Entities sampled from space with all measurements applied
        self.entities_sampled_from_space_with_all_measurements_applied = (
            entities_sampled_from_space_with_all_measurements_applied
        )
        # Entities sampled from space with partial measurements applied
        self.entities_sampled_from_space_with_partial_measurements_applied = (
            entities_sampled_from_space_with_partial_measurements_applied
        )
        # Entities yet to be sampled and measured from space
        self.entities_yet_to_be_sampled_and_measured_from_space = (
            entities_yet_to_be_sampled_and_measured_from_space
        )
        # Entities matching the space
        self.entities_matching_the_space = entities_matching_the_space
        # Matching entities in the sample store with measurement space applied
        self.matching_entities_in_sample_store_with_measurement_space_applied = (
            matching_entities_in_sample_store_with_measurement_space_applied
        )
        # Size of the entity space
        self.size_of_entity_space = size_of_entity_space

    @classmethod
    def from_space(cls, space: DiscoverySpace) -> "SpaceDetails":

        import pandas as pd

        # Entities sampled in the space
        measured_entities_table = space.measuredEntitiesTable(property_type="target")
        if (
            measured_entities_table.empty
            or "identifier" not in measured_entities_table.columns
        ):
            measured_entities_table["identifier"] = pd.Series(dtype="str")

        # Entities sampled from space with all measurements applied
        experiments_in_measurement_space = len(space.measurementSpace.experiments)
        entities_sampled_from_space_with_all_measurements_applied = 0
        for _, group in measured_entities_table.groupby("identifier"):
            if group.shape[0] == experiments_in_measurement_space:
                entities_sampled_from_space_with_all_measurements_applied += 1

        # Entities sampled from space with partial measurements applied
        entities_sampled_from_space_with_partial_measurements_applied = (
            len(measured_entities_table["identifier"].unique())
            - entities_sampled_from_space_with_all_measurements_applied
        )

        # Unmeasured entities
        if space.entitySpace is None or not isinstance(
            space.entitySpace,
            EntitySpaceRepresentation,
        ):
            entities_yet_to_be_sampled_and_measured_from_space = math.nan
        elif not space.entitySpace.isDiscreteSpace:
            entities_yet_to_be_sampled_and_measured_from_space = math.inf
        else:
            entities_yet_to_be_sampled_and_measured_from_space = (
                space.entitySpace.size
                - entities_sampled_from_space_with_partial_measurements_applied
                - entities_sampled_from_space_with_all_measurements_applied
            )

        # Entities matching the space
        entities_matching_the_space = len(space.matchingEntities())

        # Matching entities in the sample store with measurement space applied
        matching_entities_table = space.matchingEntitiesTable(property_type="target")
        if (
            matching_entities_table.empty
            or "identifier" not in matching_entities_table.columns
        ):
            matching_entities_table["identifier"] = pd.Series(dtype="str")

        matching_entities_in_sample_store_with_measurement_space_applied = 0
        for _, group in matching_entities_table.groupby("identifier"):
            if group.shape[0] == experiments_in_measurement_space:
                matching_entities_in_sample_store_with_measurement_space_applied += 1

        # Size of the entity space
        size_of_entity_space = None
        if (
            isinstance(
                space.entitySpace,
                EntitySpaceRepresentation,
            )
            and space.entitySpace.isDiscreteSpace
        ):
            size_of_entity_space = space.entitySpace.size

        return cls(
            entities_sampled_from_space_with_all_measurements_applied=entities_sampled_from_space_with_all_measurements_applied,
            entities_sampled_from_space_with_partial_measurements_applied=entities_sampled_from_space_with_partial_measurements_applied,
            entities_yet_to_be_sampled_and_measured_from_space=entities_yet_to_be_sampled_and_measured_from_space,
            entities_matching_the_space=entities_matching_the_space,
            matching_entities_in_sample_store_with_measurement_space_applied=matching_entities_in_sample_store_with_measurement_space_applied,
            size_of_entity_space=size_of_entity_space,
        )

    def to_rich_table(self) -> rich.table.Table:
        table = rich.table.Table("", header_style=None, box=None)

        # Size of the entity space
        if self.size_of_entity_space:
            table.add_row("Size of the entity space", str(self.size_of_entity_space))

        # Sampled entities with measurement space applied
        table.add_row(
            "Entities sampled from space with all measurements applied",
            str(self.entities_sampled_from_space_with_all_measurements_applied),
        )

        # Entities sampled from space with partial measurements applied
        table.add_row(
            "Entities sampled from space with partial measurements applied",
            str(self.entities_sampled_from_space_with_partial_measurements_applied),
        )

        # Entities yet to be sampled and measured from space
        table.add_row(
            "Entities yet to be sampled and measured from space",
            str(self.entities_yet_to_be_sampled_and_measured_from_space),
        )

        # Entities matching the space
        table.add_row(
            "Entities matching the space in the sample store",
            str(self.entities_matching_the_space),
        )

        # Matching entities in the sample store with measurement space applied
        table.add_row(
            "Matching entities in the sample store with measurement space applied",
            str(self.matching_entities_in_sample_store_with_measurement_space_applied),
        )

        return table

    def to_markdown(self) -> str:
        content = StringIO()
        if self.size_of_entity_space:
            content.write(f"- Size of the entity space: {self.size_of_entity_space}\n")

        content.write(
            f"- Entities sampled from space with all measurements applied: {self.entities_sampled_from_space_with_all_measurements_applied}\n"
        )

        content.write(
            f"- Entities sampled from space with partial measurements applied: {self.entities_sampled_from_space_with_partial_measurements_applied}\n"
        )

        content.write(
            f"- Entities yet to be sampled and measured from space: {self.entities_yet_to_be_sampled_and_measured_from_space}\n"
        )

        content.write(
            f"- Entities matching the space in the sample store: {self.entities_matching_the_space}\n"
        )
        content.write(
            f"- Matching entities in the sample store with measurement space applied: {self.matching_entities_in_sample_store_with_measurement_space_applied}\n"
        )

        return content.getvalue()


class SpaceSummary:

    def __init__(
        self,
        space_id: str,
        project_context: "ProjectContext",
    ) -> None:
        from orchestrator.metastore.sqlstore import SQLStore
        from orchestrator.schema.property import NonMeasuredPropertyTypeEnum

        sql = SQLStore(project_context=project_context)

        space_resource: DiscoverySpaceResource = sql.getResource(
            identifier=space_id,
            kind=CoreResourceKinds.DISCOVERYSPACE,
            raise_error_if_no_resource=True,
        )
        if (
            space_resource.kind
            != orchestrator.core.resources.CoreResourceKinds.DISCOVERYSPACE
        ):
            raise ValueError(
                f"The resource id {space_id} belongs to a {space_resource.kind.value} and not to a space."
            )

        space = DiscoverySpace.from_configuration(
            conf=space_resource.config,
            project_context=project_context,
            identifier=space_id,
        )

        if not space.measurementSpace.experiments:
            raise ValueError(f"There are no experiments defined in space {space_id}")

        related_operations = sql.getRelatedResourceIdentifiers(
            identifier=space_id,
            kind=orchestrator.core.resources.CoreResourceKinds.OPERATION.value,
        )

        constitutive_properties = {
            p.identifier: (
                p.propertyDomain.values
                if p.propertyDomain.values
                else p.propertyDomain.domainRange
            )
            for p in space_resource.config.entitySpace
            if p.propertyType == NonMeasuredPropertyTypeEnum.CONSTITUTIVE_PROPERTY_TYPE
        }

        self.id = space.resource.identifier
        self.creation_date = space_resource.created
        self.name = space_resource.config.metadata.name
        self.description = space_resource.config.metadata.description
        self.labels = space_resource.config.metadata.labels
        self.experiments = space.measurementSpace.experiments
        self.details = SpaceDetails.from_space(space=space)
        self.related_operations = related_operations
        self.constitutive_properties = constitutive_properties

    @property
    def sampled_and_measured_percentage(self) -> float:
        if (
            not self.details.size_of_entity_space
            or self.details.size_of_entity_space == 0
        ):
            return math.nan
        return (
            self.details.entities_sampled_from_space_with_all_measurements_applied
            / self.details.size_of_entity_space
        ) * 100

    @property
    def matching_and_measured_percentage(self) -> float:
        if (
            not self.details.size_of_entity_space
            or self.details.size_of_entity_space == 0
        ):
            return math.nan
        return (
            self.details.matching_entities_in_sample_store_with_measurement_space_applied
            / self.details.size_of_entity_space
        ) * 100

    @staticmethod
    def _get_dataframe_columns(
        candidate_columns: list[str],
        columns_to_hide: list[str] | None = None,
    ) -> list[str]:
        common_column_mappings = {
            "id": "Space ID",
            "experiment": "Experiments",
            "matching": "Matching and Measured",
            "sampled": "Sampled and Measured",
            "name": "Name",
            "description": "Description",
        }

        if not columns_to_hide:
            columns_to_hide = []

        columns_to_hide = {
            (
                common_column_mappings[col].lower()
                if col in common_column_mappings
                else col.lower()
            )
            for col in columns_to_hide
        }
        return [col for col in candidate_columns if col.lower() not in columns_to_hide]

    def to_markdown_text(self, heading_level: int = 1) -> str:
        content = StringIO()
        content.write(f"{'#'*heading_level} Space `{self.id}`\n")
        heading_level += 1

        content.write(f"{'#'*heading_level} General information\n\n")

        if self.name:
            content.write(f"- **Name**: {self.name}\n")

        if self.description:
            content.write(f"- **Description**: {self.description}\n")

        content.write(f"- **Created on**: {self.creation_date.strftime('%c')}\n")

        if self.labels:
            content.write("- **Labels**:\n")
            for k, v in self.labels.items():
                content.write(f"  - {k}: {v}\n")

        if self.constitutive_properties:
            content.write(f"\n{'#' * heading_level} Constitutive properties:\n\n")
            for k, v in self.constitutive_properties.items():
                content.write(f"- `{k}`")
                if v:
                    content.write(f": {v}\n")
                else:
                    content.write("\n")

        if self.experiments:
            content.write(f"\n{'#'*heading_level} Experiments in this space\n\n")
            for e in self.experiments:
                content.write(f"- `{e}`\n")

        content.write(f"\n{'#'*heading_level} Exploration status\n\n")
        if not math.isnan(self.sampled_and_measured_percentage):
            content.write(
                f"**Sampled and Measured percentage**: {math.floor(self.sampled_and_measured_percentage)}%\n"
            )

        if not math.isnan(self.matching_and_measured_percentage):
            content.write(
                f"**Matching and Measured percentage**: {math.floor(self.matching_and_measured_percentage)}%\n"
            )
        content.write("\n" + self.details.to_markdown() + "\n")

        if len(self.related_operations["IDENTIFIER"]) > 0:
            content.write(f"\n{'#'*heading_level} Related operations\n\n")
            for o in self.related_operations["IDENTIFIER"]:
                content.write(f"- `{o}`\n")

        return content.getvalue()

    def to_dataframe(
        self,
        include_properties: list[str] | None = None,
        columns_to_hide: list[str] | None = None,
    ) -> "pd.DataFrame":

        import pandas as pd

        if not include_properties:
            include_properties = []

        if not columns_to_hide:
            columns_to_hide = []

        item = {
            "Space ID": self.id,
            "Experiments": (
                [e.identifier for e in self.experiments]
                if len(self.experiments) > 1
                else self.experiments[0]
            ),
            "Sampled and Measured": (
                "N/A"
                if math.isnan(self.sampled_and_measured_percentage)
                else f"{math.floor(self.sampled_and_measured_percentage)}%"
            ),
            "Matching and Measured": (
                "N/A"
                if math.isnan(self.matching_and_measured_percentage)
                else f"{math.floor(self.matching_and_measured_percentage)}%"
            ),
            "Name": self.name if self.name else "",
            "Description": self.description if self.description else "",
        }

        if self.labels:
            item.update(self.labels)

        for p in include_properties:
            if p not in self.constitutive_properties:
                console_print(f"{p} not in properties", stderr=True)
                continue

            if not self.constitutive_properties[p]:
                item[p] = "N/A"
            elif len(self.constitutive_properties[p]) == 1:
                item[p] = self.constitutive_properties[p][0]
            else:
                item[p] = self.constitutive_properties[p]

        df = pd.DataFrame([item])

        # Reorder columns for sanity
        base_columns = [
            "Space ID",
            "Experiments",
            "Matching and Measured",
            "Sampled and Measured",
            "Name",
            "Description",
        ]
        labels = sorted(
            [
                column
                for column in df.columns
                if column not in base_columns and column not in include_properties
            ]
        )
        return df.reindex(
            columns=self._get_dataframe_columns(
                candidate_columns=base_columns + include_properties + labels,
                columns_to_hide=columns_to_hide,
            )
        )
