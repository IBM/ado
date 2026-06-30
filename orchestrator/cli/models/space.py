# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from io import StringIO

import rich.table

from orchestrator.core.discoveryspace.space import DiscoverySpace


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
        stats = space.space_statistics(lightweight_only=False)

        return cls(
            entities_sampled_from_space_with_all_measurements_applied=stats.entities_with_all_measurements,
            entities_sampled_from_space_with_partial_measurements_applied=stats.entities_with_partial_measurements,
            entities_yet_to_be_sampled_and_measured_from_space=stats.number_unmeasured_entities,
            entities_matching_the_space=stats.number_matching_entities,
            matching_entities_in_sample_store_with_measurement_space_applied=stats.matching_entities_with_all_measurements,
            size_of_entity_space=stats.size_of_entity_space,
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
