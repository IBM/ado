# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import Annotated

import pydantic


class SampleStoreStatistics(pydantic.BaseModel):
    """Aggregated statistics for a single sample store.

    Attributes:
        number_of_entities: Total number of entities in the store.
        number_of_results: Total number of measurement results in the store.
        number_of_experiments: Number of distinct experiments that have been run
            in the store (distinct ``experiment_reference`` values across all
            measurement requests).
    """

    number_of_entities: Annotated[
        int,
        pydantic.Field(description="Total number of entities in the store."),
    ]
    number_of_results: Annotated[
        int,
        pydantic.Field(description="Total number of measurement results in the store."),
    ]
    number_of_experiments: Annotated[
        int,
        pydantic.Field(
            description=(
                "Number of distinct experiments that have been run in the store."
            )
        ),
    ]
