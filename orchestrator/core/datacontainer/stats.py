# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import Annotated

import pydantic


class DataContainerStatistics(pydantic.BaseModel):
    """Aggregated statistics for a single data container.

    Attributes:
        number_of_tables: Number of named tabular-data objects stored in the
            container (``config.tabularData`` entries).
        number_of_locations: Number of named location references stored in the
            container (``config.locationData`` entries).
        number_of_key_values: Number of named key-value entries stored in the
            container (``config.data`` entries).
        total_data_bytes: Approximate serialised size of ``tabularData``,
            ``locationData``, and ``data``, excluding the ``metadata`` section.
    """

    number_of_tables: Annotated[
        int,
        pydantic.Field(
            description=(
                "Number of named tabular-data objects stored in the container "
                "(config.tabularData entries)."
            )
        ),
    ]
    number_of_locations: Annotated[
        int,
        pydantic.Field(
            description=(
                "Number of named location references stored in the container "
                "(config.locationData entries)."
            )
        ),
    ]
    number_of_key_values: Annotated[
        int,
        pydantic.Field(
            description=(
                "Number of named key-value entries stored in the container "
                "(config.data entries)."
            )
        ),
    ]
    total_data_bytes: Annotated[
        int,
        pydantic.Field(
            description=(
                "Approximate serialised size of the three data fields "
                "(tabularData, locationData, data), excluding the metadata section."
            )
        ),
    ]
