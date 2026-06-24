# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Annotated

import pydantic

if TYPE_CHECKING:
    from orchestrator.core.samplestore.sql import SQLSampleStore


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


def samplestore_statistics_for_stores(
    stores: "SQLSampleStore | list[SQLSampleStore]",
) -> "dict[str, SampleStoreStatistics]":
    """Compute statistics for one or more sample stores.

    Args:
        stores: A single :class:`~orchestrator.core.samplestore.sql.SQLSampleStore`
            instance or a list of them.  An empty list returns ``{}`` without
            any database access.

    Returns:
        A ``dict`` mapping each store's :attr:`~orchestrator.core.samplestore.base.ActiveSampleStore.uri`
        to its :class:`SampleStoreStatistics`.
    """
    store_list = stores if isinstance(stores, list) else [stores]

    if not store_list:
        return {}

    return {store.uri: store.samplestore_statistics() for store in store_list}
