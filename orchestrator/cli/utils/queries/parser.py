# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json


def prepare_query_filters_for_db(query_filters: dict | list[dict]) -> list[dict]:
    """
    Converts query filters to the JSON representation expected by MySQL.

    Parameters:
    query_filters (Union[dict, list[dict]]): Query filters to be prepared.

    Returns:
    list[dict]: Prepared query filters.
    """

    if not isinstance(query_filters, list):
        query_filters = [query_filters]

    result = []
    for current_filter in query_filters:
        for k, v in current_filter.items():
            try:
                v = json.dumps(json.loads(str(v)))
            except json.JSONDecodeError:
                v = json.dumps(v)
            result.append({f"$.{k}": v})

    return result
