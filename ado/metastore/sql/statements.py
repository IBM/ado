# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
from types import NoneType


def _quote_sql_identifier(identifier: str) -> str:
    """
    Quote a SQL identifier to prevent SQL injection.

    Uses double quotes and escapes any double quotes in the identifier by doubling them,
    which is the standard SQL way to escape quotes in identifiers.

    Args:
        identifier: The identifier to quote

    Returns:
        The quoted identifier safe for use in SQL
    """
    # Escape any double quotes by doubling them, then wrap in double quotes
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def simulate_json_contains_on_sqlite(
    path: str,
    candidate: str,
    table_name: str = "resources",
    json_column: str = "data",
    id_column: str = "identifier",
) -> str:
    """
    Simulate MySQL's JSON_CONTAINS on SQLite.

    On MySQL, JSON_CONTAINS allows searching for a JSON document within a JSON field.
    It matches all documents that contains at least the provided JSON document.

    In our simulated version, we prepare a subquery that can be used in a WHERE statement
    that filters rows making sure their ID is one that has all the fields
    from the candidate document. ``null`` candidates need separate handling because
    ``json_tree`` only emits rows for fields that exist in the document, so missing
    fields would otherwise produce no row and never match.

    Args:
        path (str): The path to the JSON field to check.
        candidate (str): The JSON document to check.
        table_name (str): Name of the table to query (default: "resources").
        json_column (str): Name of the JSON column to search (default: "data").
        id_column (str): Name of the ID column to return (default: "identifier").

    Returns:
        str: The SQLite query that checks whether the provided document exists.

    Raises:
        ValueError: If table_name, json_column, or id_column contain invalid characters.
    """
    # Quote SQL identifiers to prevent SQL injection
    quoted_table_name = _quote_sql_identifier(table_name)
    quoted_json_column = _quote_sql_identifier(json_column)
    quoted_id_column = _quote_sql_identifier(id_column)
    parsed_candidate = json.loads(candidate)

    if parsed_candidate is None:
        return f"""
            {quoted_id_column} IN (
                SELECT {quoted_id_column} FROM {quoted_table_name}
                WHERE json_extract({quoted_json_column}, '{path}') IS NULL
            )
            """  # noqa: S608 - identifiers are quoted to prevent injection

    # The subqueries produced by check_field_in_sqlite_json_document need to be
    # INTERSECT-ed to make sure we only retrieve the identifiers that match all
    # the subqueries.
    subqueries = check_field_in_sqlite_json_document(
        parsed_candidate, path, id_column=quoted_id_column
    )

    return (
        """
        {id_column} IN (
            WITH F AS (
                SELECT t.{id_column}, jt.key, jt.value, jt.path
                FROM
                    {table_name} t,
                    json_tree(t.{json_column}, '{path}') jt
            )
            {subqueries}
        )
        """
    ).format(  # noqa: S608 - identifiers are quoted to prevent injection
        path=path,
        table_name=quoted_table_name,
        json_column=quoted_json_column,
        id_column=quoted_id_column,
        subqueries="\n            INTERSECT ".join(subqueries),
    )


def check_field_in_sqlite_json_document(
    candidate: dict | list | str | float,
    path: str,
    id_column: str = "identifier",
) -> list[str]:
    """
    Generate SQLite-compatible SQL fragments to check for the presence of specific fields or values
    within a JSON document using the json_tree virtual table.

    This function recursively traverses the input JSON-like structure (dictionary, list, or scalar)
    and constructs SQL subqueries that can be used to filter rows produced by SQLite's json_tree
    function based on whether the specified fields and values exist at the given JSON path.

    Note: SQLite's json_tree quotes field names containing underscores. This function handles
    that by quoting such field names in the generated patterns. This may produce false positives
    in complex nested structures where the same field name appears at different nesting levels,
    but these are filtered by the INTERSECT logic in simulate_json_contains_on_sqlite.

    Args:
        candidate (dict | list | str | int | float): The JSON structure or scalar value to match against.
            - If a scalar (str, int, float), generates a simple query checking for value presence.
            - If a dict or list, recursively builds queries for nested fields and values.
        path (str): The JSON path (e.g., '$.config.spaces') used to locate the field within the document.
        id_column (str): Name of the ID column to select (default: "identifier").

    Returns:
        list[str]: A list of SQL SELECT statements that can be combined via INTERSECT
        to filter rows whose JSON documents contain the specified structure or values.

    Raises:
        ValueError: If id_column contains invalid characters.
    """
    # Note: id_column is expected to already be quoted by the caller
    # (simulate_json_contains_on_sqlite) to prevent SQL injection
    _ScalarType = str | int | float | bool | None

    def _searchable_scalar_value_for_query_string(value: _ScalarType) -> str:
        if isinstance(value, str):
            return f"= '{value}'"
        if isinstance(value, bool):
            return f"= {json.dumps(value)}"
        if isinstance(value, int | float):
            return f"= {value}"
        if isinstance(value, NoneType):
            return "IS NULL"
        raise ValueError(f"Unexpected type {type(value)}")

    fragments = []
    preamble = f"SELECT {id_column} FROM F WHERE "  # noqa: S608 - id_column is quoted by caller

    # The user has provided a scalar candidate.
    # There are two options:
    #   1. The path points to an object field (a field in a dictionary)
    #   2. The path points to an array value (a field in a list)
    #
    ######################################################
    #
    # An example of the path pointing to an object field is:
    #   ado get operations -q config.operation.parameters.batchSize=1
    #
    # Which translates to
    #   - candidate = batchSize
    #   - path = $.config.operation.parameter
    #
    # When creating the json_tree we will see that:
    #   - The path points to the root of the json_tree
    #   - The key is the path provided
    #   - The value is the candidate
    #
    # | identifier | key | value | path |
    # | ------------------------------------------- | ------------------------------------- | - | - |
    # | randomwalk-1.0.2.dev39+7f0c421.dirty-43dfdf | config.operation.parameters.batchSize | 2 | $ |
    #
    # Handling this case requires us to:
    #   - Strip the $. from the path and use it as a key
    #   - Searching for the value
    #
    # AP: 29/09/2025
    # In some cases it looks like this is not necessarily true.
    # It can also be:
    #
    # | identifier | key | value | path |
    # | ------------------------------------------- | --------- | - | ----------------------------- |
    # | randomwalk-1.0.2.dev39+7f0c421.dirty-43dfdf | batchSize | 2 | $.config.operation.parameters |
    #
    # Handling this case requires us to:
    #   - Remove the field selector from the path
    #   - Use the field selector as key
    #   - Searching for the value
    #
    ######################################################
    #
    # An example of the path pointing to an array value is:
    #   ado get operation -q 'config.spaces=space-dfdc98-43534b'
    #
    # Which translates to
    #   - candidate = space-dfdc98-43534b
    #   - path = $.config.spaces
    #
    # When creating the json_tree we will see that:
    #   - The path is the one provided by the user
    #   - The key is the index of the array
    #   - The value is the candidate
    #
    # | identifier | key | value | path |
    # | ------------------------------------------------- | - | ------------------- | --------------- |
    # | randomwalk-0.8.3.dev46+g054e2ff6.d20250425-beaef5 | 0 | space-dfdc98-43534b | $.config.spaces |
    #
    # Handling this case requires us to not make any assumption
    # about the key
    #
    ######################################################
    #
    # Given that we cannot know for sure which of the three cases
    # we are in because it would require us to retrieve data from
    # the database, we must OR the three clauses.
    last_dot_index = path.rfind(".")
    if isinstance(candidate, _ScalarType):
        return [
            (
                f"{preamble} "
                f"(F.key LIKE '{path[2:]}%' AND F.value {_searchable_scalar_value_for_query_string(candidate)}) OR "
                f"(F.path LIKE '{path}' AND F.value {_searchable_scalar_value_for_query_string(candidate)}) OR "
                f"(F.path = '{path[:last_dot_index]}' AND "
                f"F.key = '{path[last_dot_index + 1 :]}' AND "
                f"F.value {_searchable_scalar_value_for_query_string(candidate)})"
            )
        ]

    # We have handled an immediate scalar case, so we need to now handle:
    #   - Arrays (lists)
    #   - Objects (dictionaries)
    # Both can be iterated, returning either list elements or keys
    for field in candidate:
        # If the list element or the dictionary key is not a scalar, we need recursion.
        # Example:
        #   - ado get operation -q 'status=[{"event": "finished", "exit_state": "success"}]'
        if isinstance(field, list | dict):
            fragments.extend(
                check_field_in_sqlite_json_document(field, path, id_column)
            )
            continue

        # When dealing with lists we use recursion to ensure we process
        # their contents.
        if isinstance(candidate, list):
            fragments.extend(
                check_field_in_sqlite_json_document(field, path, id_column)
            )
            continue

        # We now know that:
        #   - candidate is a dictionary
        #   - field is a scalar that we can use to index the dictionary
        #
        # We need to check the type of candidate[field]:
        #   - If it's an array or an object, we need to use recursion. We will
        #     also update the path to keep track of the fact that we explored
        #     one field of the object.
        #   - If it's a scalar, we can create a query with all the information
        #     we have available.
        if isinstance(candidate[field], list | dict):
            # The use of % in the path is because json_tree will add list items in the path.
            # (e.g., $.config.entitySpace[2].propertyDomain). As we can't know for sure
            # whether a field is a list or not, we use the LIKE operator and a wildcard (%)

            # SQLite quotes field names containing underscores in json_tree paths.
            # Quote the field name if it contains an underscore to match SQLite's behavior.
            field_pattern = f'"{field}"' if "_" in field else field

            fragments.extend(
                check_field_in_sqlite_json_document(
                    candidate[field], f"{path}%.{field_pattern}", id_column
                )
            )
            continue

        # Here we need the % wildcard because we might be dealing
        # with an array field, for which the path would contain
        # the index.
        if isinstance(candidate[field], _ScalarType):
            # Note: We do NOT quote the field name in F.key because the 'key' column
            # in json_tree is never quoted. Only intermediate fields in the 'path'
            # column are quoted when they contain underscores.
            fragments.append(
                f"{preamble} F.path LIKE '{path}%' AND "
                f"F.key = '{field}' AND "
                f"F.value {_searchable_scalar_value_for_query_string(candidate[field])}"
            )

    return fragments
