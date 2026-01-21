# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
from types import NoneType
from typing import Literal

import sqlalchemy

from orchestrator.core.resources import ADOResource


def simulate_json_contains_on_sqlite(path: str, candidate: str) -> str:
    """
    Simulate MySQL's JSON_CONTAINS on SQLite.
    On MySQL, JSON_CONTAINS allows searching for a JSON document within a JSON field.
    It matches all documents that contains at least the provided JSON document.

    In our simulated version, we prepare a subquery that can be used in a WHERE statement
    that filters resources making sure their identifier is one that has all the fields
    from the candidate document.

    Args:
        path (str): The path to the JSON field to check.
        candidate (str): The JSON document to check.

    Returns:
        str: The SQLite query that checks whether the provided document exists.
    """

    # The subqueries produced by check_field_in_sqlite_json_document need to be
    # INTERSECT-ed to make sure we only retrieve the identifiers that match all
    # the subqueries.
    subqueries = check_field_in_sqlite_json_document(json.loads(candidate), path)

    return ("""
        identifier IN (
            WITH F AS (
                SELECT r.identifier, jt.key, jt.value, jt.path
                FROM
                    resources r,
                    json_tree(r.data, '{path}') jt
            )
            {subqueries}
        )
        """).format(  # noqa: S608 - we don't care about local sql injection
        path=path,
        subqueries="\n            INTERSECT ".join(subqueries),
    )


def check_field_in_sqlite_json_document(
    candidate: dict | list | str | float, path: str
) -> list[str]:
    """
    Generate SQLite-compatible SQL fragments to check for the presence of specific fields or values
    within a JSON document using the json_tree virtual table.

    This function recursively traverses the input JSON-like structure (dictionary, list, or scalar)
    and constructs SQL subqueries that can be used to filter rows produced by SQLite's json_tree
    function based on whether the specified fields and values exist at the given JSON path.

    Args:
        candidate (dict | list | str | int | float): The JSON structure or scalar value to match against.
            - If a scalar (str, int, float), generates a simple query checking for value presence.
            - If a dict or list, recursively builds queries for nested fields and values.
        path (str): The JSON path (e.g., '$.config.spaces') used to locate the field within the document.

    Returns:
        list[str]: A list of SQL SELECT statements that can be combined via INTERSECT
        to filter rows whose JSON documents contain the specified structure or values.
    """
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
    preamble = "SELECT identifier FROM F WHERE "

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
            f"{preamble} "
            f"(F.key LIKE '{path[2:]}%' AND F.value {_searchable_scalar_value_for_query_string(candidate)}) OR "
            f"(F.path LIKE '{path}' AND F.value {_searchable_scalar_value_for_query_string(candidate)}) OR "
            f"(F.path = '{path[:last_dot_index]}' AND "
            f"F.key = '{path[last_dot_index+1:]}' AND "
            f"F.value {_searchable_scalar_value_for_query_string(candidate)})"
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
            fragments.extend(check_field_in_sqlite_json_document(field, path))
            continue

        # When dealing with lists we use recursion to ensure we process
        # their contents.
        if isinstance(candidate, list):
            fragments.extend(check_field_in_sqlite_json_document(field, path))
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
            fragments.extend(
                check_field_in_sqlite_json_document(
                    candidate[field], f"{path}%.{field}"
                )
            )
            continue

        # Here we need the % wildcard because we might be dealing
        # with an array field, for which the path would contain
        # the index.
        if isinstance(candidate[field], _ScalarType):
            fragments.append(
                f"{preamble} F.path LIKE '{path}%' AND "
                f"F.key = '{field}' AND "
                f"F.value {_searchable_scalar_value_for_query_string(candidate[field])}"
            )

    return fragments


def resource_filter_by_arbitrary_selection(
    path: str,
    candidate: str,
    needs_where: bool = False,
    dialect: Literal["mysql", "sqlite"] = "mysql",
) -> str:

    statement_preamble = " WHERE " if needs_where else " AND "

    return (
        f"{statement_preamble} {simulate_json_contains_on_sqlite(path, candidate)}"
        if dialect == "sqlite"
        else f"{statement_preamble} JSON_CONTAINS(data, '{candidate}', '{path}')"
    )


def resource_select_data_field(
    field_name: str,
    needs_select: bool = False,
    dialect: Literal["mysql", "sqlite"] = "mysql",
) -> str:

    #
    statement_preamble = "SELECT" if needs_select else ","

    #
    data_path = f"$.{field_name}"
    statement = (
        "{statement_preamble} data -> '{data_path}' as {field_name}"
        if dialect == "sqlite"
        else "{statement_preamble} data->'{data_path}' as {field_name}"
    )

    return statement.format(
        statement_preamble=statement_preamble,
        data_path=data_path,
        field_name=field_name,
    )


def resource_select_metadata_field(
    field_name: str,
    needs_select: bool = False,
    dialect: Literal["mysql", "sqlite"] = "mysql",
) -> str:

    #
    statement_preamble = "SELECT" if needs_select else ","

    data_path = f"$.config.metadata.{field_name}"
    statement = (
        "{statement_preamble} data -> '{data_path}' as {field_name}"
        if dialect == "sqlite"
        else "{statement_preamble} data->'{data_path}' as {field_name}"
    )

    return statement.format(
        statement_preamble=statement_preamble,
        data_path=data_path,
        field_name=field_name,
    )


def resource_select_created_field(
    as_age: bool = False,
    needs_select: bool = False,
    dialect: Literal["mysql", "sqlite"] = "mysql",
) -> str:

    #
    statement_preamble = "SELECT" if needs_select else ","

    if dialect == "sqlite":
        if as_age:
            statement = """ROUND((JULIANDAY(DATETIME('NOW')) - JULIANDAY(DATETIME(data ->> '$.created'))) * 86400) as age"""
        else:
            statement = """DATETIME(data ->> '$.created')) as created"""

    else:
        # FIXME AP 23/04/2024:
        # Now that we have added timezone information to the timestamps, the created
        # field may end with Z (zulu), causing the STR_TO_DATE function to return NaT
        # As a workaround, we ensure that all our dates end with Z
        # We also use JSON_UNQUOTE because by default JSON_EXTRACT returns quoted fields
        # in mysql (-> is an alias)
        dates_in_correct_format = (
            'IF(JSON_UNQUOTE(data->"$.created") LIKE "%%Z", '
            'JSON_UNQUOTE(data->"$.created"), '
            'CONCAT(JSON_UNQUOTE(data->"$.created"), "Z"))'
        )
        statement = (
            f"""STR_TO_DATE({dates_in_correct_format}, '%%Y-%%m-%%dT%%T.%%fZ')"""
        )
        if as_age:
            statement = f"""TIMESTAMPDIFF(SECOND, {statement}, NOW()) as age"""
        else:
            statement += " as created"

    return f"{statement_preamble} {statement}"


def resource_order_by_age_desc(dialect: Literal["mysql", "sqlite"] = "mysql") -> str:
    return (
        "ORDER BY age IS NOT NULL, age DESC"
        if dialect == "sqlite"
        else "ORDER BY -ISNULL(age), age DESC"
    )


def resource_upsert(
    resource: ADOResource,
    json_representation: dict,
    dialect: Literal["mysql", "sqlite"] = "mysql",
) -> sqlalchemy.TextClause:
    if dialect == "sqlite":
        return sqlalchemy.text(
            r"INSERT INTO resources "
            r"(identifier, kind, version, data) "
            r"VALUES(:identifier, :kind, :version, :data) "
            r"ON CONFLICT(identifier) DO UPDATE SET data = excluded.data"
        ).bindparams(
            identifier=resource.identifier,
            kind=resource.kind.value,
            version=resource.version,
            data=json_representation,
        )
    return sqlalchemy.text(
        r"INSERT INTO resources"
        r"(identifier, kind, version, data)"
        r"VALUES(:identifier, :kind, :version, :data)"
        r"ON DUPLICATE KEY UPDATE data = values(data)"
    ).bindparams(
        identifier=resource.identifier,
        kind=resource.kind,
        version=resource.version,
        data=json_representation,
    )


def insert_entities_ignore_on_duplicate(
    sample_store_name: str, dialect: Literal["mysql", "sqlite"] = "mysql"
) -> sqlalchemy.TextClause:
    if dialect == "sqlite":
        query = sqlalchemy.text(f"""
             INSERT OR IGNORE INTO {sample_store_name}
             (identifier, representation)
             VALUES (:identifier, :representation)
             """)  # noqa: S608 - sample_store_name is not untrusted
    else:
        query = sqlalchemy.text(f"""
            INSERT IGNORE INTO {sample_store_name}
            (identifier, representation)
            VALUES (:identifier, :representation)
            """)  # noqa: S608 - sample_store_name is not untrusted

    return query


def upsert_entities(
    sample_store_name: str, dialect: Literal["mysql", "sqlite"] = "mysql"
) -> sqlalchemy.TextClause:
    if dialect == "sqlite":
        query = sqlalchemy.text(rf"""
            INSERT INTO {sample_store_name}
            (identifier, representation)
            VALUES (:identifier, :representation)
            ON CONFLICT(identifier) DO UPDATE SET representation = excluded.representation
            """)  # noqa: S608 - sample_store_name is not untrusted
    else:
        query = sqlalchemy.text(rf"""
            INSERT INTO {sample_store_name}
            (identifier, representation)
            VALUES (:identifier, :representation)
            ON DUPLICATE KEY UPDATE representation=values(representation)
            """)  # noqa: S608 - sample_store_name is not untrusted

    return query
