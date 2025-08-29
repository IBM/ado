# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
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
    # part of a JOIN to ensure they are all "AND"-ed (an actual AND is not sufficient
    # as json_tree returns separate rows).
    # We also add a progressive D<number> identifier to each subquery to avoid
    # possible complaints about ambiguous column names.
    subqueries = check_field_in_sqlite_json_document(json.loads(candidate), path)
    subqueries = [f"{s} D{idx}" for idx, s in enumerate(subqueries)]

    return (  # noqa: S608 - we don't care about local sql injection
        """
        identifier IN (
            WITH F AS (
                SELECT r.identifier, jt.key, jt.value, jt.path
                FROM
                    resources r,
                    json_tree(r.data, '{path}') jt
            )
            SELECT
                D0.identifier
            FROM {subqueries}
        )
        """
    ).format(
        path=path,
        subqueries=" JOIN ".join(subqueries),
    )


def check_field_in_sqlite_json_document(entries: dict, path: str) -> list[str]:
    """
    Check if a field exists in a SQLite JSON document.
    This method builds subqueries that, given a "database" F which contains
    at least the following keys:
        - identifier
        - key
        - value
        - path
    Check whether the entries are contained in the document.

    Parameters:
    entries (dict): A dictionary representing the JSON document.
    path (str): The path to the field to check.

    Returns:
    list[str]: A list of SQL statements to check if the field exists in the document.
    """

    fragments = []
    preamble = "(SELECT identifier FROM F WHERE "

    # We iterate over all the keys in the dictionary representing our JSON document:
    #   - If the value for the key is itself a dictionary, we use recursion to go deeper.
    #   - If the value for the key is a list, we create a subquery for every list item.
    #   - If the value for the key is a single value, we create a subquery for it.
    #
    # The use of % in the path is because json_tree will add list items in the path.
    # (e.g., $.config.entitySpace[2].propertyDomain). As we can't know for sure
    # whether a field is a list or not, we use the LIKE operator and a wildcard (%)
    for key in entries.keys():
        if isinstance(entries[key], dict):
            fragments.extend(
                check_field_in_sqlite_json_document(entries[key], f"{path}%.{key}")
            )
        elif isinstance(entries[key], list):
            for item in entries[key]:
                fragments.append(
                    f"{preamble} F.path LIKE '{path}.{key}' AND F.value = '{item}')"
                )
        else:
            fragments.append(
                f"{preamble} F.path LIKE '{path}%' AND F.value = '{entries[key]}')"
            )
    return fragments


def resource_filter_by_arbitrary_selection(
    path: str,
    candidate: str,
    needs_where: bool = False,
    dialect: Literal["mysql", "sqlite"] = "mysql",
) -> str:

    if needs_where:
        statement_preamble = " WHERE "
    else:
        statement_preamble = " AND "

    return (
        f"{statement_preamble} {simulate_json_contains_on_sqlite(path, candidate)}"
        if dialect == "sqlite"
        else "{statement_preamble} JSON_CONTAINS(data, '{candidate}', '{path}')".format(
            statement_preamble=statement_preamble,
            candidate=candidate,
            path=path,
        )
    )


def resource_select_data_field(
    field_name: str,
    needs_select: bool = False,
    dialect: Literal["mysql", "sqlite"] = "mysql",
) -> str:

    #
    if needs_select:
        statement_preamble = "SELECT"
    else:
        statement_preamble = ","

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
    if needs_select:
        statement_preamble = "SELECT"
    else:
        statement_preamble = ","

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
    as_age: bool = False, needs_select: bool = False, dialect="mysql"
) -> str:

    #
    if needs_select:
        statement_preamble = "SELECT"
    else:
        statement_preamble = ","

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
):
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
):
    if dialect == "sqlite":
        query = sqlalchemy.text(
            f"""
             INSERT OR IGNORE INTO {sample_store_name}
             (identifier, representation)
             VALUES (:identifier, :representation)
             """  # noqa: S608 - sample_store_name is not untrusted
        )
    else:
        query = sqlalchemy.text(
            f"""
            INSERT IGNORE INTO {sample_store_name}
            (identifier, representation)
            VALUES (:identifier, :representation)
            """  # noqa: S608 - sample_store_name is not untrusted
        )

    return query


def upsert_entities(
    sample_store_name: str, dialect: Literal["mysql", "sqlite"] = "mysql"
):
    if dialect == "sqlite":
        query = sqlalchemy.text(
            rf"""
            INSERT INTO {sample_store_name}
            (identifier, representation)
            VALUES (:identifier, :representation)
            ON CONFLICT(identifier) DO UPDATE SET representation = excluded.representation
            """  # noqa: S608 - sample_store_name is not untrusted
        )
    else:
        query = sqlalchemy.text(
            rf"""
            INSERT INTO {sample_store_name}
            (identifier, representation)
            VALUES (:identifier, :representation)
            ON DUPLICATE KEY UPDATE representation=values(representation)
            """  # noqa: S608 - sample_store_name is not untrusted
        )

    return query
