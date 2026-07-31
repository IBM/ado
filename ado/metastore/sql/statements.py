# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
from types import NoneType
from typing import TYPE_CHECKING, Literal

import sqlalchemy

from ado.core.resources import ADOResource

if TYPE_CHECKING:
    from ado.core.resources import CoreResourceKinds

# The resource graph can include operation→operation nesting and document
# edges, so a larger cap is needed; 10 is sufficient for any realistic chain.
_MAX_HIERARCHY_HOPS = 10


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


def table_exists_query(
    tablename: str,
    dialect: Literal["mysql", "sqlite"],
) -> sqlalchemy.TextClause:
    """Return a bound SQL query that checks whether a table exists in the database.

    ``dialect`` is a `sqlalchemy.engine.Dialect.name` (e.g. ``mysql``, ``sqlite``).

    Args:
        tablename: The name of the table to check for.
        dialect: "mysql" or "sqlite"

    Returns:
        A bound :class:`sqlalchemy.TextClause` that returns one row when the
        table exists and no rows when it does not.

    Raises:
        ValueError: If ``dialect`` is neither sqlite nor mysql.
    """
    if dialect == "sqlite":
        return sqlalchemy.text(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:name"
        ).bindparams(name=tablename)
    if dialect == "mysql":
        return sqlalchemy.text(
            "SELECT 1 FROM information_schema.tables"
            " WHERE table_schema = DATABASE() AND table_name = :name LIMIT 1"
        ).bindparams(name=tablename)
    raise ValueError(
        f"Unsupported dialect for table_exists_query: {dialect!r} "
        "(expected 'sqlite' or 'mysql')"
    )


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
        else (
            f"{statement_preamble} "
            f"(JSON_CONTAINS(data, '{candidate}', '{path}') "
            f"OR ('{candidate}' = 'null' "
            f"AND NOT JSON_CONTAINS_PATH(data, 'one', '{path}')))"
        )
    )


def resource_select_data_field(
    field_name: str,
    needs_select: bool = False,
    dialect: Literal["mysql", "sqlite"] = "mysql",
    output_field_name: str | None = None,
) -> str:
    """Extract a field from the JSON ``data`` column.

    Args:
        field_name: Dot-notation path within the JSON ``data`` object
            (e.g. ``"status"`` or ``"config.spaces[0]"``).
        needs_select: If True, prefix with SELECT; otherwise prefix with a
            comma (for appending to an existing SELECT clause).
        dialect: The SQL dialect to use. Determines the JSON extraction syntax.
        output_field_name: SQL alias for the extracted column. Defaults to
            ``field_name`` when not provided.

    Returns:
        A SQL fragment that extracts ``$.{field_name}`` from the ``data``
        column and aliases it as ``output_field_name`` (or ``field_name``).
    """
    statement_preamble = "SELECT" if needs_select else ","
    data_path = f"$.{field_name}"
    alias = output_field_name or field_name
    statement = (
        "{statement_preamble} data ->> '{data_path}' as {alias}"
        if dialect == "sqlite"
        else "{statement_preamble} data->>'{data_path}' as {alias}"
    )

    return statement.format(
        statement_preamble=statement_preamble,
        data_path=data_path,
        alias=alias,
    )


def resource_select_metadata_field(
    field_name: str,
    needs_select: bool = False,
    dialect: Literal["mysql", "sqlite"] = "mysql",
) -> str:

    #
    statement_preamble = "SELECT" if needs_select else ","

    data_path = f"$.config.metadata.{field_name}"

    # MySQL returns a JSON null instead of an SQL NULL when
    # a field is null. Harmonize it by forcing a SQL NULL.
    statement = (
        "{statement_preamble} data ->> '{data_path}' as {field_name}"
        if dialect == "sqlite"
        else "{statement_preamble} NULLIF(data->>'{data_path}', 'null') as {field_name}"
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
        statement = """STR_TO_DATE(data->>"$.created", '%%Y-%%m-%%dT%%T.%%fZ')"""
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
        kind=resource.kind.value,
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


def resource_select_latest_by_kinds(
    kinds: list[str],
    dialect: Literal["mysql", "sqlite"] = "mysql",
) -> sqlalchemy.TextClause:
    """Generate SQL to fetch the most recently created resource for each specified kind.

    Uses a window function (ROW_NUMBER) to rank resources by creation time within
    each kind, then filters to only the most recent (rank=1) for each kind.

    Args:
        kinds: List of resource kind values to query
        dialect: SQL dialect ("mysql" or "sqlite") - unused, kept for API consistency

    Returns:
        Bound SQLAlchemy TextClause with kinds parameter

    Example result:
        identifier                          | kind              | created
        ------------------------------------|-------------------|-------------------------
        space-f3660b-default                | discoveryspace    | 2026-04-13T08:00:00.000Z
        randomwalk-1.7.0-c47119             | operation         | 2026-04-13T09:00:00.000Z
    """
    if not kinds:
        raise ValueError("kinds list cannot be empty")

    # Both MySQL and SQLite support this syntax
    # Use parameter binding with expanding=True for the IN clause
    return sqlalchemy.text("""
        WITH ranked_resources AS (
            SELECT
                identifier,
                kind,
                data->>'$.created' as created,
                ROW_NUMBER() OVER (
                    PARTITION BY kind
                    ORDER BY data->>'$.created' DESC
                ) as rn
            FROM resources
            WHERE kind IN :kinds
        )
        SELECT identifier, kind, created
        FROM ranked_resources
        WHERE rn = 1
    """).bindparams(sqlalchemy.bindparam("kinds", value=kinds, expanding=True))


def graph_traversal_query(
    kind: "CoreResourceKinds",
    relationship_direction: Literal["outgoing", "incoming", "both"],
    origin_identifiers: set[str],
    max_hops: int | None = None,
    dialect: Literal["sqlite", "mysql"] = "sqlite",
) -> sqlalchemy.TextClause:
    """Build and return a recursive CTE traversal SQL for the resource graph.

    The query uses the raw ``resource_relationships`` table as logical edges.

    Traversal starts from every identifier in ``origin_identifiers`` whose
    resource kind matches ``kind`` and follows edges according to
    ``relationship_direction``, with a ``visited_path`` cycle guard.

    Args:
        kind: Starting resource kind.
        relationship_direction: ``'outgoing'`` (follow edges where the current
            node is the ``from`` end), ``'incoming'`` (follow edges where the
            current node is the ``to`` end), or ``'both'`` (follow edges in
            either direction).
        origin_identifiers: Set of start resource identifiers to bind.
        max_hops: Maximum number of hops to follow from the start resource.
            When ``None`` the traversal runs to the full depth cap
            (``_MAX_HIERARCHY_HOPS``). Must be a positive integer; values
            exceeding the cap are silently capped.
        dialect: SQL dialect — ``'sqlite'`` (default) or ``'mysql'``. Controls
            string-concatenation syntax (``||`` vs ``CONCAT()``).

    Returns:
        A bound :class:`sqlalchemy.TextClause` ready to execute.

    Raises:
        ValueError: If ``relationship_direction`` is invalid.
    """
    if relationship_direction not in {"outgoing", "incoming", "both"}:
        raise ValueError(
            "relationship_direction must be 'outgoing', 'incoming' or 'both', "
            f"got {relationship_direction!r}"
        )

    if max_hops is not None and max_hops < 1:
        raise ValueError(f"max_hops must be a positive integer, got {max_hops!r}")

    effective_max_hops = (
        _MAX_HIERARCHY_HOPS if max_hops is None else min(max_hops, _MAX_HIERARCHY_HOPS)
    )

    # logical_edges is a non-recursive CTE; WITH RECURSIVE is required at the
    # clause level by SQLite because the subsequent traversal CTE is recursive.
    #
    # The traversal works directly from stored relationships in
    # ``resource_relationships`` using subject→object as the logical direction.
    logical_edges_sql = """
        WITH RECURSIVE logical_edges AS (
            SELECT parent.identifier AS from_identifier,
                   parent.kind       AS from_kind,
                   child.identifier  AS to_identifier,
                   child.kind        AS to_kind
            FROM   resource_relationships rr
            JOIN   resources parent
              ON   parent.identifier = rr.subject_identifier
            JOIN   resources child
              ON   child.identifier = rr.object_identifier
        )
    """

    # Build the traversal CTE body depending on relationship_direction.
    # All variants use a visited_path cycle guard seeded as ',id,' so that
    # membership checks are unambiguous prefix/suffix matches.
    #
    # SQLite uses || for string concatenation; MySQL uses CONCAT().
    def _concat(*parts: str) -> str:
        if dialect == "sqlite":
            return " || ".join(parts)
        return f"CONCAT({', '.join(parts)})"

    if relationship_direction == "outgoing":
        step_join = "logical_edges.from_identifier = traversal.current_identifier"
        next_id = "logical_edges.to_identifier"
        next_kind = "logical_edges.to_kind"
    elif relationship_direction == "incoming":
        step_join = "logical_edges.to_identifier = traversal.current_identifier"
        next_id = "logical_edges.from_identifier"
        next_kind = "logical_edges.from_kind"
    else:  # "both"
        step_join = (
            "logical_edges.from_identifier = traversal.current_identifier"
            " OR logical_edges.to_identifier = traversal.current_identifier"
        )
        next_id = (
            "CASE WHEN logical_edges.from_identifier = traversal.current_identifier"
            " THEN logical_edges.to_identifier"
            " ELSE logical_edges.from_identifier END"
        )
        next_kind = (
            "CASE WHEN logical_edges.from_identifier = traversal.current_identifier"
            " THEN logical_edges.to_kind"
            " ELSE logical_edges.from_kind END"
        )

    seed_path = _concat("','", "origin.identifier", "','")
    step_path = _concat("traversal.visited_path", next_id, "','")
    guard_like = _concat("'%,'", next_id, "',%'")

    traversal_sql = f"""
        , traversal AS (
            SELECT origin.identifier  AS origin_identifier,
                   origin.kind        AS current_kind,
                   origin.identifier  AS current_identifier,
                   0                  AS depth,
                   {seed_path}        AS visited_path
            FROM   resources origin
            WHERE  origin.kind = :start_kind
              AND  origin.identifier IN :origins

            UNION ALL

            SELECT traversal.origin_identifier  AS origin_identifier,
                   {next_kind}                  AS current_kind,
                   {next_id}                    AS current_identifier,
                   traversal.depth + 1          AS depth,
                   {step_path}                  AS visited_path
            FROM   traversal
            JOIN   logical_edges
              ON   {step_join}
            WHERE  traversal.depth < {effective_max_hops}
              AND  traversal.visited_path NOT LIKE {guard_like}
        )
        SELECT traversal.origin_identifier AS origin_identifier,
               traversal.current_identifier AS identifier,
               traversal.current_kind AS kind
        FROM   traversal
        WHERE  traversal.depth > 0
    """  # noqa: S608

    binds = [
        sqlalchemy.bindparam(key="start_kind", value=kind.value),
        sqlalchemy.bindparam(
            key="origins", value=list(origin_identifiers), expanding=True
        ),
    ]

    return sqlalchemy.text(logical_edges_sql + traversal_sql).bindparams(*binds)
