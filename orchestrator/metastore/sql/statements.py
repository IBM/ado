# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
from types import NoneType
from typing import TYPE_CHECKING, Literal

import sqlalchemy

from orchestrator.core.resources import ADOResource

if TYPE_CHECKING:
    from orchestrator.core.resources import CoreResourceKinds


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
    from the candidate document.

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

    # The subqueries produced by check_field_in_sqlite_json_document need to be
    # INTERSECT-ed to make sure we only retrieve the identifiers that match all
    # the subqueries.
    subqueries = check_field_in_sqlite_json_document(
        json.loads(candidate), path, id_column=quoted_id_column
    )

    return ("""
        {id_column} IN (
            WITH F AS (
                SELECT t.{id_column}, jt.key, jt.value, jt.path
                FROM
                    {table_name} t,
                    json_tree(t.{json_column}, '{path}') jt
            )
            {subqueries}
        )
        """).format(  # noqa: S608 - identifiers are quoted to prevent injection
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
        else f"{statement_preamble} JSON_CONTAINS(data, '{candidate}', '{path}')"
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
    hierarchy_direction: Literal["up", "down"],
    origin_identifiers: set[str],
    stop_at_resource_kind: "CoreResourceKinds | None" = None,
) -> sqlalchemy.TextClause:
    """Build and return the traversal SQL for the resource hierarchy.

    Each branch of the hierarchy emits three columns:

    * ``origin_identifier`` — the start resource identifier.
    * ``identifier``        — the discovered resource identifier.
    * ``kind``              — the kind of the discovered resource.

    All active branches are joined with ``UNION ALL``; Python post-processing
    handles deduplication.  The ``origins`` binding is expanded as an
    ``IN``-list by SQLAlchemy.

    Resource relationships::

        samplestore
          └── discoveryspace
                └── operation
                      └── datacontainer

        actuatorconfiguration ──► operation
                               (subject)  (object)

    actuatorconfiguration resources exist independently and are associated with
    an operation at creation time.  The association is recorded as
    ``subject_identifier=actconf``, ``object_identifier=operation`` — the
    opposite direction from the samplestore/discoveryspace/operation/datacontainer
    chain, where parent resources are ``subject_identifier`` and children are
    ``object_identifier`` (see ``addResourceWithRelationships``).

    Allowed traversal families
    --------------------------
    * ``up``   from ``operation``      (→ discoveryspace → samplestore)
    * ``up``   from ``discoveryspace`` (→ samplestore)
    * ``down`` from ``samplestore``    (→ discoveryspace → operation → dc/ac)
    * ``down`` from ``discoveryspace`` (→ operation → dc/ac)
    * ``down`` from ``operation``      (→ datacontainer / actuatorconfiguration)

    Args:
        kind: Starting resource kind.
        hierarchy_direction: ``'up'`` (child → parent) or ``'down'`` (parent → child).
        origin_identifiers: Set of start resource identifiers to bind.
        stop_at_resource_kind: When provided, only branches reaching up to and
            including this kind are included.  Must be reachable from
            ``kind`` in the requested ``hierarchy_direction``; raises ``ValueError``
            otherwise.

    Returns:
        A bound :class:`sqlalchemy.TextClause` ready to execute.

    Raises:
        ValueError: If ``hierarchy_direction`` is not ``'up'`` or ``'down'``.
        ValueError: If ``(kind, hierarchy_direction)`` is not a supported family.
        ValueError: If ``stop_at_resource_kind`` is not reachable in the family.
    """
    from orchestrator.core.resources import CoreResourceKinds

    if hierarchy_direction not in ("up", "down"):
        raise ValueError(
            f"hierarchy_direction must be 'up' or 'down', got {hierarchy_direction!r}"
        )

    _K = CoreResourceKinds
    _SS = _K.SAMPLESTORE.value
    _DS = _K.DISCOVERYSPACE.value
    _OP = _K.OPERATION.value
    _DC = _K.DATACONTAINER.value
    _AC = _K.ACTUATORCONFIGURATION.value

    # Each branch is (deepest CoreResourceKinds reached, sql_string).
    # The sql_string uses the literal token :origins which is bound below.
    # All kind values in the SQL are from CoreResourceKinds.value — not user input.
    _branches: list[tuple[CoreResourceKinds, str]]

    if kind == _K.OPERATION and hierarchy_direction == "up":
        _branches = [
            (  # op → discoveryspace
                _K.DISCOVERYSPACE,
                f"""
                SELECT rr1.object_identifier  AS origin_identifier,
                       rr1.subject_identifier AS identifier,
                       r1.kind                AS kind
                FROM   resource_relationships rr1
                JOIN   resources r1
                  ON   r1.identifier = rr1.subject_identifier
                 AND   r1.kind = '{_DS}'
                WHERE  rr1.object_identifier IN :origins
                """,  # noqa: S608 - kind values are enum literals, not user input
            ),
            (  # op → discoveryspace → samplestore
                _K.SAMPLESTORE,
                f"""
                SELECT rr1.object_identifier  AS origin_identifier,
                       rr2.subject_identifier AS identifier,
                       r2.kind                AS kind
                FROM   resource_relationships rr1
                JOIN   resources r1
                  ON   r1.identifier = rr1.subject_identifier
                 AND   r1.kind = '{_DS}'
                JOIN   resource_relationships rr2
                  ON   rr2.object_identifier = rr1.subject_identifier
                JOIN   resources r2
                  ON   r2.identifier = rr2.subject_identifier
                 AND   r2.kind = '{_SS}'
                WHERE  rr1.object_identifier IN :origins
                """,  # noqa: S608
            ),
        ]

    elif kind == _K.DISCOVERYSPACE and hierarchy_direction == "up":
        _branches = [
            (  # discoveryspace → samplestore
                _K.SAMPLESTORE,
                f"""
                SELECT rr1.object_identifier  AS origin_identifier,
                       rr1.subject_identifier AS identifier,
                       r1.kind                AS kind
                FROM   resource_relationships rr1
                JOIN   resources r1
                  ON   r1.identifier = rr1.subject_identifier
                 AND   r1.kind = '{_SS}'
                WHERE  rr1.object_identifier IN :origins
                """,  # noqa: S608
            ),
        ]

    elif kind == _K.SAMPLESTORE and hierarchy_direction == "down":
        _branches = [
            (  # samplestore → discoveryspace
                _K.DISCOVERYSPACE,
                f"""
                SELECT rr1.subject_identifier AS origin_identifier,
                       rr1.object_identifier  AS identifier,
                       r1.kind                AS kind
                FROM   resource_relationships rr1
                JOIN   resources r1
                  ON   r1.identifier = rr1.object_identifier
                 AND   r1.kind = '{_DS}'
                WHERE  rr1.subject_identifier IN :origins
                """,  # noqa: S608
            ),
            (  # samplestore → discoveryspace → operation
                _K.OPERATION,
                f"""
                SELECT rr1.subject_identifier AS origin_identifier,
                       rr2.object_identifier  AS identifier,
                       r2.kind                AS kind
                FROM   resource_relationships rr1
                JOIN   resources r1
                  ON   r1.identifier = rr1.object_identifier
                 AND   r1.kind = '{_DS}'
                JOIN   resource_relationships rr2
                  ON   rr2.subject_identifier = rr1.object_identifier
                JOIN   resources r2
                  ON   r2.identifier = rr2.object_identifier
                 AND   r2.kind = '{_OP}'
                WHERE  rr1.subject_identifier IN :origins
                """,  # noqa: S608
            ),
            (  # samplestore → discoveryspace → operation → datacontainer
                _K.DATACONTAINER,
                f"""
                SELECT rr1.subject_identifier AS origin_identifier,
                       rr3.object_identifier  AS identifier,
                       r3.kind                AS kind
                FROM   resource_relationships rr1
                JOIN   resources r1
                  ON   r1.identifier = rr1.object_identifier
                 AND   r1.kind = '{_DS}'
                JOIN   resource_relationships rr2
                  ON   rr2.subject_identifier = rr1.object_identifier
                JOIN   resources r2
                  ON   r2.identifier = rr2.object_identifier
                 AND   r2.kind = '{_OP}'
                JOIN   resource_relationships rr3
                  ON   rr3.subject_identifier = rr2.object_identifier
                JOIN   resources r3
                  ON   r3.identifier = rr3.object_identifier
                 AND   r3.kind = '{_DC}'
                WHERE  rr1.subject_identifier IN :origins
                """,  # noqa: S608
            ),
            (  # samplestore → discoveryspace → operation → actuatorconfiguration
                # actconf is subject, operation is object (reverse of other hops)
                _K.ACTUATORCONFIGURATION,
                f"""
                SELECT rr1.subject_identifier AS origin_identifier,
                       rr3.subject_identifier AS identifier,
                       r3.kind                AS kind
                FROM   resource_relationships rr1
                JOIN   resources r1
                  ON   r1.identifier = rr1.object_identifier
                 AND   r1.kind = '{_DS}'
                JOIN   resource_relationships rr2
                  ON   rr2.subject_identifier = rr1.object_identifier
                JOIN   resources r2
                  ON   r2.identifier = rr2.object_identifier
                 AND   r2.kind = '{_OP}'
                JOIN   resource_relationships rr3
                  ON   rr3.object_identifier = rr2.object_identifier
                JOIN   resources r3
                  ON   r3.identifier = rr3.subject_identifier
                 AND   r3.kind = '{_AC}'
                WHERE  rr1.subject_identifier IN :origins
                """,  # noqa: S608
            ),
        ]

    elif kind == _K.DISCOVERYSPACE and hierarchy_direction == "down":
        _branches = [
            (  # discoveryspace → operation
                _K.OPERATION,
                f"""
                SELECT rr1.subject_identifier AS origin_identifier,
                       rr1.object_identifier  AS identifier,
                       r1.kind                AS kind
                FROM   resource_relationships rr1
                JOIN   resources r1
                  ON   r1.identifier = rr1.object_identifier
                 AND   r1.kind = '{_OP}'
                WHERE  rr1.subject_identifier IN :origins
                """,  # noqa: S608
            ),
            (  # discoveryspace → operation → datacontainer
                _K.DATACONTAINER,
                f"""
                SELECT rr1.subject_identifier AS origin_identifier,
                       rr2.object_identifier  AS identifier,
                       r2.kind                AS kind
                FROM   resource_relationships rr1
                JOIN   resources r1
                  ON   r1.identifier = rr1.object_identifier
                 AND   r1.kind = '{_OP}'
                JOIN   resource_relationships rr2
                  ON   rr2.subject_identifier = rr1.object_identifier
                JOIN   resources r2
                  ON   r2.identifier = rr2.object_identifier
                 AND   r2.kind = '{_DC}'
                WHERE  rr1.subject_identifier IN :origins
                """,  # noqa: S608
            ),
            (  # discoveryspace → operation → actuatorconfiguration
                # actconf is subject, operation is object (reverse of other hops)
                _K.ACTUATORCONFIGURATION,
                f"""
                SELECT rr1.subject_identifier AS origin_identifier,
                       rr2.subject_identifier AS identifier,
                       r2.kind                AS kind
                FROM   resource_relationships rr1
                JOIN   resources r1
                  ON   r1.identifier = rr1.object_identifier
                 AND   r1.kind = '{_OP}'
                JOIN   resource_relationships rr2
                  ON   rr2.object_identifier = rr1.object_identifier
                JOIN   resources r2
                  ON   r2.identifier = rr2.subject_identifier
                 AND   r2.kind = '{_AC}'
                WHERE  rr1.subject_identifier IN :origins
                """,  # noqa: S608
            ),
        ]

    elif kind == _K.OPERATION and hierarchy_direction == "down":
        _branches = [
            (  # operation → datacontainer
                _K.DATACONTAINER,
                f"""
                SELECT rr1.subject_identifier AS origin_identifier,
                       rr1.object_identifier  AS identifier,
                       r1.kind                AS kind
                FROM   resource_relationships rr1
                JOIN   resources r1
                  ON   r1.identifier = rr1.object_identifier
                 AND   r1.kind = '{_DC}'
                WHERE  rr1.subject_identifier IN :origins
                """,  # noqa: S608
            ),
            (  # operation → actuatorconfiguration
                # actconf is subject, operation is object (reverse of other hops)
                _K.ACTUATORCONFIGURATION,
                f"""
                SELECT rr1.object_identifier  AS origin_identifier,
                       rr1.subject_identifier AS identifier,
                       r1.kind                AS kind
                FROM   resource_relationships rr1
                JOIN   resources r1
                  ON   r1.identifier = rr1.subject_identifier
                 AND   r1.kind = '{_AC}'
                WHERE  rr1.object_identifier IN :origins
                """,  # noqa: S608
            ),
        ]

    else:
        raise ValueError(
            f"Unsupported traversal family: kind={kind!r}, hierarchy_direction={hierarchy_direction!r}. "
            f"Allowed families: up from operation/discoveryspace; "
            f"down from samplestore/discoveryspace/operation."
        )

    # Apply stop_at_resource_kind trimming
    if stop_at_resource_kind is not None:
        reachable_kinds = [b[0] for b in _branches]
        if stop_at_resource_kind not in reachable_kinds:
            raise ValueError(
                f"stop_at_resource_kind={stop_at_resource_kind!r} is not reachable from "
                f"kind={kind!r} with hierarchy_direction={hierarchy_direction!r}. "
                f"Reachable kinds: {reachable_kinds}"
            )
        cutoff = reachable_kinds.index(stop_at_resource_kind)
        _branches = _branches[: cutoff + 1]

    full_sql = "\nUNION ALL\n".join(sql for _, sql in _branches)
    return sqlalchemy.text(full_sql).bindparams(
        sqlalchemy.bindparam(
            key="origins", value=list(origin_identifiers), expanding=True
        )
    )
