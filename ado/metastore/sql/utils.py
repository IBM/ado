# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import sqlalchemy

from ado.utilities.location import SQLStoreConfiguration
from ado.utilities.pydantic import pydantic_aware_json_serializer

# Process-level cache: reuse the same SQLAlchemy Engine (and its connection pool)
# for every call with the same database URL.  This means the metastore and the
# samplestore — which both point at the same MySQL server — share one pool and
# avoid the overhead of opening a second TCP connection.
_engine_cache: dict[str, sqlalchemy.Engine] = {}

# The resource graph can include operation→operation nesting and document
# edges, so a larger cap is needed; 10 is sufficient for any realistic chain.
_MAX_HIERARCHY_HOPS = 10


def engine_for_sql_store(
    configuration: SQLStoreConfiguration, database: str | None = None
) -> sqlalchemy.Engine:
    """Return a SQLAlchemy Engine for the given store configuration.

    Engines are cached by their connection URL so that multiple components
    connecting to the same database reuse a single connection pool rather than
    each opening their own TCP connection.

    Args:
        configuration: Database connection parameters.
        database: Optional database name override.

    Returns:
        A (possibly cached) SQLAlchemy Engine.
    """
    if configuration is None:
        raise ValueError("engine_for_sql_store requires a valid SQLStoreConfiguration")

    configuration.database = (
        database if database is not None else configuration.database
    )

    # AP 28/04/2025:
    # We cannot use the URL method for SQLite, as it will URL-encode
    # the path. In case there is a space in the path, it will become
    # %20, causing failures. This is particularly problematic as the
    # default directory for ado on macOS is:
    # /Users/$USER/Library/Application Support/
    db_location = (
        f"sqlite:///{configuration.path}"
        if configuration.scheme == "sqlite"
        else configuration.url().unicode_string()
    )

    if db_location in _engine_cache:
        return _engine_cache[db_location]

    engine_args: dict = {
        "echo": False,
        "json_serializer": pydantic_aware_json_serializer,
    }
    if configuration.scheme != "sqlite":
        # Prevent "Lost connection to MySQL server during query" (error 2013) when
        # connections sit idle during long-running operations (e.g. CPLEX trials).
        # pool_pre_ping: test connections before use, evict stale ones
        # This adds some latency on engine creation to test connection
        engine_args["pool_pre_ping"] = True
        # pool_recycle: recycle connections before MySQL wait_timeout (often 3600s)
        # This is the alternative but requires knowing the timeout of the db
        # Other components on the connection also may close the connection at
        # other unknown intervals
        # engine_args["pool_recycle"] = 1800

    engine = sqlalchemy.create_engine(db_location, **engine_args)
    _engine_cache[db_location] = engine
    return engine


def json_extract_field_as_string(
    col: sqlalchemy.Column,
    path: str,
) -> sqlalchemy.ColumnElement:
    """Extract a scalar text value from a JSON column using the ``->>`` operator.

    The ``->>`` operator extracts a scalar string from a JSON document.
    Wrapping the result in :func:`sqlalchemy.cast` with :class:`sqlalchemy.String`
    prevents SQLAlchemy's JSON column type processor from attempting to
    JSON-decode the already-extracted scalar value, which would raise a
    ``JSONDecodeError`` for plain strings or silently coerce numeric strings
    to ``int``.

    Args:
        col: The JSON :class:`~sqlalchemy.Column` to extract from.
        path: A JSON path expression (e.g. ``$.config.metadata.name``).

    Returns:
        A SQLAlchemy column expression that evaluates to the extracted scalar
        text value, typed as :class:`sqlalchemy.String`.
    """
    return sqlalchemy.cast(
        col.op("->>")(sqlalchemy.literal(path)),
        sqlalchemy.String,
    )


def create_sql_resource_store(
    engine: sqlalchemy.Engine,
) -> sqlalchemy.MetaData:
    """Create the metastore tables if they do not exist and return the schema.

    Args:
        engine: SQLAlchemy engine for the target database.

    Returns:
        A :class:`~sqlalchemy.MetaData` instance containing the
        ``resources`` and ``resource_relationships`` table definitions.
    """
    from sqlalchemy import JSON, String

    meta = sqlalchemy.MetaData()

    sqlalchemy.Table(
        "resources",
        meta,
        sqlalchemy.Column("identifier", String(256), primary_key=True),
        sqlalchemy.Column("kind", String(256), index=True),
        sqlalchemy.Column("version", String(128)),
        sqlalchemy.Column("data", JSON(False)),
    )

    sqlalchemy.Table(
        "resource_relationships",
        meta,
        sqlalchemy.Column(
            "subject_identifier",
            String(256),
            sqlalchemy.ForeignKey("resources.identifier"),
            primary_key=True,
        ),
        sqlalchemy.Column(
            "object_identifier",
            String(256),
            sqlalchemy.ForeignKey("resources.identifier"),
            primary_key=True,
        ),
    )

    meta.create_all(engine, checkfirst=True)

    return meta
