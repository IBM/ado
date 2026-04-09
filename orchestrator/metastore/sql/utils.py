# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import sqlalchemy

from orchestrator.metastore.sql.statements import table_exists_query
from orchestrator.utilities.location import SQLStoreConfiguration

# Process-level cache: reuse the same SQLAlchemy Engine (and its connection pool)
# for every call with the same database URL.  This means the metastore and the
# samplestore — which both point at the same MySQL server — share one pool and
# avoid the overhead of opening a second TCP connection.
_engine_cache: dict[str, sqlalchemy.Engine] = {}


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

    engine_args: dict = {"echo": False}
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


def check_table_exists(engine: sqlalchemy.Engine, tablename: str) -> bool:
    """Return whether ``tablename`` exists in the database behind ``engine``.

    First tries a single round-trip using :func:`table_exists_query` with
    ``engine.dialect.name``. On any exception (unsupported dialect, execution
    error, etc.), falls back to :func:`sqlalchemy.inspect` and
    :meth:`~sqlalchemy.engine.reflection.Inspector.has_table`.

    Args:
        engine: SQLAlchemy engine for the target database.
        tablename: Unqualified table name to check.

    Returns:
        ``True`` if the table exists, ``False`` otherwise.
    """
    try:
        query = table_exists_query(tablename, dialect=engine.dialect.name)
        with engine.connect() as conn:
            return conn.execute(query).fetchone() is not None
    except Exception:
        inspector = sqlalchemy.inspect(engine)
        return inspector.has_table(tablename)


def create_sql_resource_store(engine: sqlalchemy.Engine) -> sqlalchemy.Engine:
    from sqlalchemy import JSON, String

    # Create the tables if they don't exist
    meta = sqlalchemy.MetaData()

    resources = sqlalchemy.Table(  # noqa: F841
        "resources",
        meta,
        sqlalchemy.Column("identifier", String(256), primary_key=True),
        sqlalchemy.Column("kind", String(256), index=True),
        sqlalchemy.Column("version", String(128)),
        # Use to store resource objecte (1MB)
        sqlalchemy.Column("data", JSON(False)),
    )

    # Holds relationships between two objects
    # Since the predicate between two kinds is known we don't have to store it
    resourceRelationships = sqlalchemy.Table(  # noqa: F841
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

    return engine
