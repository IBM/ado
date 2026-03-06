# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import sqlalchemy

from orchestrator.utilities.location import SQLStoreConfiguration


def engine_for_sql_store(
    configuration: SQLStoreConfiguration, database: str | None = None
) -> sqlalchemy.Engine:
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
    return sqlalchemy.create_engine(db_location, **engine_args)


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
