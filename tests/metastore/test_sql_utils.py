# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest
import sqlalchemy

from ado.metastore.sql.statements import table_exists_query
from ado.metastore.sql.utils import check_table_exists


def test_table_exists_query_sqlite() -> None:
    q = table_exists_query("t", dialect="sqlite")
    assert "sqlite_master" in str(q)


def test_table_exists_query_mysql_dialect_name() -> None:
    q = table_exists_query("t", dialect="mysql")
    assert "information_schema" in str(q)


def test_table_exists_query_unsupported_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported dialect"):
        table_exists_query("t", dialect="postgresql")


def test_check_table_exists_sqlite_memory() -> None:
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    assert check_table_exists(engine, "missing") is False

    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("CREATE TABLE missing (id INTEGER)"))

    assert check_table_exists(engine, "missing") is True
