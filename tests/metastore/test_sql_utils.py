# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import sqlalchemy

from ado.metastore.sql.statements import simulate_json_contains_on_sqlite
from tests.conftest import requires_sqlite_3_38


@requires_sqlite_3_38
def test_simulate_json_contains_on_sqlite_matches_null_and_missing_fields() -> None:
    engine = sqlalchemy.create_engine("sqlite:///:memory:")

    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text("CREATE TABLE resources (identifier TEXT, data TEXT)")
        )
        conn.execute(
            sqlalchemy.text(
                """
                INSERT INTO resources (identifier, data)
                VALUES
                    ('stored-null', '{"config": {"metadata": {"name": null}}}'),
                    ('missing-field', '{"config": {"metadata": {}}}'),
                    ('non-null', '{"config": {"metadata": {"name": "ado"}}}')
                """
            )
        )

        where_clause = simulate_json_contains_on_sqlite(
            "$.config.metadata.name", "null"
        )
        query = sqlalchemy.text(
            f"""
            SELECT identifier FROM resources
            WHERE {where_clause}
            ORDER BY identifier
            """  # noqa: S608 - test builds a SQL fragment from a controlled helper
        )
        result = conn.execute(query).scalars().all()

    assert result == ["missing-field", "stored-null"]
