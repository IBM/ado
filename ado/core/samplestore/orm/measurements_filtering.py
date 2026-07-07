# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Filter builder for measurement queries using SQLAlchemy."""

import json
import logging
from typing import Literal

import sqlalchemy
from sqlalchemy import Column, func, text
from sqlalchemy.sql import ColumnElement

from ado.metastore.sql.statements import simulate_json_contains_on_sqlite

logger = logging.getLogger(__name__)


# Direct column mappings for measurement requests
# Maps field names to actual database column names
MEASUREMENT_REQUEST_COLUMN_MAPPINGS = {
    "requestid": "request_id",
    "requestIndex": "request_index",
    "status": "status",
    "timestamp": "timestamp",
    "type": "type",
    "experimentReference": "experiment_reference",
}

# Direct column mappings for measurement results
# Maps field names to actual database column names
MEASUREMENT_RESULT_COLUMN_MAPPINGS = {
    "uid": "uid",
    "entityIdentifier": "entity_id",  # Schema field: entityIdentifier (camelCase)
}


class MeasurementFilterBuilder:
    """
    Builds SQLAlchemy filter conditions for measurement queries.

    Handles both direct column access and JSON field queries with
    proper dialect-specific SQL generation.
    """

    def __init__(self, dialect: Literal["mysql", "sqlite"] = "mysql") -> None:
        """
        Initialize the filter builder.

        Args:
            dialect: Database dialect (mysql or sqlite)
        """
        self.dialect = dialect

    def _parse_filter_value(
        self, json_encoded_value: str
    ) -> str | int | float | bool | dict | list | None:
        """
        Parse a JSON-encoded filter value.

        Args:
            json_encoded_value: JSON-encoded value string

        Returns:
            Parsed Python value
        """
        try:
            return json.loads(json_encoded_value)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse filter value '{json_encoded_value}': {e}")
            return json_encoded_value

    def _build_json_contains_condition(
        self,
        json_column: Column,
        json_path: str,
        value: str,
        table_name: str,
        id_column: str = "uid",
    ) -> ColumnElement | sqlalchemy.TextClause:
        """
        Build a JSON contains condition for the appropriate dialect.

        Args:
            json_column: The JSON column to query
            json_path: JSON path (e.g., "$.metadata.key")
            value: JSON-encoded value to match
            table_name: Name of the table (for SQLite simulation)
            id_column: Name of the ID column (for SQLite simulation)

        Returns:
            SQLAlchemy filter condition
        """
        if self.dialect == "mysql":
            # MySQL: JSON_CONTAINS(column, value, path) = 1
            return func.json_contains(json_column, value, json_path) == 1

        # SQLite: Use the proper simulation of JSON_CONTAINS
        # This generates a complex query with json_tree and INTERSECT
        sql_fragment = simulate_json_contains_on_sqlite(
            path=json_path,
            candidate=value,
            table_name=table_name,
            json_column=json_column.name,
            id_column=id_column,
        )
        # Return as a text() condition that can be used in WHERE clause
        return text(sql_fragment)

    def build_request_filter(
        self,
        request_table: sqlalchemy.Table,
        path: str,
        json_encoded_value: str,
    ) -> ColumnElement | sqlalchemy.TextClause:
        """
        Build a filter condition for measurement requests.

        Args:
            request_table: The measurement_requests table
            path: Field path (e.g., "status", "metadata.key")
                  Can optionally start with "$." which will be stripped
            json_encoded_value: JSON-encoded value to match

        Returns:
            SQLAlchemy filter condition

        Raises:
            ValueError: If path is not a known column or JSON field
        """
        # Strip leading "$." if present for compatibility
        path = path.removeprefix("$.")

        # Check if path maps to a direct column (top-level field)
        column_name = MEASUREMENT_REQUEST_COLUMN_MAPPINGS.get(path)
        if column_name:
            value = self._parse_filter_value(json_encoded_value)
            return request_table.c[column_name] == value

        # Check if path explicitly targets metadata JSON column
        if path.startswith("metadata.") or path == "metadata":
            json_path = f"$.{path}" if not path.startswith("$.") else path
            return self._build_json_contains_condition(
                request_table.c.metadata,
                json_path,
                json_encoded_value,
                table_name=request_table.name,
                id_column="uid",
            )

        # Invalid path - not a known column and not a metadata field
        valid_columns = ", ".join(sorted(MEASUREMENT_REQUEST_COLUMN_MAPPINGS.keys()))
        raise ValueError(
            f"Invalid filter path '{path}'. Must be either a known column "
            f"({valid_columns}) or a metadata field (metadata.* or metadata)"
        )

    def build_result_filter(
        self,
        result_table: sqlalchemy.Table,
        path: str,
        json_encoded_value: str,
    ) -> ColumnElement | sqlalchemy.TextClause:
        """
        Build a filter condition for measurement results.

        Args:
            result_table: The measurement_results table
            path: Field path (e.g., "uid", "entityIdentifier", or any field in data JSON)
                  Can optionally start with "$." which will be stripped
            json_encoded_value: JSON-encoded value to match

        Returns:
            SQLAlchemy filter condition

        Raises:
            ValueError: If path is not a known column or data field
        """
        # Strip leading "$." if present for compatibility
        path = path.removeprefix("$.")

        # Check if path maps to a direct column (top-level field)
        column_name = MEASUREMENT_RESULT_COLUMN_MAPPINGS.get(path)
        if column_name:
            value = self._parse_filter_value(json_encoded_value)
            return result_table.c[column_name] == value

        # For measurement results, any other path queries the data JSON column
        # (unlike requests which only allow metadata.* paths)
        json_path = f"$.{path}" if not path.startswith("$.") else path
        return self._build_json_contains_condition(
            result_table.c.data,
            json_path,
            json_encoded_value,
            table_name=result_table.name,
            id_column="uid",
        )

    def apply_filters(
        self,
        filters: list[dict[str, str]],
        table: sqlalchemy.Table,
        filter_type: Literal["request", "result"],
    ) -> list[ColumnElement | sqlalchemy.TextClause]:
        """
        Apply multiple filters and return a list of conditions.

        Args:
            filters: List of filter dictionaries [{"path": "value"}, ...]
            table: The table to filter on
            filter_type: Type of filter ("request" or "result")

        Returns:
            List of SQLAlchemy filter conditions
        """
        conditions = []

        for filter_dict in filters:
            for path, value in filter_dict.items():
                if filter_type == "request":
                    condition = self.build_request_filter(table, path, value)
                else:
                    condition = self.build_result_filter(table, path, value)
                conditions.append(condition)

        return conditions


# Made with Bob
