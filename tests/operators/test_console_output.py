# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests that live-console result tables survive sample-store read failures."""

import pandas as pd
from rich.table import Table

from ado.modules.operators.console_output import output_operation_results


class _RaisingDiscoverySpace:
    """Stand-in DiscoverySpace whose timeseries read fails like a MySQL drop."""

    def complete_measurement_request_with_results_timeseries(
        self,
        operation_id: str,
        output_format: str,
    ) -> pd.DataFrame:
        raise SystemError(
            f"Unable to get the measurement requests for operation {operation_id}. "
            "Error: (pymysql.err.OperationalError) (2013, "
            "'Lost connection to MySQL server during query')"
        )


class _FlakyDiscoverySpace:
    """Returns one successful timeseries frame, then raises SystemError."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe
        self.calls = 0
        self.measurementSpace = type("MeasurementSpace", (), {"experiments": ["exp"]})()

    def complete_measurement_request_with_results_timeseries(
        self,
        operation_id: str,
        output_format: str,
    ) -> pd.DataFrame:
        self.calls += 1
        if self.calls > 1:
            raise SystemError(
                f"Unable to get the measurement requests for operation {operation_id}. "
                "Error: (pymysql.err.OperationalError) (2013, "
                "'Lost connection to MySQL server during query')"
            )
        return self._dataframe


def test_output_operation_results_returns_table_when_sample_store_read_fails() -> None:
    """A sample-store read error must not raise; the live UI gets an empty table."""
    table = output_operation_results(
        discovery_space=_RaisingDiscoverySpace(),  # type: ignore[arg-type]
        operation_id="ray_tune@2.0.9-optuna-b99dcd",
    )

    assert isinstance(table, Table)
    assert "temporarily unavailable" in (table.title or "")
    assert table.row_count == 0


def test_output_operation_results_keeps_fallback_table_when_refresh_fails() -> None:
    """A later failed refresh must keep the last successfully rendered table."""
    dataframe = pd.DataFrame(
        {
            "request_index": [0],
            "result_index": [0],
            "identifier": ["entity-1"],
            "experiment_id": ["custom_experiments.solve_mip"],
            "objective_values-mean": [1.5],
        }
    )
    space = _FlakyDiscoverySpace(dataframe)
    first_table = output_operation_results(
        discovery_space=space,  # type: ignore[arg-type]
        operation_id="op-1",
    )
    assert space.calls == 1
    assert isinstance(first_table, Table)
    assert first_table.row_count == 1

    second_table = output_operation_results(
        discovery_space=space,  # type: ignore[arg-type]
        operation_id="op-1",
        fallback=first_table,
    )

    assert space.calls == 2
    assert second_table is first_table
