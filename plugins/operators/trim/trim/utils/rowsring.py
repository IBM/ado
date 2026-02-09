# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from collections import deque
from collections.abc import Callable

import pandas as pd
from typing_extensions import Self


class RowsRing:
    """
    Fixed-size FIFO ring for tabular rows. Supports dict, pd.Series, or single-row pd.DataFrame.
    Ensures schema coherence (same set of columns) across rows.
    Provides a .df property that returns a pandas DataFrame view of the current ring
    with index reset to 0..size-1 (oldest → newest).

    Optionally accepts a validator(prev_df, new_row_series) -> bool to enforce domain-level coherence.
    - prev_df: DataFrame composed of the current ring contents (before adding the new row)
    - new_row_series: the new row as a Series aligned to the established schema
    """

    def __init__(
        self,
        maxlen: int,
        validator: Callable[[pd.DataFrame, pd.Series], bool] | None = None,
    ) -> None:
        if maxlen <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(maxlen)
        self._rows = deque(maxlen=self.capacity)
        self._columns: list[str] | None = None
        self._validator = validator

    def _normalize_row(self, row: dict | pd.Series | pd.DataFrame) -> pd.Series:
        """Convert input (dict / Series / single-row DataFrame) to a Series aligned to the schema."""
        if isinstance(row, pd.DataFrame):
            if len(row) != 1:
                raise ValueError("pd.DataFrame input must contain exactly one row")
            ser = row.iloc[0]
        elif isinstance(row, pd.Series):
            ser = row
        elif isinstance(row, dict):
            ser = pd.Series(row)
        else:
            raise TypeError("Row must be dict, pd.Series, or single-row pd.DataFrame")

        # Initialize schema from the first row
        if self._columns is None:
            self._columns = list(ser.index)
            # reorder to the established schema
            return ser[self._columns]

        # Enforce schema coherence: same set of columns as the ring
        if set(ser.index) != set(self._columns):
            raise ValueError(
                f"Row schema mismatch. Expected columns {self._columns}, "
                f"got {list(ser.index)}"
            )
        # Reorder to canonical column order
        return ser[self._columns]

    def append(self, row: dict | pd.Series | pd.DataFrame) -> None:
        """Append a new row after schema and optional domain validation."""
        ser = self._normalize_row(row)

        # Domain-level validation (coherence) if provided
        if self._validator is not None:
            prev = self.df  # DataFrame view before adding
            ok = self._validator(prev, ser)
            if not ok:
                raise ValueError("Validation failed for the new row")

        self._rows.append(ser)

    def __iadd__(self, row: dict | pd.Series | pd.DataFrame) -> Self:
        """
        Enable in-place add: `ring += row`
        `row` can be a dict, pd.Series, or single-row pd.DataFrame.
        """
        self.append(row)
        return self  # important: keep `yielded_rows` as RowsRing

    @property
    def df(self) -> pd.DataFrame:
        """Return a DataFrame view of the ring with index reset (0..size-1, oldest → newest)."""
        if len(self._rows) == 0:
            return pd.DataFrame(columns=self._columns or [])
        df = pd.DataFrame(list(self._rows))
        # Ensure column order stays canonical and index is 0..size-1
        if self._columns is not None:
            df = df[self._columns]
        df.index = list(range(len(df)))
        return df

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> pd.Series:
        return self._rows[idx]

    def to_list(self) -> list[pd.Series]:
        """Return the rows as a list of Series (oldest → newest)."""
        return list(self._rows)
