# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""
Module providing Rich Console output helpers for live operation progress, spinners, and result tables.
Designed to be used for real-time visualization of long-running Discovery operations using Ray actors and the rich library for advanced command line UI.
Contains message definitions, FIFO Ray actor queue for cross-process safe messaging, and utilities for rendering and updating progress/spinner views as well as result tables from DiscoverySpace data.
"""

import queue
import typing
from typing import Literal

import ray.util.queue
from pydantic import BaseModel
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from orchestrator.core.discoveryspace.space import DiscoverySpace

if typing.TYPE_CHECKING:
    import pandas as pd


class RichConsoleMessage(BaseModel):
    """
    Base class for typed rich console UI update messages.

    Attributes:
        id (str): Unique message or progress item ID for lookup and updates.
        label (str): Human-readable display label or description for the progress item.
    """

    id: str
    label: str


class RichConsoleSpinnerMessage(RichConsoleMessage):
    """
    Message for controlling a UI spinner in the Rich Console output.
    Inherits id and label, and adds a 'state' to control spinner visibility.

    Attributes:
        state (Literal["start", "stop"]): Spinner control state ('start' to show, 'stop' to hide/mark done).
    """

    state: Literal["start", "stop"]


class RichConsoleProgressMessage(RichConsoleMessage):
    """
    Message for updating a progress bar in the Rich Console output.
    Inherits id and label, and adds 'progress' percent field.

    Attributes:
        progress (int): Progress completion as a percentage (0-100).
    """

    progress: int


RichConsoleMessageType = RichConsoleSpinnerMessage | RichConsoleProgressMessage


@ray.remote
class RichConsoleQueue:
    """
    Ray-distributed FIFO queue actor for rich console output messages.
    Used to pass spinner/progress update messages from multiple Ray actors/processes to a single UI
    updating process in a thread-safe way.
    """

    def __init__(self) -> None:
        self._queue = queue.Queue()

    def put(self, message: RichConsoleMessageType) -> None:
        """
        Append a single RichConsoleMessageType instance (progress or spinner message) to the FIFO queue.

        Args:
            message (RichConsoleMessageType): Message instance to append.

        Raises:
            ValueError: If message is not an allowed type.
        """
        # Only accept typed messages
        if not isinstance(
            message, RichConsoleSpinnerMessage | RichConsoleProgressMessage
        ):
            raise ValueError("Unsupported message type for RichConsoleQueue.put")

        self._queue.put(message)

    def get(
        self, timeout: float | None = None
    ) -> RichConsoleSpinnerMessage | RichConsoleProgressMessage | None:
        """
        Pop and return the next queued RichConsoleMessage, or None if queue is empty.

        Args:
            timeout (float|None): If None, block until a message is available. 0 for non-blocking, >0 for timed wait (currently ignored).

        Returns:
            RichConsoleSpinnerMessage | RichConsoleProgressMessage | None: The next message, or None if empty.
        """
        try:
            msg = self._queue.get_nowait()
        except queue.Empty:
            msg = None
        return msg


def rich_console_spinner_from_message(msg: RichConsoleSpinnerMessage) -> Progress:
    """
    Construct a Rich Spinner instance configured from the given spinner message.

    Args:
        msg (RichConsoleSpinnerMessage): Spinner message to create Spinner from.
    Returns:
        Spinner: Instance of rich Spinner for use in Live UI updates.
    """

    bar = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
    )
    bar.add_task(msg.label, total=None)
    return bar


def rich_console_progress_bar_from_message(msg: RichConsoleProgressMessage) -> Progress:
    """
    Construct a Rich Progress bar object from a given progress message.

    Args:
        msg (RichConsoleProgressMessage): Message with progress percent and label.
    Returns:
        Progress: Rich Progress bar object for dynamic UI updates.
    """
    bar = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
    bar.add_task(msg.label, total=100, completed=msg.progress)
    return bar


def rich_console_progress_bar_update_from_message(
    bar: Progress, msg: RichConsoleProgressMessage
) -> bool:
    """
    Update an existing Rich Progress bar with the given progress message state.

    Args:
        bar (Progress): Progress bar object to update.
        msg (RichConsoleProgressMessage): Incoming message with updated percent.
    Returns:
        bool: True if progress is complete (progress >= 100), False otherwise.
    """
    # Assumes single task, always 0th index
    # We know the progress bar must have one task from creation
    task_id = bar.task_ids[0]
    bar.update(task_id, description=msg.label, completed=(msg.progress))
    return bar.tasks[0].finished or msg.progress >= 100


def rich_console_spinner_update_from_message(
    spinner: Progress, msg: RichConsoleSpinnerMessage
) -> bool:
    """
    Update an existing Spinner with the latest label or state described by the message.
    This only updates displayed label (spinner cannot mark completion directly).

    Args:
        spinner (Spinner): Spinner object to update label/state for.
        msg (RichConsoleSpinnerMessage): Message with state and label.
    Returns:
        bool: True if spinner update marks it finished (state is 'stop'), False otherwise.
    """

    # We know the spinner must have one task from creation
    task_id = spinner.task_ids[0]
    spinner.update(task_id, description=msg.label)

    return msg.state == "stop"


def output_operation_results(
    discovery_space: DiscoverySpace, operation_id: str, row_limit: int | None = None
) -> Table:
    """
    Generate and return a Rich Table summarizing the most recent measurements for a DiscoverySpace operation.
    Dynamically determines columns shown based on terminal width. Can limit number of rows.

    Args:
        discovery_space (DiscoverySpace): Source of measurement/result timeseries.
        operation_id (str): Operation ID/tag to extract measurements for.
        row_limit (int|None): Maximum number of table rows to show (None = no limit).

    Returns:
        Table: Rendered rich.Table object containing formatted measurement data.
    """
    import numpy as np

    df: pd.DataFrame = (
        discovery_space.complete_measurement_request_with_results_timeseries(
            operation_id=operation_id,
            output_format="target",
        )
    )

    table_title = (
        f"Latest measurements - {operation_id}"
        if row_limit
        else f"Measurements - {operation_id}"
    )

    table = Table(title=table_title)

    if df.empty:
        return table

    # Remove unnecessary columns and setup index
    df = df.drop(
        columns=["result_index", "generatorid", "identifier"],
        errors="ignore",
    )
    df.insert(0, "index", np.arange(len(df)))

    # Optionally drop/convert experiment column
    if len(discovery_space.measurementSpace.experiments) == 1:
        df = df.drop(columns=["experiment_id"], errors="ignore")
    else:
        df["experiment_id"] = df["experiment_id"].apply(
            lambda x: x.experimentIdentifier
        )

    # AP 09/12/2025:
    # rich does not really give us tools to dynamically determine the height
    # of a renderable. Since we can't/don't want to use Panels to determine
    # height ratios, we try to calculate how many rows our table should have.
    #
    # NOTE: This calculation assumes that each table row is only 1-2 terminal
    # rows and there is only 1 spinner active.

    # We need to remove lines for:
    # - The table title
    # - The 3 table border segments (2 header, 1 bottom)
    # - The column names
    # - 3 more to take into account a spinner and the panel borders
    # Total: 8
    console = Console()
    if not row_limit:
        row_limit = max(int(console.height / 2) - 8, 3)

    # Determine the amount of columns the screen can fit
    # We create a support Table where we add column by column and
    # if the terminal wouldn't fit another column of minimum width
    # (based on the current size of the table), we stop adding them.
    #
    # NOTE: This is very conservative, as rich can squeeze columns
    # a bit to decrease their size.
    terminal_width = console.width
    measurement_table = Table(title=table_title)
    measurement_table.add_column("... (+XX more)")
    for column in df.columns:
        if (
            terminal_width - console.measure(measurement_table).maximum
        ) < console.measure(measurement_table).minimum:
            break
        measurement_table.add_column(column)

    max_columns = len(measurement_table.columns)
    visible_columns = list(df.columns[:max_columns])
    hidden_columns = len(df.columns) - max_columns
    if hidden_columns > 0:
        visible_columns.append(f"... (+{hidden_columns} more)")

    for col in visible_columns:
        table.add_column(col, overflow="fold")

    # We add the rows in descending order
    for row_number, (_, row) in enumerate(df[::-1].iterrows()):

        if row_limit and row_number == row_limit:
            break

        row_data = [
            (f"{row[col]:.2f}" if isinstance(row[col], float) else str(row[col]))
            for col in df.columns[:max_columns]
        ]

        if hidden_columns > 0:
            row_data.append("...")
        table.add_row(*row_data)

    return table


def render_progress_indicators(progress_items: dict) -> Panel:
    """
    Render a rich.Panel containing all currently active spinner/progress renderables.
    Used to visually group multiple progress/spinner items in the UI.

    Args:
        progress_items (dict): Mapping from string id to Progress or Spinner objects.
    Returns:
        Panel: Panel with all renderables grouped together, one per row.
    """
    return Panel(Group(*progress_items.values()))


def run_operation_live_updates(
    discovery_space: DiscoverySpace,
    operation_id: str,
    console_queue: ray.actor.ActorHandle[RichConsoleQueue],
    operation_future: ray.ObjectRef,
) -> None:
    """
    Continuously updates the live Rich Console display during the execution of a distributed operation.
    Handles periodic refresh of the operation results table and processes all messages from the Ray queue to update progress bars and spinners in real-time.
    Designed for use inside a Ray worker overseeing a DiscoverySpace operation.

    Args:
        discovery_space (DiscoverySpace): Measurement/orchestration context.
        operation_id (str): Operation identifier (tag).
        console_queue (ActorHandle): Ray actor handle for receiving rich UI message updates.
        operation_future (ObjectRef): Ray object ref for the call being awaited (used to determine finish).
    """
    finished = []
    # Dict of message.id --> renderable (Spinner or Progress)
    progress_items = {}

    with Live(
        Group(
            render_progress_indicators(progress_items=progress_items),
            output_operation_results(
                discovery_space=discovery_space,
                operation_id=operation_id,
            ),
        ),
        refresh_per_second=10,
    ) as live:
        while not finished:
            # 1. Update measurement results
            results_table = output_operation_results(
                discovery_space=discovery_space,
                operation_id=operation_id,
            )
            # Update results table now in case there are no progress items
            live.update(
                Group(
                    render_progress_indicators(progress_items=progress_items),
                    results_table,
                )
            )
            # 2. Fetch all pending messages (FIFO)
            while True:
                try:
                    msg = ray.get(console_queue.get.remote(timeout=0))
                except Exception:
                    break

                if msg is None:
                    break

                if msg.id in progress_items:
                    renderable = progress_items[msg.id]

                # Update spinner or progress item in the group
                if isinstance(msg, RichConsoleSpinnerMessage):
                    if msg.id not in progress_items:
                        renderable = rich_console_spinner_from_message(msg)
                        progress_items[msg.id] = renderable

                    renderable_is_complete = rich_console_spinner_update_from_message(
                        renderable, msg
                    )

                elif isinstance(msg, RichConsoleProgressMessage):
                    if msg.id not in progress_items:
                        renderable = rich_console_progress_bar_from_message(msg)
                        renderable._task_label = msg.label
                        progress_items[msg.id] = renderable

                    renderable_is_complete = (
                        rich_console_progress_bar_update_from_message(renderable, msg)
                    )

                if renderable_is_complete:
                    del progress_items[msg.id]

                # Update live view after every change
                live.update(
                    Group(
                        render_progress_indicators(progress_items=progress_items),
                        results_table,
                    )
                )

            finished, _ = ray.wait(ray_waitables=[operation_future], timeout=2)
        # Final whole-table output
        live.update(
            Group(
                render_progress_indicators(progress_items=progress_items),
                output_operation_results(
                    discovery_space=discovery_space,
                    operation_id=operation_id,
                ),
            )
        )
