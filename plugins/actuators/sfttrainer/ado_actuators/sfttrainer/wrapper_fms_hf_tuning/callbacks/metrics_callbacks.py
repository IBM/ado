# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
from typing import Any

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from . import metrics_tracker


class ResourceMetricsCallbacks(TrainerCallback):
    """Example implementation of a TrainerCallback that uses the ResourceTracker"""

    def __init__(self, period: float | None = None) -> None:
        """Initializes the callback with a resource tracker.

        Args:
            period:
                Seconds between successive metric snapshots, or None for default.
        """
        if period is None:
            self.tracker = metrics_tracker.ResourceTracker()
        else:
            self.tracker = metrics_tracker.ResourceTracker(period=period)

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Starts metric tracking when training begins.

        Args:
            args:
                Training arguments.
            state:
                Current trainer state.
            control:
                Trainer control flow state.
            **kwargs:
                Additional keyword arguments.
        """
        self.tracker.begin_track()

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Stops metric tracking when training ends.

        Args:
            args:
                Training arguments.
            state:
                Current trainer state.
            control:
                Trainer control flow state.
            **kwargs:
                Additional keyword arguments.
        """
        self.tracker.end_track()

    def __del__(self) -> None:
        """Stops the resource tracker when this object goes out of scope."""
        try:
            tracker = self.tracker
        except AttributeError:
            return
        tracker.end_track()
