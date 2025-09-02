# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from . import metrics_tracker


class ResourceMetricsCallbacks(TrainerCallback):
    def __init__(self, period: float | None = None):
        if period is None:
            period = 30.0

        self.tracker = metrics_tracker.ResourceTracker(period=period)

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        self.tracker.begin_track()

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        self.tracker.end_track()

    def __del__(self):
        # VV: ensure that we kill the thread when this object goes out of scope
        self.tracker.end_track()
