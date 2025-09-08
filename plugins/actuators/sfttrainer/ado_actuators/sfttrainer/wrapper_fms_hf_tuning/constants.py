# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum


class AutoStopMethod(str, enum.Enum):
    # VV: The warmup phase consists of the first few optimization steps whose execution time is at least 60 seconds.
    # The stable phase is the remaining optimization steps. It lasts for at least 10 steps and at least 120 seconds.
    WARMUP_60S_STABLE_120S_OR_10_STEPS = "WARMUP_60S_STABLE_120S_OR_10_STEPS"
