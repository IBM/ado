# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


class InsufficientDataError(Exception):
    """
    Raised when the dataset is too small to allow for reliable validation or testing.
    """

    def __init__(self, message="Not enough Data retrieved from the space"):
        self.message = message
        super().__init__(self.message)
