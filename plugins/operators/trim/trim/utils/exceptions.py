# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


class InsufficientDataError(Exception):
    """
    Raised when the dataset is too small to allow for reliable validation or testing.
    """

    def __init__(
        self, message: str = "Not enough Data retrieved from the space"
    ) -> None:
        self.message = message
        super().__init__(self.message)
