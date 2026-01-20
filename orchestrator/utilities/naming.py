# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import random
import string


def get_random_name_extension(length: int = 5) -> str:
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    return "".join(
        random.choice(alphabet)  # noqa: S311 - not cryptographic purposes
        for i in range(length)
    )
