# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import random
import string


def get_random_name_extension(length=5):
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    return "".join(random.choice(alphabet) for i in range(length))
