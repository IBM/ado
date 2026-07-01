# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


class OperatorVersionMismatchError(Exception):
    """Raised when the version of a resolved operator does not match the reference.

    This error is raised when :func:`resolve_operator_reference` is called with
    an explicit ``operatorVersion`` that differs from the version registered in
    the operator collection.
    """
