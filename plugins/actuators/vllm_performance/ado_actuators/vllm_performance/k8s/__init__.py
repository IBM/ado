# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


class K8sEnvironmentCreationError(Exception):
    """Error raised when K8 environment cannot be created for some reason"""


class K8sConnectionError(Exception):
    """Error raised when there is an issue connecting to K8s or a service its hosting"""
