# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


class ExperimentVersionMismatchError(Exception):
    """Raised when the version of a resolved experiment does not match the reference.

    This error is only raised when :meth:`ExperimentCatalog.experimentForReference` is
    called with ``resolve=True``, ``match_on='fully_qualified_version'``, and the
    exact version in the catalog differs from the version recorded on the
    :class:`~orchestrator.schema.reference.ExperimentReference`.
    """


class ExperimentNotInCatalogError(Exception):

    pass


class UnknownExperimentError(Exception):
    pass


class AmbiguousExperimentIdentifierError(Exception):
    """There are multiple matches for the given identifier in the catalog"""


class DeprecatedExperimentError(Exception):
    """Raised when an actuator is attempting to run an experiment that has been deprecated."""


class MissingConfigurationForExperimentError(Exception):
    """Raised when an actuator is attempting to run an experiment but required configuration information is not present"""


class UnknownActuatorError(Exception):
    """The actuator was never registered to the registry"""


class MissingActuatorConfigurationForCatalogError(Exception):
    """The actuator requires configuration information for it catalog, but it hasn't been provided"""


class UnexpectedCatalogRetrievalError(Exception):
    """The actuator catalog method raised on unexpected exception"""


class MeasurementError(Exception):
    """Raised when an error occurs while an actuator is measuring properties of entities."""
