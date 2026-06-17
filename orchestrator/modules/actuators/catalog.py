# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import abc
import enum
import logging
from typing import Annotated, Literal

import pydantic

from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.reference import ExperimentReference


class ActuatorCatalogExtensionConf(pydantic.BaseModel):
    """Represents a dynamically loadable set of experiments for an actuator"""

    name: Annotated[
        str, pydantic.Field(description="The name of the catalog extension")
    ]
    location: Annotated[
        str, pydantic.Field(description="The location of the catalog extension")
    ]

    @property
    def catalogExtensionLocation(self) -> str:
        import os

        return os.path.join(self.location, self.name)


class ActuatorCatalogExtension(pydantic.BaseModel):
    """A list of experiments that can be added to a catalog

    TODO: This should be combined with ExperimentCatalog so they are the same class.
    Holding off on this as ExperimentCatalog has internal data-structures that need to change
    """

    experiments: Annotated[
        list[Experiment], pydantic.Field(description="A list of experiments")
    ]


class BaseCatalog(abc.ABC):

    @property
    @abc.abstractmethod
    def experiments(self) -> list[Experiment]:
        pass

    @property
    @abc.abstractmethod
    def experiment_semantic_identifiers(self) -> list[str]:
        pass

    @abc.abstractmethod
    def experimentForReference(self, reference: ExperimentReference) -> Experiment:
        pass


class ExperimentCatalog(BaseCatalog):
    """Base class for class that provide information on the available experiments"""

    def __init__(
        self, experiments: dict | None = None, catalogIdentifier: str = "UnnamedCatalog"
    ) -> None:
        """
        Parameters:
            experiments: A dictionary whose keys are experiment identifiers
                The values are orchestrator.model.data.Experiment instances

        :return: An ExperimentCatalog subclass
        """

        import os

        LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
        logging.basicConfig(level=LOGLEVEL)
        self.log = logging.getLogger("experiment-catalog")
        self._identifier = catalogIdentifier
        self._experiments = experiments if experiments is not None else {}

    def __str__(self) -> str:

        return f"Catalog {self._identifier} with {len(self._experiments)} experiments"

    @property
    def experiments(self) -> list[Experiment]:
        return list(self._experiments.values())

    @property
    def supported_experiments(self) -> list[Experiment]:
        return [e for e in self.experiments if not e.deprecated]

    @property
    def deprecated_experiments(self) -> list[Experiment]:
        return [e for e in self.experiments if e.deprecated]

    @property
    def identifier(self) -> str:

        return self._identifier

    @property
    def experiment_semantic_identifiers(self) -> list[str]:
        """Return the semantic identifiers of the experiments in the catalog

        Returns:
            Dict keyed by each experiment's semantic identifier.
        """
        return [e.semantic_identifier for e in self.experiments]

    def experimentForReference(
        self, reference: ExperimentReference
    ) -> Experiment | None:
        """Return the experiment matching reference or None if there is no match.

        Matching compares on actuator and semantic experiment identifier.
        Parameterization on the reference is ignored for catalog lookup purposes.

        The catalog stores at most one experiment per semantic identifier, so
        this method returns either a single match or ``None``.

        Args:
            reference: The experiment reference to look up.

        Returns:
            The matching Experiment, or None if no match is found.
        """
        for experiment in self.experiments:
            if (
                experiment.reference.actuatorIdentifier == reference.actuatorIdentifier
                and experiment.reference.semantic_experiment_identifier
                == reference.semantic_experiment_identifier
            ):
                return experiment
        return None

    def experiments_matching_identifier(
        self, reference: ExperimentReference
    ) -> list[Experiment]:
        """Return experiments with the same actuator and base experiment as reference

        Args:
            reference: The experiment reference whose identifier and actuator
                should be matched.

        Returns:
            Experiments with the same actuator and experiment identifier.
        """
        return [
            e
            for e in self.experiments
            if e.actuatorIdentifier == reference.actuatorIdentifier
            and e.identifier == reference.experimentIdentifier
        ]

    def addExperiment(self, experiment: Experiment) -> None:
        """Add an experiment to the catalog.

        Args:
            experiment: The experiment to add.

        Raises:
            ValueError: If an experiment with the same semantic identifier is
                already present
        """

        existing = self._experiments.get(experiment.semantic_identifier)
        if existing is not None:
            if existing.model_dump() == experiment.model_dump():
                # Identical experiment already registered — idempotent re-add is fine
                return
            raise ValueError(
                f"An experiment with semantic identifier {experiment.semantic_identifier!r} "
                f"is already registered in catalog {self._identifier!r}. "
            )

        self._experiments[experiment.semantic_identifier] = experiment

    def resolve_reference(
        self,
        reference: ExperimentReference,
        mode: Literal["semantic", "fully_qualified"] = "semantic",
    ) -> Experiment | ParameterizedExperiment:
        """Resolve a reference to an experiment, including parameterization if any

        This is the preferred entry point for execution paths (actuator
        ``submit`` methods, registry checks).

        The method:

        1. Looks up the experiment using :meth:`experimentForReference` (semantic
           comparison).
        2. For ``mode='fully_qualified'`` additionally requires the exact version
           to match — raises :class:`AlgorithmVersionMismatchError` on mismatch.
        3. Raises :class:`~orchestrator.modules.actuators.base.DeprecatedExperimentError`
           if the resolved experiment is deprecated.
        4. Wraps the result in a :class:`~orchestrator.schema.experiment.ParameterizedExperiment`
           if the reference carries parameterization.

        Args:
            reference: The experiment reference to resolve.
            mode: ``"semantic"`` (default) — Match on MAJOR version
                ``"fully_qualified"`` — additionally requires the exact version
                (MAJOR.MINOR.PATCH) to match.

        Returns:
            The resolved :class:`~orchestrator.schema.experiment.Experiment` or
            :class:`~orchestrator.schema.experiment.ParameterizedExperiment`.

        Raises:
            AlgorithmVersionMismatchError: When ``mode='fully_qualified'`` and
                the resolved experiment's version does not match the reference's
                version.
            :class:`~orchestrator.modules.actuators.registry.UnknownExperimentError`:
                If no matching experiment is found.
            :class:`~orchestrator.modules.actuators.base.DeprecatedExperimentError`:
                If the resolved experiment is marked deprecated.
        """
        from orchestrator.modules.actuators.base import DeprecatedExperimentError
        from orchestrator.modules.actuators.registry import UnknownExperimentError

        experiment = self.experimentForReference(reference)
        if experiment is None:
            raise UnknownExperimentError(
                f"No experiment matching {reference!s} found in catalog {self._identifier!r}."
            )

        if (
            mode == "fully_qualified"
            and experiment.fully_qualified_identifier
            != reference.fully_qualified_experiment_identifier
        ):
            raise ExperimentVersionMismatchError(
                f"Algorithm version mismatch for experiment "
                f"{reference.experimentIdentifier!r} in catalog {self._identifier!r}. "
                f"Reference requires version "
                f"{reference.fully_qualified_experiment_identifier!r} but catalog "
                f"provides {experiment.fully_qualified_identifier!r}."
            )

        if experiment.deprecated:
            raise DeprecatedExperimentError(
                f"{experiment.actuatorIdentifier}.{experiment.identifier} is deprecated."
            )

        if reference.parameterization:
            return ParameterizedExperiment(
                parameterization=reference.parameterization, **experiment.model_dump()
            )
        return experiment


class ExperimentVersionMismatchError(Exception):
    """Raised when the  version of a resolved experiment does not match the reference.

    This error is only raised when :meth:`ExperimentCatalog.resolve_reference` is
    called with ``mode='fully_qualified'`` and the exact version in the catalog
    differs from the version recorded on the :class:`~orchestrator.schema.reference.ExperimentReference`.
    """


class ExperimentNotInCatalogError(Exception):

    pass


class CatalogConfigurationRequirementEnum(enum.Enum):

    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    OPTIONAL = "optional"
