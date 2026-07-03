# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import abc
import enum
import logging
from typing import Annotated, Literal

import pydantic

from orchestrator.modules.actuators.errors import (
    AmbiguousExperimentIdentifierError,
    DeprecatedExperimentError,
    ExperimentVersionMismatchError,
    UnknownExperimentError,
)
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.reference import ExperimentReference

ExperimentReferenceMatchMode = Literal[
    "major_version",
    "fully_qualified_version",
    "base",
    "any",
]


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
    def experiment_major_version_identifiers(self) -> list[str]:
        pass

    @abc.abstractmethod
    def experimentForReference(
        self,
        reference: ExperimentReference,
        *,
        match_on: ExperimentReferenceMatchMode = "major_version",
        resolve: bool = False,
    ) -> Experiment | ParameterizedExperiment | None:
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
    def experiment_major_version_identifiers(self) -> list[str]:
        """Return the major version identifiers of the experiments in the catalog

        Returns:
            A list of major version identifiers.
        """
        return [e.major_version_identifier for e in self.experiments]

    def _ambiguous_experiment_identifier_error(
        self, reference: ExperimentReference, matches: list[Experiment]
    ) -> AmbiguousExperimentIdentifierError:
        """Build an error when multiple catalog experiments share a base identifier."""
        available_versions = ", ".join(
            sorted({matched.version for matched in matches if matched.version})
        )
        return AmbiguousExperimentIdentifierError(
            f"The given identifier, {reference.experimentIdentifier!r}, is ambiguous: "
            f"catalog contains {len(matches)} versions "
            f"({available_versions}). "
            f"Specify a version suffix, e.g. "
            f"{reference.experimentIdentifier}@<version>."
        )

    def _check_and_parameterize_experiment(
        self,
        reference: ExperimentReference,
        experiment: Experiment,
        *,
        resolve: bool,
    ) -> Experiment | ParameterizedExperiment:
        """Apply resolve-time checks and parameterization wrapping."""
        if resolve and experiment.deprecated:
            raise DeprecatedExperimentError(
                f"{experiment.actuatorIdentifier}.{experiment.identifier} is deprecated."
            )

        if resolve and reference.parameterization:
            return ParameterizedExperiment(
                parameterization=reference.parameterization, **experiment.model_dump()
            )
        return experiment

    def experimentForReference(
        self,
        reference: ExperimentReference,
        *,
        match_on: ExperimentReferenceMatchMode = "major_version",
        resolve: bool = False,
    ) -> Experiment | ParameterizedExperiment | None:
        """Return the experiment matching reference.

        Matching compares on actuator and major version experiment identifier.
        When ``match_on='fully_qualified_version'``, the exact version must also
        match. When ``match_on='base'``, matching uses the base experiment
        identifier only. When ``match_on='any'``, matching tries fully qualified,
        major, then base matching in order. When ``resolve=True``, deprecated
        experiments cause an error to be raised and references with
        parameterization are wrapped in :class:`ParameterizedExperiment`.

        The catalog stores at most one experiment per major version identifier.

        Args:
            reference: The experiment reference to look up.
            match_on: Matching mode. ``"major_version"`` (default) matches on
                MAJOR version only. ``"fully_qualified_version"`` additionally
                requires the exact version (MAJOR.MINOR.PATCH) to match.
                ``"base"`` matches on actuator and base experiment identifier.
                ``"any"`` tries fully qualified, major, then base matching.
            resolve: When ``True``, raise on miss or version mismatch, reject
                deprecated experiments, and apply parameterization from the
                reference. When ``False``, return ``None`` on miss or version
                mismatch.

        Returns:
            The matching :class:`~orchestrator.schema.experiment.Experiment` or
            :class:`~orchestrator.schema.experiment.ParameterizedExperiment`, or
            ``None`` if no match is found and ``resolve=False``.

        Raises:
            AmbiguousExperimentIdentifierError: When ``match_on='base'`` or
                ``match_on='any'`` finds multiple catalog versions for the same
                base identifier.
            ExperimentVersionMismatchError: When ``resolve=True``,
                ``match_on='fully_qualified_version'``, and the resolved
                experiment's version does not match the reference's version.
            UnknownExperimentError: If no matching experiment is found and
                ``resolve=True``.
            DeprecatedExperimentError: If the resolved experiment is marked
                deprecated and ``resolve=True``.
        """
        if match_on == "any":
            experiment: Experiment | None = None
            for mode in ("fully_qualified_version", "major_version", "base"):
                experiment = self.experimentForReference(
                    reference, match_on=mode, resolve=False
                )
                if experiment is not None:
                    break
            if experiment is None:
                if resolve:
                    raise UnknownExperimentError(
                        f"No experiment matching {reference!s} found in catalog {self._identifier!r}."
                    )
                return None
            return self._check_and_parameterize_experiment(
                reference, experiment, resolve=resolve
            )

        if match_on == "base":
            matches = self.experiments_matching_identifier(reference)
            if len(matches) == 0:
                if resolve:
                    raise UnknownExperimentError(
                        f"No experiment matching {reference!s} found in catalog {self._identifier!r}."
                    )
                return None
            if len(matches) > 1:
                raise self._ambiguous_experiment_identifier_error(reference, matches)
            return self._check_and_parameterize_experiment(
                reference, matches[0], resolve=resolve
            )

        experiment = None
        for candidate in self.experiments:
            if (
                candidate.reference.actuatorIdentifier == reference.actuatorIdentifier
                and candidate.reference.major_version_experiment_identifier
                == reference.major_version_experiment_identifier
            ):
                experiment = candidate
                break

        if experiment is None:
            if resolve:
                raise UnknownExperimentError(
                    f"No experiment matching {reference!s} found in catalog {self._identifier!r}."
                )
            return None

        if (
            match_on == "fully_qualified_version"
            and experiment.fully_qualified_identifier
            != reference.fully_qualified_experiment_identifier
        ):
            if resolve:
                raise ExperimentVersionMismatchError(
                    f"Algorithm version mismatch for experiment "
                    f"{reference.experimentIdentifier!r} in catalog {self._identifier!r}. "
                    f"Reference requires version "
                    f"{reference.fully_qualified_experiment_identifier!r} but catalog "
                    f"provides {experiment.fully_qualified_identifier!r}."
                )
            return None

        return self._check_and_parameterize_experiment(
            reference, experiment, resolve=resolve
        )

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
            ValueError: If an experiment with the same major version identifier is
                already present
        """

        existing = self._experiments.get(experiment.major_version_identifier)
        if existing is not None:
            if existing.model_dump() == experiment.model_dump():
                # Identical experiment already registered — idempotent re-add is fine
                return
            raise ValueError(
                f"An experiment with major version identifier {experiment.major_version_identifier!r} "
                f"is already registered in catalog {self._identifier!r}. "
            )

        self._experiments[experiment.major_version_identifier] = experiment


class CatalogConfigurationRequirementEnum(enum.Enum):

    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    OPTIONAL = "optional"
