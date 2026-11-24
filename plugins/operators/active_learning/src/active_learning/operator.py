# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Measurement collection and predictor output shared by active-learning operators."""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from active_learning.data import candidate_entities, measured_labels
from active_learning.parameters import (
    ActiveLearningParameters,
    ActiveLearningPredictorMetadata,
)
from active_learning.sampler import ForestSampleSelector, build_forest
from ado.core.operation.operation import OperationOutput
from ado.core.operation.resource import OperationExitStateEnum

if TYPE_CHECKING:
    from collections.abc import Iterator

    from autogluon.tabular import TabularPredictor
    from sklearn.ensemble import RandomForestRegressor

    from ado.core.discoveryspace.space import DiscoverySpace
    from ado.core.operation.config import FunctionOperationInfo
    from ado.modules.operators.randomwalk import RandomWalkParameters
    from ado.schema.observed_property import ObservedProperty

logger = logging.getLogger(__name__)
_MIN_AUTOGLUON_LABELS = 4


class PredictorExport:
    """Fit a predictor and write its artifacts for the measured candidate pool."""

    def __init__(
        self,
        discovery_space: DiscoverySpace,
        target: ObservedProperty,
        parameters: ActiveLearningParameters,
        directory: Path,
        sampled_identifiers: set[str],
        sampler: type[ForestSampleSelector],
    ) -> None:
        """Refresh the target values and check that the measurement budget was met."""
        # Refresh labels before enumerating the pool to retain stored entity identifiers.
        labels = measured_labels(
            discovery_space, target.identifier, sampler.target_label
        )
        if missing := sampled_identifiers.difference(labels):
            raise ValueError(
                f"Missing finite target measurements for '{target.identifier}': "
                f"{', '.join(sorted(missing))}"
            )
        entities = candidate_entities(discovery_space)
        self.frame, self.features, self.encoder = sampler.encode_feature_matrix(
            discovery_space, entities
        )
        self.frame.index = [entity.identifier for entity in entities]
        self.labels = pd.Series(labels, dtype=float).reindex(self.frame.index)
        self.measured = self.labels.notna().to_numpy()
        measured_count = int(self.measured.sum())
        if (
            parameters.finalPredictor == "autogluon"
            and measured_count < _MIN_AUTOGLUON_LABELS
        ):
            raise ValueError(
                f"AutoGluon requires at least {_MIN_AUTOGLUON_LABELS} target labels "
                f"and collected {measured_count}."
            )
        if measured_count == 0:
            raise ValueError(f"No completed measurements for '{target.identifier}'.")
        if (
            len(sampled_identifiers) < parameters.numberEntities
            and not self.measured.all()
        ):
            raise ValueError(
                "Measurement budget is incomplete: collected "
                f"{len(sampled_identifiers)} of {parameters.numberEntities} new entities "
                "while unmeasured candidates remain."
            )
        self.target_name = target.targetProperty.identifier
        self.parameters = parameters
        self.directory = directory

    def fit_autogluon(
        self,
        *,
        numeric_as_float: bool = False,
        random_state: int | None = None,
        stringify_categories: bool = False,
    ) -> TabularPredictor:
        """Fit and save the regression model on measured rows of the original feature frame.

        Args:
            numeric_as_float: Cast every numeric training column to float.
            random_state: Seed for AutoGluon's learner; None leaves it unseeded.
            stringify_categories: Apply PKH's category conversion to raw predictions.
        """
        from autogluon.tabular import TabularPredictor

        training = self.frame.loc[self.measured].copy()
        training[self.target_name] = self.labels[self.measured]
        if numeric_as_float:
            training = training.astype(
                dict.fromkeys(training.select_dtypes("number"), float)
            )
        model = TabularPredictor(
            label=self.target_name,
            problem_type="regression",
            path=str(self.directory / "autogluon"),
            verbosity=1,
            learner_kwargs={"random_state": random_state}
            if random_state is not None
            else {},
        )
        options = self.parameters.autoGluon
        fit_args = (
            options.model_dump(by_alias=True, exclude_none=True) if options else {}
        )
        categorical_columns = training.select_dtypes(include="category").columns
        if stringify_categories and not categorical_columns.empty:
            from active_learning.pkh.autogluon import PKHCategoricalFeatureGenerator

            training = training.astype(dict.fromkeys(categorical_columns, "string"))
            fit_args["feature_generator"] = PKHCategoricalFeatureGenerator()
        model.fit(train_data=training, **fit_args)
        return model

    def save_forest(self, forest: RandomForestRegressor | None = None) -> Pipeline:
        """Save a fitted forest with its encoder, fitting on all labels when needed."""
        if forest is None:
            forest = build_forest(self.parameters, self.parameters.seed).fit(
                self.features[self.measured], self.labels[self.measured]
            )
        model = Pipeline([("features", self.encoder), ("forest", forest)])
        with (self.directory / "model.pkl").open("wb") as model_file:
            pickle.dump(model, model_file, protocol=5)
        return model

    def save_predictions(
        self,
        model: Pipeline | TabularPredictor,
        metadata_type: type[
            ActiveLearningPredictorMetadata
        ] = ActiveLearningPredictorMetadata,
        **metadata_fields: object,
    ) -> None:
        """Write full-space predictions and the operator's predictor metadata."""
        table = self.frame.assign(entityIdentifier=self.frame.index)
        table[self.target_name] = self.labels
        table["is_predicted"] = ~self.measured
        if (~self.measured).any():
            table.loc[~self.measured, self.target_name] = np.asarray(
                model.predict(self.frame.loc[~self.measured])
            )
        table.to_csv(self.directory / "combined_predictions.csv", index=False)
        metadata = metadata_type(
            backend=self.parameters.finalPredictor,
            targetOutput=self.target_name,
            featureNames=self.frame.columns.tolist(),
            categoricalColumns=self.frame.select_dtypes(
                include="category"
            ).columns.tolist(),
            measuredCount=int(self.measured.sum()),
            createdTimestamp=dt.datetime.now(dt.UTC).isoformat(),
            **metadata_fields,
        )
        (self.directory / "metadata.json").write_text(
            metadata.model_dump_json(indent=2)
        )


def _sampling_parameters(
    parameters: ActiveLearningParameters,
    sampler: type[ForestSampleSelector],
) -> RandomWalkParameters:
    """Configure Random Walk with the operator's sampler and original target name."""
    # Import after plugin registration because Random Walk loads operator entry points.
    from ado.modules.operators.randomwalk import (
        CustomSamplerConfiguration,
        RandomWalkParameters,
        SamplerModuleConf,
    )

    return RandomWalkParameters(
        batchSize=1,
        numberEntities=parameters.numberEntities,
        samplerConfig=CustomSamplerConfiguration(
            module=SamplerModuleConf(
                moduleName=sampler.__module__,
                moduleClass=sampler.__name__,
            ),
            parameters=parameters,
        ),
    )


def _output_after_sampling(
    walk: OperationOutput,
    discovery_space: DiscoverySpace,
    parameters: ActiveLearningParameters,
    target: ObservedProperty,
    final_directory: Path,
    sampler: type[ForestSampleSelector],
) -> tuple[OperationOutput, PredictorExport | None]:
    """Preserve the walk's status and prepare predictor export on success."""
    result = OperationOutput(resources=[walk.operation], exitStatus=walk.exitStatus)
    if (
        not walk.exitStatus
        or walk.exitStatus.exit_state != OperationExitStateEnum.SUCCESS
    ):
        return result, None
    sampled_identifiers = discovery_space.sample_store.entity_identifiers_in_operations(
        walk.operation.identifier
    )
    export = PredictorExport(
        discovery_space,
        target,
        parameters,
        final_directory,
        sampled_identifiers,
        sampler,
    )
    return result, export


@contextlib.contextmanager
def collect_measurements(
    discovery_space: DiscoverySpace,
    operation_info: FunctionOperationInfo | None,
    parameters: ActiveLearningParameters,
    target: ObservedProperty,
    sampler: type[ForestSampleSelector],
) -> Iterator[tuple[OperationOutput, PredictorExport | None]]:
    """Reserve the output directory and collect one entity at a time through Random Walk.

    Yields:
        The operation output and, on success, the measured pool for predictor export.
    """
    from ado.modules.operators.collections import explore

    if parameters.finalPredictor == "autogluon":
        label_count = parameters.numberEntities
        if label_count < _MIN_AUTOGLUON_LABELS:
            label_count += len(
                measured_labels(
                    discovery_space, target.identifier, sampler.target_label
                )
            )
        max_labels = min(
            label_count, getattr(discovery_space.entitySpace, "size", label_count)
        )
        if max_labels < _MIN_AUTOGLUON_LABELS:
            raise ValueError(
                f"AutoGluon requires at least {_MIN_AUTOGLUON_LABELS} target labels "
                f"and can reach at most {max_labels} with this budget and space."
            )
    random_walk = explore.operators["random_walk"].function
    if random_walk is None:
        raise RuntimeError("The random_walk operator is not available.")
    base_directory = Path(parameters.outputDirectory).expanduser().resolve()
    final_directory = base_directory.with_name(f"{base_directory.name}_finalized")
    final_directory.mkdir(parents=True, exist_ok=False)
    try:
        walk = random_walk(
            discovery_space,
            operation_info,
            parameters=_sampling_parameters(parameters, sampler),
        )
        result, export = _output_after_sampling(
            walk, discovery_space, parameters, target, final_directory, sampler
        )
        yield result, export
        if export is not None:
            logger.info("Saved model and predictions to %s", final_directory)
    finally:
        with contextlib.suppress(OSError):
            final_directory.rmdir()
