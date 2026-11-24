# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Transient AutoGluon tree predictors for completing PKH measurements."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from autogluon.common.features.types import R_OBJECT
from autogluon.features.generators import (
    AutoMLPipelineFeatureGenerator,
    IdentityFeatureGenerator,
)
from autogluon.tabular import TabularPredictor
from joblib import effective_n_jobs

from active_learning.pkh.parameters import (
    ActiveLearningAutoGluonParameters,
    PKHParameters,
)


def predict_unmeasured(
    frame: pd.DataFrame,
    labels: pd.Series,
    target_name: str,
    parameters: PKHParameters,
) -> tuple[np.ndarray, str]:
    """Fit temporary tree models and predict the rows with missing target values.

    A level-two ensemble learns weights on held-out predictions. Refit its selected
    base models on all labels before predicting the unmeasured rows.
    """
    measured = labels.notna()
    training = frame.loc[measured].copy()
    training[target_name] = labels.loc[measured]
    settings = parameters.autoGluon or ActiveLearningAutoGluonParameters()
    trees = parameters.prediction_hyperparameters()
    expected_models = sum(
        len(options) if isinstance(options, list) else 1 for options in trees.values()
    )
    weighted = parameters.finalPredictor == "weighted_ensemble" or (
        parameters.finalPredictor == "autogluon" and expected_models > 1
    )
    cpus = effective_n_jobs(parameters.nJobs)
    with TemporaryDirectory(prefix="ado-pkh-") as directory:
        predictor = TabularPredictor(
            label=target_name,
            problem_type="regression",
            path=str(Path(directory) / "model"),
            learner_kwargs={"random_state": parameters.seed}
            if parameters.seed is not None
            else {},
            verbosity=1,
        )
        predictor.fit(
            train_data=training,
            # PKH already supplies numeric and categorical features, including
            # categories that occur only once in a small measurement budget.
            feature_generator=IdentityFeatureGenerator(),
            presets=settings.presets,
            hyperparameters=trees,
            time_limit=settings.timeLimit,
            num_cpus=cpus,
            num_stack_levels=0,
            dynamic_stacking=False,
            fit_weighted_ensemble=weighted,
            raise_on_model_failure=True,
            refit_full=False,
            set_best_to_refit_full=False,
        )
        fitted = predictor.model_names(level=1)
        if len(fitted) != expected_models:
            raise ValueError(
                f"AutoGluon fitted {len(fitted)} of {expected_models} requested tree models: "
                f"{fitted}. Check the tree settings and autoGluon.timeLimit."
            )
        model = predictor.model_best
        if weighted:
            level_two = predictor.model_names(level=2)
            if len(level_two) != 1:
                raise ValueError(
                    "AutoGluon could not fit the requested level-two weighted ensemble."
                )
            model = level_two[0]
        full_model = predictor.refit_full(model=model, num_cpus=cpus)[model]
        predictions = predictor.predict(frame.loc[~measured], model=full_model)
        return predictions.to_numpy(dtype=float), full_model


class PKHCategoricalFeatureGenerator(AutoMLPipelineFeatureGenerator):
    """Transform raw categories when loading predictors saved by earlier PKH runs."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert raw category values to their training strings before prediction."""
        columns = self.feature_metadata_in.get_features(valid_raw_types=[R_OBJECT])
        frame = X.copy()
        values = frame[columns].astype(object)
        frame[columns] = values.where(values.notna(), None).map(str)
        return super().transform(frame)
