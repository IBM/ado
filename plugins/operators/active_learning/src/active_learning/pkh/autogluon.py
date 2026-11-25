# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Transient AutoGluon predictors for completing PKH measurements."""

import logging
from math import ceil
from numbers import Integral, Real
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from autogluon.common.features.types import R_OBJECT
from autogluon.core.utils.utils import default_holdout_frac
from autogluon.features.generators import (
    AutoMLPipelineFeatureGenerator,
    IdentityFeatureGenerator,
)
from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.presets_configs import tabular_presets_dict
from joblib import effective_n_jobs

from active_learning.pkh.parameters import (
    ActiveLearningAutoGluonParameters,
    PKHParameters,
)
from ado.schema.domain import VariableTypeEnum
from ado.schema.entityspace import EntitySpaceRepresentation

logger = logging.getLogger(__name__)

# Reserve room for AutoGluon's seed offsets during bagged fitting.
_AUTOGLUON_SEED_MODULUS = 2**31


def validate_knn_prediction(
    parameters: PKHParameters, label_count: int, entity_space: EntitySpaceRepresentation
) -> None:
    """Check an explicit KNN's features and default validation budget."""
    settings = parameters.autoGluon
    if settings is None or not settings.hyperparameters:
        return
    variants = settings.hyperparameters.get("KNN")
    if variants is None:
        return

    has_numeric_feature = False
    for prop in entity_space.constitutiveProperties:
        domain = prop.propertyDomain
        if domain.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE or (
            domain.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE
            and all(
                isinstance(value, (Real, np.bool_)) for value in domain.values or ()
            )
        ):
            has_numeric_feature = True
            break

    bagged = tabular_presets_dict[settings.presets]["auto_stack"]
    if bagged:
        # AutoGluon KNN uses leave-one-out validation by default when bagged.
        available = label_count - 1
    else:
        holdout = ceil(label_count * round(default_holdout_frac(label_count), 4))
        # AutoGluon's learner reserves at least two validation rows.
        available = label_count - max(2, holdout)

    for variant in variants if isinstance(variants, list) else [variants]:
        if not has_numeric_feature and "valid_raw_types" not in (
            variant.get("ag_args_fit") or {}
        ):
            raise ValueError("AutoGluon KNN requires at least one numeric feature.")

        neighbors = variant.get("n_neighbors", 5)
        if (
            not isinstance(neighbors, Integral)
            or isinstance(neighbors, bool)
            or neighbors < 1
        ):
            raise ValueError("AutoGluon KNN n_neighbors must be a positive integer.")

        ensemble = variant.get("ag_args_ensemble") or {}
        if bagged and {"custom_splits", "use_child_oof", "num_folds"} & ensemble.keys():
            continue
        if neighbors > available:
            raise ValueError(
                f"AutoGluon KNN n_neighbors={neighbors} exceeds {available} available "
                "training rows. Reduce n_neighbors or increase the measurement budget."
            )


def _minimum_training_rows(
    predictor: TabularPredictor, models: list[str]
) -> int | None:
    """Return the smallest reported fit sample count among contributing regressors."""
    counts = []
    for name in models:
        info = predictor.model_info(name)
        # A bag wrapper records the entire pool even when it retains a single fold.
        components = (
            info["children_info"].values() if "children_info" in info else [info]
        )
        for component in components:
            count = component.get("num_samples")
            if count is None:
                return None
            counts.append(count)
    return min(counts, default=None)


def predict_unmeasured(
    frame: pd.DataFrame,
    labels: pd.Series,
    target_name: str,
    parameters: PKHParameters,
) -> tuple[np.ndarray, dict[str, str]]:
    """Predict missing target values and return the selected model's metadata labels.

    Compare the individual models and any weighted ensemble by validation score.
    Let AutoGluon finalize the winner and its contributing base models.
    """
    measured = labels.notna()
    training = frame.loc[measured].copy()
    training[target_name] = labels.loc[measured]
    settings = parameters.autoGluon or ActiveLearningAutoGluonParameters()
    hyperparameters = parameters.prediction_hyperparameters()
    # Omitting hyperparameters lets the AutoGluon preset choose the candidate models.
    fit_options = {}
    expected_models = None
    if hyperparameters is not None:
        fit_options["hyperparameters"] = hyperparameters
        expected_models = sum(
            len(variants) if isinstance(variants, list) else 1
            for variants in hyperparameters.values()
        )
    weighted = parameters.finalPredictor == "autogluon" and expected_models != 1
    cpus = effective_n_jobs(parameters.nJobs)
    with TemporaryDirectory(prefix="ado-pkh-") as directory:
        predictor = TabularPredictor(
            label=target_name,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            path=str(Path(directory) / "model"),
            learner_kwargs={"random_state": parameters.seed % _AUTOGLUON_SEED_MODULUS}
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
            time_limit=settings.timeLimit,
            num_cpus=cpus,
            num_stack_levels=0,
            dynamic_stacking=False,
            fit_weighted_ensemble=weighted,
            raise_on_model_failure=hyperparameters is not None,
            refit_full=False,
            set_best_to_refit_full=False,
            **fit_options,
        )
        leaderboard = predictor.leaderboard()
        base_models = leaderboard.loc[leaderboard["stack_level"] == 1, "model"].tolist()
        if expected_models is not None:
            if len(base_models) != expected_models:
                raise ValueError(
                    f"AutoGluon fitted {len(base_models)} of {expected_models} "
                    f"requested models: {base_models}. Check the model settings "
                    "and autoGluon.timeLimit."
                )
            if weighted and not leaderboard["stack_level"].eq(2).any():
                raise ValueError(
                    "AutoGluon could not fit the requested level-two weighted ensemble."
                )
        # AutoGluon reports all scores as higher-is-better. Break ties by stack
        # level and model name so elapsed time cannot change the predictions.
        ranked = leaderboard.dropna(subset=["score_val"]).sort_values(
            ["score_val", "stack_level", "model"], ascending=[False, True, True]
        )
        if ranked.empty:
            raise ValueError(
                "AutoGluon returned no validation scores for PKH prediction."
            )
        model = ranked.iloc[0]["model"]
        refits = predictor.refit_full(
            model=model, num_cpus=cpus, set_best_to_refit_full=False
        )
        full_model = refits[model]
        metadata = {"predictor": full_model}
        contributing_models = [refits[name] for name in base_models if name in refits]
        rows = _minimum_training_rows(predictor, contributing_models)
        if rows is not None:
            metadata["predictorTrainingRows"] = str(rows)
            if rows < len(training):
                logger.warning(
                    "A base regressor in %s was fitted on %d of %d measured rows.",
                    full_model,
                    rows,
                    len(training),
                )
        predictions = predictor.predict(frame.loc[~measured], model=full_model)
        return predictions.to_numpy(dtype=float), metadata


class PKHCategoricalFeatureGenerator(AutoMLPipelineFeatureGenerator):
    """Transform raw categories when loading predictors saved by earlier PKH runs."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert raw category values to their training strings before prediction."""
        columns = self.feature_metadata_in.get_features(valid_raw_types=[R_OBJECT])
        frame = X.copy()
        values = frame[columns].astype(object)
        frame[columns] = values.where(values.notna(), None).map(str)
        return super().transform(frame)
