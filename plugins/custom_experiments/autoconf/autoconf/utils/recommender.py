# Copyright (c) IBM Corporation

# SPDX-License-Identifier: MIT

import logging

import pandas as pd
from autogluon.tabular import TabularPredictor

from autoconf.utils.pydantic_models import JobConfig
from autoconf.utils.rule_based_classifier import is_row_valid

# Configure logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[logging.StreamHandler()],  # Output to console
)

logger = logging.getLogger(__name__)

# NOTE: this list will not be used if the user provides them

VALID_N_GPUS = [1, 2, 4, 8, 16, 32]


def get_model_prediction_and_metadata(
    config: pd.DataFrame | dict | JobConfig, predictor: TabularPredictor
) -> tuple[int, dict]:
    """Gets valid/invalid prediction and reason why"""
    if isinstance(config, dict):
        config = pd.DataFrame(config, index=[0])
    if isinstance(config, JobConfig):
        config = pd.DataFrame([config.model_dump()], index=[0])

    metadata = {}
    pred = None
    machine_learning_classifier_error = None
    row_is_valid, rule_based_classifier_error = is_row_valid(config)
    if int(row_is_valid) == 1:
        try:
            pred = predictor.predict(config).values[0]
            logger.debug("Prediction succeeded")
        except Exception as e:
            logger.debug("Prediction FAILED")
            machine_learning_classifier_error = str(e)

    metadata["Rule-Based Classifier error"] = " ".join(rule_based_classifier_error)
    metadata["Predictive Model Classifier error"] = machine_learning_classifier_error

    # NOTE:  is added here to account for invalid by rule based classifier
    pred = int(pred) if pred else 0
    return pred, metadata


def recommend_min_gpu(
    job_config: JobConfig,
    predictor: str | TabularPredictor,
    valid_n_gpu_list: list[int] | None = None,
) -> tuple[int, dict]:
    """Recommends the minimum number of GPUs required for a SFT job defined by the fields of the pydantic model :job_config:
    Returns
        min_n_gpu: the minimum number of valid gpus
        -1 if no gpu number in the valid_n_gpu list is predicted to be valid"""
    # We use list() here to copy VALID_N_GPUS and thus avoid any unintentional update to VALID_N_GPUS.
    if valid_n_gpu_list is None:
        valid_n_gpu_list = list(VALID_N_GPUS)

    if isinstance(predictor, str):
        predictor = TabularPredictor.load(predictor, require_py_version_match=False)

    metadata = {"default": "User config was not provided"}

    # VV: Find the minimum number of GPUs that the recommender predicts will successfully run this tuning job.
    # Start from the lowest candidate and stop when the recommender predicts that the workload will successfully
    # run with that GPU count.

    min_number_gpus = -1

    valid_n_gpu_list = sorted(valid_n_gpu_list)
    for candidate_number_gpus in valid_n_gpu_list:
        logger.info(f"Testing number_gpus={candidate_number_gpus}")
        if job_config.number_gpus and candidate_number_gpus == job_config.number_gpus:
            logger.info(
                "This is the value provided by the user, for this configuration the recommender will provide additional metadata"
            )
            gpus_can_support_run, _ = get_model_prediction_and_metadata(
                job_config, predictor=predictor
            )
        else:
            new_job_config = job_config.model_copy(
                update={"number_gpus": candidate_number_gpus}
            )
            gpus_can_support_run, metadata = get_model_prediction_and_metadata(
                new_job_config, predictor=predictor
            )

        logger.info(
            f"Prediction for ngpu={candidate_number_gpus}\t:\t{gpus_can_support_run}\t(note:0 is not valid, 1 is Valid)"
        )

        if gpus_can_support_run == 1:
            min_number_gpus = candidate_number_gpus
            break

    logger.info(f"""Metadata related to the model prediction
        (number_gpus={job_config.number_gpus if job_config.number_gpus else 'Not provided'})
        :{metadata}""")

    if min_number_gpus == -1:
        logger.info(f"""A recommendation for 'number_gpus' cannot be provided because
            no values for 'number_gpus' of the list {valid_n_gpu_list} would result
            in a valid run according to the predictive model.""")
    else:
        logger.info(f"The recommended number_gpus={min_number_gpus}.")

    return min_number_gpus, metadata


class MinGpuRecommender:
    def __init__(
        self, predictor: str | TabularPredictor, valid_n_gpu: list[int] | None = None
    ) -> None:
        self.valid_n_gpu = valid_n_gpu or list(VALID_N_GPUS)

        if isinstance(predictor, str):
            self.predictor = TabularPredictor.load(
                predictor, require_py_version_match=False
            )
        else:
            self.predictor = predictor

    def recommend_min_gpu(self, job_config: JobConfig) -> tuple[int, dict]:
        return recommend_min_gpu(job_config, self.predictor, self.valid_n_gpu)

    def predict(self, job_config: JobConfig | pd.DataFrame) -> list[int]:
        if isinstance(job_config, pd.DataFrame):
            # Convert DataFrame rows to JobConfig instances
            job_configs = [
                JobConfig.model_validate(row.dropna().to_dict())
                for _, row in job_config.iterrows()
            ]
        else:
            job_configs = [job_config]

        # Run prediction for each config
        return [
            recommend_min_gpu(config, self.predictor, self.valid_n_gpu)[0]
            for config in job_configs
        ]


class NoRecommendationError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __str__(self) -> str:
        return f"Unable to recommend minimum number of GPUs to avoid GPU OOM: {self.reason}"
