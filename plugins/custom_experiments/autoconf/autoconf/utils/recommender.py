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

# NOTE: this list will not be used if the user provides one
VALID_N_GPUS = [1, 2, 4, 8, 16, 32]


def get_model_prediction_and_metadata(
    config: pd.DataFrame | dict | JobConfig, predictor
) -> tuple[int, dict]:
    """Gets valid/invalid prediction and reason why"""
    if isinstance(config, dict):
        config = pd.DataFrame(config, index=[0])
    if isinstance(config, JobConfig):
        config = pd.DataFrame([config.model_dump()], index=[0])

    metadata = {}
    pred = None
    C_err = None
    b, RBC_err = is_row_valid(config)
    if int(b) == 1:
        try:
            pred = predictor.predict(config).values[0]
            logger.debug("Prediction succeeded")
        except Exception as e:
            logger.debug("Prediction FAILED")
            C_err = str(e)

    metadata["Rule-Based Classifier error"] = " ".join(RBC_err)
    metadata["Predictive Model Classifier error"] = C_err

    # NOTE:  is added here to account for invalid by rule based classifier
    pred = int(pred) if pred else 0
    return pred, metadata


def recommend_min_gpu(
    job_config: JobConfig, predictor, valid_n_gpu_list: list[int] | None = None
) -> tuple[int, dict]:
    """Recommends the minimum number of GPUs required for a SFT job defined by the fields of the pydantic model :job_config:
    Returns
        min_n_gpu: the minimum number of valid gpus
        -1 if no gpu number in the valid_n_gpu list is predicted to be valid"""
    if valid_n_gpu_list is None:
        valid_n_gpu_list = list(VALID_N_GPUS)

    res_dict = {}
    if isinstance(predictor, str):
        predictor = TabularPredictor.load(predictor, require_py_version_match=False)

    metadata = {"default": "User config was not provided"}
    for n in valid_n_gpu_list:
        logger.info(f"Testing number_gpus={n}")
        if job_config.number_gpus and n == job_config.number_gpus:
            logger.info(
                "This is the value provided by the user, for this configuration the recommender will provide additional metadata"
            )
            p, m = get_model_prediction_and_metadata(job_config, predictor=predictor)
            res_dict[n] = p
            metadata = m
        else:
            new_job_config = job_config.model_copy(update={"number_gpus": n})
            p, m = get_model_prediction_and_metadata(
                new_job_config, predictor=predictor
            )
            res_dict[n] = p
            metadata = m

        logger.info(
            f"Prediction for ngpu={n}\t:\t{p}\t(note:0 is not valid, 1 is Valid)"
        )

    logger.info(
        f"""Metadata related to the model prediction
        (number_gpus={job_config.number_gpus if job_config.number_gpus else 'Not provided'})
        :{metadata}"""
    )

    min_key = min((k for k, v in res_dict.items() if int(v) == 1), default=-1)
    if min_key == -1:
        logger.info(
            f"""A recommendation for 'number_gpus' cannot be provided because
            no values for 'number_gpus' of the list {valid_n_gpu_list} would result
            in a valid run according to the predictive model."""
        )
    else:
        logger.info(f"The recommended number_gpus={min_key}.")

    return min_key, metadata


def validate_as_jobconfig(config_to_test):
    from pydantic import ValidationError

    try:
        job = JobConfig(**config_to_test)
        print("Validation successful:", job)
    except ValidationError as e:
        print("Validation error:", e)
    return job


class MinGpuRecommender:
    def __init__(self, predictor, valid_n_gpu: list[int] | None = None):
        if valid_n_gpu is None:
            valid_n_gpu = VALID_N_GPUS

        self.valid_n_gpu = valid_n_gpu

        if isinstance(predictor, str):
            self.predictor = TabularPredictor.load(
                predictor, require_py_version_match=False
            )
        else:
            self.predictor = predictor

    def recommend_min_gpu(self, job_config):
        return recommend_min_gpu(job_config, self.predictor, self.valid_n_gpu)

    def fit(self, X=None, y=None):
        # No fitting needed, but included for compatibility
        return self

    def predict(self, job_config: JobConfig | pd.DataFrame):
        if isinstance(job_config, pd.DataFrame):
            # Convert DataFrame rows to JobConfig instances
            job_configs = [
                JobConfig(**row.dropna().to_dict()) for _, row in job_config.iterrows()
            ]
        else:
            job_configs = [job_config]

        # Run prediction for each config
        return [
            recommend_min_gpu(config, self.predictor, self.valid_n_gpu)[0]
            for config in job_configs
        ]


class NoRecommendationError(ValueError):
    def __init__(self, reason: str):
        self.reason = reason

    def __str__(self):
        return f"Unable to recommend minimum number of GPUs to avoid GPU OOM: {self.reason}"
