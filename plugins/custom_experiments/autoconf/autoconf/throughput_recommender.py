# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Recommend a feasible fine-tuning configuration with maximum throughput."""

import functools
import itertools
import json
import math
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from ado.modules.actuators.custom_experiments import custom_experiment
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.property import ConstitutiveProperty
from autoconf.min_gpu_recommender import (
    GPUModel,
    TokensPerSample,
    TuningMethod,
    load_model,
)
from autoconf.model_paths import DEFAULT_MODEL_ROOT, MODEL_VERSION
from autoconf.utils.autoconf_build.ml_classifier import (
    DEFAULT_DATA_ROOT_DIR,
    DEFAULT_FILE_NAME,
    DEFAULT_HF_FILENAME,
    DEFAULT_HF_REPO_ID,
    ensure_dataset,
    prepare_training_data,
)
from autoconf.utils.config_mapper import map_valid_model_name

THROUGHPUT_MODEL_VERSION = "4.1.0"
REGRESSOR_DIRECTORY = "v4-1-0-regressor"
KNOWN_MODELS_FILE = "known_model_names.json"
TARGET = "dataset_tokens_per_second"
FEATURE_COLUMNS = [
    "model_name",
    "method",
    "number_gpus",
    "gpu_model",
    "tokens_per_sample",
    "batch_size",
]
REGRESSION_COLUMNS = [*FEATURE_COLUMNS, TARGET]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 128, 256]
GPU_COUNTS = [1, 2, 4, 8, 16]

ModelName = ConstitutiveProperty(
    identifier="model_name",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE
    ),
)
MaxNodes = ConstitutiveProperty(
    identifier="max_nodes",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[1, 1025],
        interval=1,
    ),
)
GPUsPerNode = ConstitutiveProperty(
    identifier="gpus_per_node",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[1, 1025],
        interval=1,
    ),
)
ModelVersion = ConstitutiveProperty(
    identifier="model_version",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=[THROUGHPUT_MODEL_VERSION],
    ),
)


def regressor_path(model_root: Path | None = None) -> Path:
    """Return the independent, versioned path for the throughput regressor."""
    return (model_root or DEFAULT_MODEL_ROOT) / REGRESSOR_DIRECTORY


def candidate_combinations(max_nodes: int, gpus_per_node: int) -> list[tuple[int, int]]:
    """Produce ordered, feasible (effective batch size, total GPUs) pairs."""
    if max_nodes < 1 or gpus_per_node < 1:
        raise ValueError("max_nodes and gpus_per_node must be positive integers")

    max_gpus = min(GPU_COUNTS[-1], max_nodes * gpus_per_node)
    max_effective_batch_size = min(
        BATCH_SIZES[-1] * GPU_COUNTS[-1], BATCH_SIZES[-1] * max_nodes * gpus_per_node
    )
    effective_batch_sizes = []
    current = BATCH_SIZES[0] * GPU_COUNTS[0]
    while current <= max_effective_batch_size:
        effective_batch_sizes.append(current)
        current *= 2

    return [
        (batch_size, number_gpus)
        for batch_size, number_gpus in itertools.product(
            effective_batch_sizes, GPU_COUNTS
        )
        if number_gpus <= max_gpus
        and (number_gpus <= gpus_per_node or number_gpus % gpus_per_node == 0)
        and batch_size % number_gpus == 0
        and batch_size // number_gpus <= BATCH_SIZES[-1]
    ]


def prepare_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep successful, divisible measurements with positive throughput."""
    missing = sorted({*REGRESSION_COLUMNS, "is_valid"}.difference(df.columns))
    if missing:
        raise ValueError(f"Regression dataset is missing required columns: {missing}")

    numeric = df.copy()
    numeric["number_gpus"] = pd.to_numeric(numeric["number_gpus"], errors="coerce")
    numeric["batch_size"] = pd.to_numeric(numeric["batch_size"], errors="coerce")
    numeric[TARGET] = pd.to_numeric(numeric[TARGET], errors="coerce")
    mask = (
        (numeric["number_gpus"] > 0)
        & (numeric["batch_size"] % numeric["number_gpus"] == 0)
        & (numeric["is_valid"] == 1)
        & np.isfinite(numeric[TARGET])
        & (numeric[TARGET] > 0)
    )
    prepared = numeric.loc[mask, REGRESSION_COLUMNS].drop_duplicates().copy()
    if prepared.empty:
        raise ValueError(
            "Regression dataset has no valid positive throughput measurements"
        )
    return prepared


def build_regressor(
    model_root: Path = DEFAULT_MODEL_ROOT,
    data_root_dir: Path = DEFAULT_DATA_ROOT_DIR,
    file_name: str = DEFAULT_FILE_NAME,
    repo_id: str = DEFAULT_HF_REPO_ID,
    filename: str = DEFAULT_HF_FILENAME,
    fit_options: dict[str, Any] | None = None,
) -> Path:
    """Train and save the throughput regressor from the classifier's dataset."""
    destination = regressor_path(model_root)
    if destination.exists():
        raise FileExistsError(f"Regressor already exists at {destination}")

    data_path = ensure_dataset(repo_id, filename, data_root_dir / file_name)
    training_data = prepare_regression_data(
        prepare_training_data(pd.read_csv(data_path))
    )
    known_models = sorted(training_data["model_name"].dropna().unique().tolist())
    model_root.mkdir(parents=True, exist_ok=True)
    options = fit_options or {
        "presets": "good",
        "excluded_model_types": ["GBM"],
        "time_limit": 1800,
        "num_bag_folds": 5,
    }
    with tempfile.TemporaryDirectory(
        dir=model_root, prefix="autoconf-regression-training-"
    ) as temporary_directory:
        predictor = TabularPredictor(
            label=TARGET,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            path=str(Path(temporary_directory) / "model"),
        ).fit(train_data=TabularDataset(training_data), **options)
        predictor.clone_for_deployment(path=str(destination))
    (destination / KNOWN_MODELS_FILE).write_text(
        json.dumps(known_models), encoding="utf-8"
    )
    return destination


@functools.cache
def load_regressor(
    model_version: str = THROUGHPUT_MODEL_VERSION,
    model_root: Path | None = None,
) -> tuple[TabularPredictor, frozenset[str]]:
    """Load a cached regressor, training it on first use if needed."""
    if model_version != THROUGHPUT_MODEL_VERSION:
        raise ValueError(f"Unknown throughput model_version: {model_version}")

    path = regressor_path(model_root)
    if not path.is_dir():
        warnings.warn(
            "AutoConf throughput regressor not found. Training it from the default "
            "Hugging Face dataset; this may take several minutes.",
            stacklevel=2,
        )
        build_regressor(model_root=model_root or DEFAULT_MODEL_ROOT)
    known_models = frozenset(
        json.loads((path / KNOWN_MODELS_FILE).read_text(encoding="utf-8"))
    )
    return TabularPredictor.load(
        str(path), require_py_version_match=False
    ), known_models


def warn_if_unknown_model(model_name: str, known_models: frozenset[str]) -> None:
    """Warn that an unobserved model has no validated throughput accuracy."""
    if model_name not in known_models:
        warnings.warn(
            f"Model {model_name!r} was absent from AutoConf training data; its "
            "throughput estimate is unvalidated. Use a model represented in "
            "the dataset, or add measurements and retrain the models.",
            UserWarning,
            stacklevel=2,
        )


def select_best_candidate(
    candidates: pd.DataFrame,
    valid_predictions: pd.Series,
    throughput_predictions: pd.Series,
    gpus_per_node: int = 8,
) -> dict[str, bool | int | float]:
    """Select the first valid candidate with maximum predicted throughput."""
    valid_mask = np.asarray(valid_predictions) == 1
    valid_candidates = candidates.loc[valid_mask].copy()
    if valid_candidates.empty:
        return {"can_recommend": False}

    valid_candidates["estimated_throughput"] = np.asarray(throughput_predictions)
    valid_candidates = valid_candidates.loc[
        np.isfinite(valid_candidates["estimated_throughput"])
    ]
    if valid_candidates.empty:
        return {"can_recommend": False}

    best = valid_candidates.loc[valid_candidates["estimated_throughput"].idxmax()]
    batch_size = int(best["batch_size"])
    number_gpus = int(best["number_gpus"])
    workers = math.ceil(number_gpus / gpus_per_node)
    return {
        "can_recommend": True,
        "gpus": number_gpus // workers,
        "workers": workers,
        "effective_batch_size": batch_size,
        "per_device_batch_size": batch_size // number_gpus,
        "estimated_throughput": float(best["estimated_throughput"]),
    }


def recommend_throughput(
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    max_nodes: int,
    gpus_per_node: int,
    classifier: TabularPredictor,
    regressor: TabularPredictor,
    known_models: frozenset[str],
) -> dict[str, bool | int | float]:
    """Classify all candidates and regress throughput for feasible ones."""
    mapped_model_name = map_valid_model_name(model_name)
    warn_if_unknown_model(mapped_model_name, known_models)
    combos = candidate_combinations(max_nodes, gpus_per_node)
    candidates = pd.DataFrame(
        [
            {
                "model_name": mapped_model_name,
                "method": method,
                "number_gpus": number_gpus,
                "gpu_model": gpu_model,
                "tokens_per_sample": tokens_per_sample,
                "batch_size": batch_size,
            }
            for batch_size, number_gpus in combos
        ],
        columns=FEATURE_COLUMNS,
    )
    valid_predictions = classifier.predict(candidates)
    valid_mask = np.asarray(valid_predictions) == 1
    throughput_predictions = (
        regressor.predict(candidates.loc[valid_mask])
        if valid_mask.any()
        else pd.Series(dtype=float)
    )
    return select_best_candidate(
        candidates, valid_predictions, throughput_predictions, gpus_per_node
    )


@custom_experiment(
    required_properties=[ModelName, TuningMethod, GPUModel, TokensPerSample],
    optional_properties=[MaxNodes, GPUsPerNode, ModelVersion],
    output_property_identifiers=[
        "can_recommend",
        "gpus",
        "workers",
        "effective_batch_size",
        "per_device_batch_size",
        "estimated_throughput",
    ],
    metadata={
        "description": "Recommend the feasible batch size and GPU configuration "
        "with the highest predicted fine-tuning throughput."
    },
    parameterization={},
)
def throughput_recommender(
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    max_nodes: int = 1,
    gpus_per_node: int = 8,
    model_version: str = THROUGHPUT_MODEL_VERSION,
) -> dict[str, bool | int | float]:
    """Run the ADO custom experiment using cached AutoConf predictors."""
    if model_version != THROUGHPUT_MODEL_VERSION:
        raise ValueError(f"Unknown throughput model_version: {model_version}")
    classifier = load_model(MODEL_VERSION)
    regressor, known_models = load_regressor(model_version)
    return recommend_throughput(
        model_name,
        method,
        gpu_model,
        tokens_per_sample,
        max_nodes,
        gpus_per_node,
        classifier,
        regressor,
        known_models,
    )
