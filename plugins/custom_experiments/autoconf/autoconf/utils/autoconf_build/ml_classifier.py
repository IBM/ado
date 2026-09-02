# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

import argparse
import logging
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import confusion_matrix

from autoconf.model_paths import (
    DEFAULT_MODEL_ROOT,
    MODEL_VERSION,
    model_path,
)
from autoconf.utils.rule_based_classifier import is_row_valid

logger = logging.getLogger(__name__)

# Default values
DEFAULT_DATA_ROOT_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_FILE_NAME = "ado-sfttrainer-dataset.csv"
DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/ibm-research/"
    "LLMFineTuningBench/resolve/main/ado-sfttrainer-dataset.csv"
)
DEFAULT_REFIT = False
DEFAULT_TRAIN_FRACTION = 1.0
DEFAULT_PRESET_QUALITY = "medium_quality"
# Constants
COLS_TO_USE = [
    "model_name",
    "method",  # LoRA, FULL
    "number_gpus",
    "gpu_model",
    "tokens_per_sample",  # this is: max_sequence_lenght
    "batch_size",
    "is_valid",  # Has the job being successful or did it have OOM problems?
    # NOTE: jobs that are not successful for incorrect specification of the config file are filtered out before training the model.
]

TARGET = "is_valid"


def ensure_dataset(dataset_url: str, data_path: Path) -> Path:
    """Download the Hugging Face dataset when it is not already available.

    Args:
        dataset_url: URL of the source CSV on Hugging Face.
        data_path: Local destination for the CSV.

    Returns:
        The local dataset path.

    Raises:
        ValueError: If the dataset URL uses an unsupported scheme.
    """
    if data_path.is_file():
        logger.info("Using existing dataset at %s", data_path)
        return data_path

    scheme = urllib.parse.urlparse(dataset_url).scheme
    if scheme not in {"file", "http", "https"}:
        raise ValueError(f"Unsupported dataset URL scheme: {scheme}")

    data_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = data_path.with_suffix(f"{data_path.suffix}.part")
    logger.info("Downloading dataset from %s to %s", dataset_url, data_path)
    try:
        urllib.request.urlretrieve(dataset_url, partial_path)  # noqa: S310
        partial_path.replace(data_path)
    finally:
        partial_path.unlink(missing_ok=True)

    return data_path


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Hugging Face dataset for classifier training.

    The public subset does not carry an explicit ``is_valid`` label. The complete
    dataset uses the same schema, so a missing label is derived from the presence
    of ``train_runtime``.

    Args:
        df: Raw Hugging Face dataset.

    Returns:
        A copy of the data containing the binary training target.

    Raises:
        ValueError: If columns needed to derive the target are missing.
    """
    prepared = df.copy()
    if TARGET not in prepared:
        if "train_runtime" not in prepared:
            raise ValueError(
                "Dataset must contain either 'is_valid' or 'train_runtime'"
            )
        prepared[TARGET] = prepared["train_runtime"].notna().astype(int)

    return prepared


def validate_training_data(
    df: pd.DataFrame, *, require_both_target_classes: bool
) -> None:
    """Validate the dataset schema and classifier target distribution.

    Args:
        df: Prepared classifier data.
        require_both_target_classes: Whether both binary target classes are
            required. Schema-only validation of the current public subset sets
            this to ``False``.

    Raises:
        ValueError: If required columns or target classes are missing.
    """
    missing_columns = sorted(set(COLS_TO_USE).difference(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    target_values = set(df[TARGET].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError("The is_valid target must contain only 0 and 1")
    if require_both_target_classes and target_values != {0, 1}:
        raise ValueError(
            "Classifier training requires both successful and failed measurements"
        )


def filter_valid_with_hard_logic(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Length of the DataFrame before filtering: {len(df)}")
    valid_indices = [i for i, config in df.iterrows() if is_row_valid(config)[0]]
    df_filtered = df.loc[valid_indices].copy()
    logger.info(f"Length of the DataFrame after filtering {len(df_filtered)}")
    return df_filtered


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AutoGluon ML classifier for autoconf",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Full path to the CSV data file. If not provided, uses data-root-dir and file-name.",
    )

    parser.add_argument(
        "--data-root-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT_DIR,
        help="Root directory containing the data files",
    )

    parser.add_argument(
        "--file-name",
        type=str,
        default=DEFAULT_FILE_NAME,
        help="Name of the CSV data file",
    )

    parser.add_argument(
        "--dataset-url",
        default=DEFAULT_DATASET_URL,
        help="URL of the source CSV on Hugging Face",
    )

    parser.add_argument(
        "--model-root-dir",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Root directory for locally generated models",
    )

    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="Download and validate the dataset without fitting a model",
    )

    parser.add_argument(
        "--refit",
        action="store_true",
        default=DEFAULT_REFIT,
        help="Whether to refit the model (improves inference speed but may diminish accuracy)",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Fraction of data to use for training (0.0 to 1.0)",
    )

    parser.add_argument(
        "--preset-quality",
        type=str,
        default=DEFAULT_PRESET_QUALITY,
        choices=[
            "best_quality",
            "high_quality",
            "good_quality",
            "medium_quality",
            "optimize_for_deployment",
        ],
        help="AutoGluon preset quality level",
    )

    return parser.parse_args(arguments)


# TRAINING FUNCTION
def fit_tabular_predictor(
    df: pd.DataFrame,
    train_fraction: float,
    preset_quality: str,
    output_path: Path,
    cols_to_use: list[str] = COLS_TO_USE,
) -> tuple[TabularPredictor, pd.DataFrame, pd.DataFrame, float]:
    """Fit an AutoGluon predictor in a temporary output directory.

    Args:
        df: Prepared and filtered training data.
        train_fraction: Fraction of rows assigned to training.
        preset_quality: AutoGluon preset used for fitting.
        output_path: Temporary predictor output path.
        cols_to_use: Feature and target columns used for fitting.

    Returns:
        The predictor, training data, test data, and elapsed training time.
    """
    train_idx = int(len(df) * train_fraction)
    df_train = df.iloc[:train_idx][cols_to_use]
    df_test = df.iloc[train_idx:][cols_to_use]
    df_test = filter_valid_with_hard_logic(df_test)
    fit_params = {"presets": [preset_quality], "excluded_model_types": "GBM"}
    train_data = TabularDataset(df_train)
    train_data.head()
    start = time.time()
    predictor = TabularPredictor(label=TARGET, path=output_path).fit(
        train_data, **fit_params
    )
    elapsed_time = time.time() - start
    return predictor, df_train, df_test, elapsed_time


# TEST
def log_metrics(
    predictor: TabularPredictor,
    df_test: pd.DataFrame,
    df_train: pd.DataFrame,
    train_fraction: float,
) -> dict[str, Any]:
    if not df_test.empty:
        test_data = TabularDataset(df_test)
        metrics_dict = predictor.evaluate(test_data, silent=True)
        logger.info(f"The model performance on the test data is: {metrics_dict}")
        # Print confusion matrix on test data
        y_true = df_test[TARGET]
        y_pred = predictor.predict(test_data)
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion Matrix on test data:\n{cm}")
        logger.info("Confusion Matrix format:\n [[TN, FP],\n [FN, TP]]")
    else:
        train_data = TabularDataset(df_train)
        metrics_dict = predictor.evaluate(train_data, silent=True)
        logger.info(f"The test df was empty, train fraction = {train_fraction}.")
        logger.info(f"The model performance on the training data is: {metrics_dict}")
    return metrics_dict


def main() -> None:
    """Main execution function."""
    logging.basicConfig(level=logging.INFO)
    # Parse command line arguments
    args = parse_arguments()

    # Determine data path
    path = args.data_path or args.data_root_dir / args.file_name
    path = ensure_dataset(args.dataset_url, path)

    logger.info(f"Using data path: {path}")
    logger.info(f"REFIT: {args.refit}")
    logger.info(f"TRAIN_FRACTION: {args.train_fraction}")
    logger.info(f"PRESET_QUALITY: {args.preset_quality}")

    # Load and process data
    df_original = pd.read_csv(path)
    df_original = prepare_training_data(df_original)
    validate_training_data(
        df_original, require_both_target_classes=not args.validate_data_only
    )
    logger.info(f"Models supported are: {set(df_original['model_name'].values)}")
    logger.info("Target distribution: %s", df_original[TARGET].value_counts().to_dict())

    if args.validate_data_only:
        logger.info("Dataset validation completed successfully")
        return

    # Filter and shuffle data
    df = filter_valid_with_hard_logic(df_original)
    df = df.sample(frac=1).reset_index(drop=True)
    validate_training_data(df, require_both_target_classes=True)

    logger.info(
        f"Percentage of valid runs in the filtered DataFrame: {len(df[df['is_valid'] == 1]) / len(df)}"
    )

    final_model_path = model_path(args.model_root_dir)
    if final_model_path.exists():
        raise FileExistsError(
            f"Model already exists at {final_model_path}. Remove it before retraining."
        )

    args.model_root_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.model_root_dir, prefix="autoconf-training-"
    ) as temporary_directory:
        training_path = Path(temporary_directory) / "model"
        predictor, df_train, df_test, elapsed_time = fit_tabular_predictor(
            df,
            train_fraction=args.train_fraction,
            preset_quality=args.preset_quality,
            output_path=training_path,
        )
        size_original = predictor.disk_usage()
        logger.info(f"Temporary model path is: {predictor.path}")

        # Refitting can improve inference speed at the cost of accuracy.
        if args.refit:
            predictor.refit_full(model="best", set_best_to_refit_full=True)

        predictor.clone_for_deployment(path=final_model_path)

    predictor_clone_opt = TabularPredictor.load(path=final_model_path)

    # Logging size comparison
    size_opt = predictor_clone_opt.disk_usage()
    logger.info(f"Size Original:  {size_original} bytes")
    logger.info(f"Size Optimized: {size_opt} bytes")
    logger.info(
        f"Optimized predictor achieved a {round((1 - (size_opt / size_original)) * 100, 1)}% reduction in disk usage."
    )
    metrics = log_metrics(
        predictor_clone_opt,
        df_test=df_test,
        df_train=df_train,
        train_fraction=args.train_fraction,
    )

    model_card_data = {
        "data_path": str(path),
        "dataset_url": args.dataset_url,
        "refit": args.refit,
        "model_version": MODEL_VERSION,
        "train_fraction": args.train_fraction,
        "preset_quality": args.preset_quality,
        "size_original_bytes": size_original,
        "size_optimized_bytes": size_opt,
        "elapsed_time": elapsed_time,
        "disk_usage_reduction_percent": round(
            (1 - (size_opt / size_original)) * 100, 1
        ),
    }

    if metrics:
        model_card_data.update(metrics)

    df_model_card = pd.DataFrame([model_card_data])

    model_card_path = final_model_path / "modelcard.csv"
    df_model_card.to_csv(model_card_path, index=False)

    logger.info(f"Model card saved successfully at: {model_card_path}")


if __name__ == "__main__":
    main()
