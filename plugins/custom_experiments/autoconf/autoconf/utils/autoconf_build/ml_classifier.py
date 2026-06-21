# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

# To build the model on an environment which is as clean as possible
# follow the instructions in plugins/custom_experiments/autoconf/autoconf/AutoGluonModels/README.md

import argparse
import glob
import logging
import os
import shutil
import time
from typing import Any

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import confusion_matrix

from autoconf.utils.rule_based_classifier import is_row_valid

logger = logging.getLogger(__name__)

# Default values
DEFAULT_DATA_ROOT_DIR = "/../../autoconf_data"
DEFAULT_FILE_NAME = "lh_dashboard_136_date_01_13_2026.csv"
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


def filter_valid_with_hard_logic(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Length of the DataFrame before filtering: {len(df)}")
    valid_indices = [i for i, config in df.iterrows() if is_row_valid(config)[0]]
    df_filtered = df.loc[valid_indices].copy()
    logger.info(f"Length of the DataFrame after filtering {len(df_filtered)}")
    return df_filtered


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AutoGluon ML classifier for autoconf",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Full path to the CSV data file. If not provided, uses data-root-dir and file-name.",
    )

    parser.add_argument(
        "--data-root-dir",
        type=str,
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

    return parser.parse_args()


# TRAINING FUNCTION
def fit_tabular_predictor(
    df: pd.DataFrame,
    train_fraction: float,
    preset_quality: str,
    cols_to_use: list[str] = COLS_TO_USE,
) -> tuple[TabularPredictor, pd.DataFrame, pd.DataFrame, float]:
    train_idx = int(len(df) * train_fraction)
    df_train = df.iloc[:train_idx][cols_to_use]
    df_test = df.iloc[train_idx:][cols_to_use]
    df_test = filter_valid_with_hard_logic(df_test)
    fit_params = {"presets": [preset_quality], "excluded_model_types": "GBM"}
    train_data = TabularDataset(df_train)
    train_data.head()
    start = time.time()
    predictor = TabularPredictor(label=TARGET).fit(train_data, **fit_params)
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
    # Parse command line arguments
    args = parse_arguments()

    # Determine data path
    path = args.data_path or os.path.join(args.data_root_dir, args.file_name)

    logger.info(f"Using data path: {path}")
    logger.info(f"REFIT: {args.refit}")
    logger.info(f"TRAIN_FRACTION: {args.train_fraction}")
    logger.info(f"PRESET_QUALITY: {args.preset_quality}")

    # Check available CSVs in data root directory
    if os.path.exists(args.data_root_dir):
        logger.info("These are the available csvs")
        available_files = glob.glob("*", root_dir=args.data_root_dir)
        logger.info(f"Available files: {available_files}")

    # Load and process data
    df_original = pd.read_csv(path)
    list(df_original.columns)
    logger.info(f"Models supported are: {set(df_original['model_name'].values)}")

    # Filter and shuffle data
    df = filter_valid_with_hard_logic(df_original)
    df = df.sample(frac=1).reset_index(drop=True)

    logger.info(
        f"Percentage of valid runs in the filtered DataFrame: {len(df[df['is_valid']==1]) / len(df)}"
    )

    # Create suffix for model folder name
    suffix = f"-clone-opt-train_frac_{args.train_fraction}"

    # Train model
    predictor, df_train, df_test, elapsed_time = fit_tabular_predictor(
        df, train_fraction=args.train_fraction, preset_quality=args.preset_quality
    )
    model_path = predictor.path
    size_original = predictor.disk_usage()
    logger.info(f"Model path is: {model_path}")

    # Refitting the original model is discretionary, it improves inference speed but diminishes accuracy
    # docs at <https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html>
    if args.refit:
        predictor.refit_full(model="best", set_best_to_refit_full=True)
        suffix = "-refit" + suffix

    save_path_refit_clone_opt = model_path + suffix
    predictor.clone_for_deployment(path=save_path_refit_clone_opt)
    predictor_clone_opt = TabularPredictor.load(path=save_path_refit_clone_opt)

    # Logging size comparison
    size_opt = predictor_clone_opt.disk_usage()
    logger.info(f"Size Original:  {size_original} bytes")
    logger.info(f"Size Optimized: {size_opt} bytes")
    logger.info(
        f"Optimized predictor achieved a {round((1 - (size_opt/size_original)) * 100, 1)}% reduction in disk usage."
    )
    metrics = log_metrics(
        predictor_clone_opt,
        df_test=df_test,
        df_train=df_train,
        train_fraction=args.train_fraction,
    )

    # Cleaning up files, keeping only the refit-opt model
    if model_path and os.path.isdir(model_path):
        try:
            shutil.rmtree(model_path, ignore_errors=True)
            logger.info(f"Deleted model directory: {model_path}")
        except Exception as e:
            logger.info(f"Could not delete model directory '{model_path}': {e}")

    # Saves in the model folder which is save_path_refit_clone_opt a the modelcard.csv which
    # has all the value fixed in this script (data_path, refit, suffix, train_percentages, size, etc) +
    # all the metrics contained in metrics dict which are additional columns in the csv
    # ('accuracy', 'balanced_accuracy', ...) to do this we extract key values pairs from metrics

    # 1. Create a dictionary with the metadata/configuration values
    model_card_data = {
        "data_path": path,
        "refit": args.refit,
        "suffix": suffix,
        "train_fraction": args.train_fraction,
        "preset_quality": args.preset_quality,
        "size_original_bytes": size_original,
        "size_optimized_bytes": size_opt,
        "elapsed_time": elapsed_time,
        "disk_usage_reduction_percent": round(
            (1 - (size_opt / size_original)) * 100, 1
        ),
    }

    # 2. Merge the metrics dictionary into the metadata dictionary
    # This adds keys like 'accuracy', 'balanced_accuracy', etc. as new columns
    if metrics:
        model_card_data.update(metrics)

    # 3. Create a DataFrame (wrapping data in a list to create a single row)
    df_model_card = pd.DataFrame([model_card_data])

    # 4. Construct the full path and save to CSV
    model_card_path = os.path.join(save_path_refit_clone_opt, "modelcard.csv")
    df_model_card.to_csv(model_card_path, index=False)

    logger.info(f"Model card saved successfully at: {model_card_path}")


if __name__ == "__main__":
    main()
