# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

# %% Run this script with IPython
import glob
import logging
import os
import shutil
import time
from typing import Any

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from autoconf.utils.rule_based_classifier import is_row_valid

logger = logging.getLogger(__name__)
logger.info("These are the available csvs")
data_root_dir = "/../../autoconf_data"  # %change this to the data folder
glob.glob("*", root_dir=data_root_dir)
# %%
file_name = "lh_dashboard_136_date_01_13_2026.csv"  # %change this to the data file name
path = os.path.join(data_root_dir, file_name)
# %%
REFIT = False
TRAIN_FRACTION = 0.8
PRESET_QUALITY = "medium_quality"
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

suffix = f"-clone-opt-train_frac_{TRAIN_FRACTION}"  # this will be attached to the model folder name

df_original = pd.read_csv(path)
clist = list(df_original.columns)
logger.info("Models supported are", set(df_original["model_name"].values))

# %%
target = "is_valid"


def filter_valid_with_hard_logic(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Length of the DataFrame before filtering: {len(df)}")
    valid_indices = [i for i, config in df.iterrows() if is_row_valid(config)[0]]
    df_filtered = df.loc[valid_indices].copy()
    logger.info(f"Length of the DataFrame after filtering {len(df_filtered)}")
    return df_filtered


# Our default is filtering valid rows with hard logic first
df = filter_valid_with_hard_logic(df_original)
df = df.sample(frac=1).reset_index(drop=True)

logger.info(
    f"Percentage of valid runs in the filtered DataFrame: {len(df[df['is_valid']==1]) / len(df)}"
)


# %% TRAININING FUNCTION
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
    predictor = TabularPredictor(label=target).fit(train_data, **fit_params)
    elapsed_time = time.time() - start
    return predictor, df_train, df_test, elapsed_time


# %% TEST
def log_metrics(
    predictor: TabularPredictor, df_test: pd.DataFrame, df_train: pd.DataFrame
) -> dict[str, Any]:
    if not df_test.empty:
        test_data = TabularDataset(df_test)
        metrics_dict = predictor.evaluate(test_data, silent=True)
        logger.info("The model performance on the test data is", metrics_dict)
    else:
        train_data = TabularDataset(df_train)
        metrics_dict = predictor.evaluate(train_data, silent=True)
        logger.info(f"The test df was empty, train fraction = {TRAIN_FRACTION}.")
        logger.info(" The model performance on the training data is", metrics_dict)
    return metrics_dict


predictor, df_train, df_test, elapsed_time = fit_tabular_predictor(
    df, train_fraction=TRAIN_FRACTION, preset_quality=PRESET_QUALITY
)
model_path = predictor.path
size_original = predictor.disk_usage()
logger.info("Model path is: ", model_path)


# %% Refitting the original model is discretionary,  it improves inference speed but diminishes accuracy
# docs at <https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html>
if REFIT:
    predictor.refit_full(model="best", set_best_to_refit_full=True)
    suffix = "-refit" + suffix

save_path_refit_clone_opt = model_path + suffix
path_clone_opt = predictor.clone_for_deployment(path=save_path_refit_clone_opt)
predictor_clone_opt = TabularPredictor.load(path=save_path_refit_clone_opt)

# %% Logging size comparison
size_opt = predictor_clone_opt.disk_usage()
logger.info(f"Size Original:  {size_original} bytes")
logger.info(f"Size Optimized: {size_opt} bytes")
logger.info(
    f"Optimized predictor achieved a {round((1 - (size_opt/size_original)) * 100, 1)}% reduction in disk usage."
)
metrics = log_metrics(predictor_clone_opt, df_test=df_test, df_train=df_train)
# %% cleaning up files, keeping only the refit-opt model
if model_path and os.path.isdir(model_path):
    try:
        shutil.rmtree(model_path, ignore_errors=True)
        logger.info(f"Deleted model directory: {model_path}")
    except Exception as e:
        logger.info(f"Could not delete model directory '{model_path}': {e}")

# %% saves in the model folder which is save_path_refit_clone_opt a the modelcard.csv which
# has all the value fixed in this script (data_path, refit, suffix, train_percetages, size, etc) +
# all the metrics contained in metrics dict which are additional columns in the csv
# ('accuracy', 'balanced_accuracy', ...) to do this we extract key values pairs from metrics

# 1. Create a dictionary with the metadata/configuration values
model_card_data = {
    "data_path": path,
    "refit": REFIT,
    "suffix": suffix,
    "train_fraction": TRAIN_FRACTION,
    "preset_quality": PRESET_QUALITY,
    "size_original_bytes": size_original,
    "size_optimized_bytes": size_opt,
    "elapsed_time": elapsed_time,
    "disk_usage_reduction_percent": round((1 - (size_opt / size_original)) * 100, 1),
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
