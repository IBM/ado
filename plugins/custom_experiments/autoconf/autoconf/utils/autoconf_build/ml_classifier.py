# Copyright (c) IBM Corporation

# SPDX-License-Identifier: MIT

# %% Run this script with IPython
import glob
import logging
import os
import shutil

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from autoconf.utils.rule_based_classifier import is_row_valid

logger = logging.getLogger(__name__)
logger.info("These are the available csvs")
data_root_dir = "/Users/danielelotito/autoconf_data"  # %change this to the data folder
glob.glob("*", root_dir=data_root_dir)
# %%
file_name = "lh_dashboard_136_date_01_13_2026.csv"
path = os.path.join(data_root_dir, file_name)
# %%
REFIT = False
train_fraction = 0.8
fit_params = {"presets": ["medium_quality"], "excluded_model_types": "GBM"}
suffix = f"-clone-opt-train_frac_{train_fraction}"  # this will be attached to the model folder name


df_original = pd.read_csv(path)
clist = list(df_original.columns)
cols_to_use = [
    "model_name",
    "method",  # LoRA, FULL
    "number_gpus",
    "gpu_model",
    "tokens_per_sample",  # this is: max_sequence_lenght
    "batch_size",
    "is_valid",  # Has the job being successful or did it have OOM problems?
    # NOTE: jobs that are not successful for incorrect specification of the config file are filtered out before training the model.
]
logger.info(set(df_original["model_name"].values))

# %%
target = "is_valid"


def filter_valid_with_hard_logic(df: pd.DataFrame):
    logger.debug(f"l before {len(df)}")
    valid_indices = [i for i, config in df.iterrows() if is_row_valid(config)[0]]
    df_filtered = df.loc[valid_indices].copy()
    logger.debug(f"l after {len(df_filtered)}")
    return df_filtered


# Our default is filtering valid rows with hard logic first
df = filter_valid_with_hard_logic(df_original)
df = df.sample(frac=1).reset_index(drop=True)


# %% You can decide here if you want to train
train_idx = int(len(df) * train_fraction)
df_train = df.iloc[:train_idx][cols_to_use]
df_test = df.iloc[train_idx:][cols_to_use]

df_test = filter_valid_with_hard_logic(df_test)

# %% TRAIN
train_data = TabularDataset(df_train)
train_data.head()
predictor = TabularPredictor(label=target).fit(train_data, **fit_params)
model_path = predictor.path
size_original = predictor.disk_usage()
logger.info("Model path is: ", model_path)


# %% TEST
def log_metrics(predictor, df_test):
    if not df_test.empty:
        test_data = TabularDataset(df_test)
        metrics_dict = predictor.evaluate(test_data, silent=True)
        logger.info("The model performance on the test data is", metrics_dict)
    else:
        metrics_dict = predictor.evaluate(train_data, silent=True)
        logger.info(f"The test df was empty, train fraction = {train_fraction}.")
        logger.info(" The model performance on the training data is", metrics_dict)
    return metrics_dict


# %% Refitting the original model is discretionary,  it improves inference speed but diminishes accuracy
# docs at <https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html>
if REFIT:
    predictor.refit_full(model="best", set_best_to_refit_full=True)
    suffix = "-refit" + suffix

save_path_refit_clone_opt = model_path + suffix
path_clone_opt = predictor.clone_for_deployment(path=save_path_refit_clone_opt)
predictor_clone_opt = TabularPredictor.load(path=save_path_refit_clone_opt)

# %% Logging size comparison
size_refit_opt = predictor_clone_opt.disk_usage()
logger.info(f"Size Original:  {size_original} bytes")
logger.info(f"Size Optimized: {size_refit_opt} bytes")
logger.info(
    f"Optimized predictor achieved a {round((1 - (size_refit_opt/size_original)) * 100, 1)}% reduction in disk usage."
)
metrics = log_metrics(predictor_clone_opt, df_test=df_test)
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
    "train_fraction": train_fraction,
    "size_original_bytes": size_original,
    "size_optimized_bytes": size_refit_opt,
    "disk_usage_reduction_percent": round(
        (1 - (size_refit_opt / size_original)) * 100, 1
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

# %%
