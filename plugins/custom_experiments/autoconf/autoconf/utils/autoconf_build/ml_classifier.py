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
glob.glob("*", root_dir="../../../data")  # %%

# %%
REFIT = True

d_s = "11-13-dashboard-163-for-min-gpu"
path = f"../../../data/{d_s}.csv"
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
fit_params = {"presets": ["medium_quality"], "excluded_model_types": "GBM"}
target = "is_valid"


def filter_valid_with_hard_logic(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug(f"l before {len(df)}")
    valid_indices = [i for i, config in df.iterrows() if is_row_valid(config)[0]]
    df_filtered = df.loc[valid_indices].copy()
    logger.debug(f"l after {len(df_filtered)}")
    return df_filtered


# Our default is filtering valid rows with hard logic first
df = filter_valid_with_hard_logic(df_original)
df = df.sample(frac=1).reset_index(drop=True)


# %% You can decide here if you want to train
train_fraction = 0.8
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
test_data = TabularDataset(df_test)
y_pred = predictor.predict(test_data.drop(columns=[target]))

d = predictor.evaluate(test_data, silent=True)
d_name = [predictor.eval_metric.name]
logger.info("Evaluation is", d)


# %% Refitting the original model is discretionary,  it improves inference speed but diminishes accuracy
# docs at <https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html>
if REFIT:
    predictor.refit_full(model="best", set_best_to_refit_full=True)


save_path_refit_clone_opt = model_path + "-refit-clone-opt"
path_clone_opt = predictor.clone_for_deployment(path=save_path_refit_clone_opt)
predictor_clone_opt = TabularPredictor.load(path=save_path_refit_clone_opt)

# %% Logging size comparison
size_refit_opt = predictor_clone_opt.disk_usage()
logger.info(f"Size Original:  {size_original} bytes")
logger.info(f"Size Optimized: {size_refit_opt} bytes")
logger.info(
    f"Optimized predictor achieved a {round((1 - (size_refit_opt/size_original)) * 100, 1)}% reduction in disk usage."
)

predictor_clone_opt.evaluate(test_data, silent=True)
# %% cleaning up files, keeping only the refit-opt model
if model_path and os.path.isdir(model_path):
    try:
        shutil.rmtree(model_path, ignore_errors=True)
        logger.info(f"Deleted model directory: {model_path}")
    except Exception as e:
        logger.info(f"Could not delete model directory '{model_path}': {e}")
