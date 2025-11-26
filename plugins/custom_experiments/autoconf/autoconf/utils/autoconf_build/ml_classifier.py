# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

# %% RUN IN IPYTHON KERNEL TO BUILD MODELS up to the end of the py file

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from min_gpu_recommender.utils.rule_based_classifier import filter_valid_with_hard_logic
import os
import shutil


# %% TO DEBUG
import glob
print('These are the available csvs')
glob.glob("*", root_dir="../../../data")  # %%

# %%
d_s = "11-13-dashboard-163-for-min-gpu"
path = f"../../../data/{d_s}.csv"
df_original = pd.read_csv(path)
clist = list(df_original.columns)
cols_to_use = [
    "model_name",
    "method",  # LoRA, FULL
    "number_gpus",
    "gpu_model",
    "tokens_per_sample",  # aka max_sequence_lenght
    "batch_size",
    "is_valid",  # Has the job being successfull or did it have OOM problems?
    # NOTE: jobs that are not successfull for incorrect specification of the config file must have been filtered out before.
    # As an example if the dataset_text_field was not specified correctly and the run fails I should not have an entry in this dataset
    # The only exception to this rule are the rules explicited in the imported function filter_valid_with_hard_logic
]
# %% Print models
print(set(list(df_original['model_name'].values)))

# %%
fit_params = { 'presets':['medium_quality'] , 'excluded_model_types': 'GBM'  }
# Evaluation is {'accuracy': 0.9037209302325582, 'balanced_accuracy': np.float64(0.9023296426071108), 'mcc': 0.8073578855359427, 'roc_auc': np.float64(0.9742294022168647), 'f1': 0.8961364776718515, 'precision': 0.918724279835391, 'recall': 0.8746327130264446}


# predictor.persist() 
target = "is_valid"
# %%
df = filter_valid_with_hard_logic(df_original)
df = df.sample(frac=1).reset_index(drop=True)
train_fraction = 0.8
train_idx = int(len(df) * train_fraction)
df_train = df.iloc[:train_idx][cols_to_use]
df_test = df.iloc[train_idx:][cols_to_use]




df_test = filter_valid_with_hard_logic(df_test)

# %% TRAIN
train_data = TabularDataset(df_train)
train_data.head()
predictor = TabularPredictor(label=target).fit(
    train_data, **fit_params
)  
# %% TEST
test_data = TabularDataset(df_test)
y_pred = predictor.predict(test_data.drop(columns=[target]))


d = predictor.evaluate(test_data, silent=True)
d_name = [predictor.eval_metric.name]
print('Evaluation is', d)
# %%

model_path = predictor.path # "/Users/danielelotito/Documents/github/genai-planning/autoconf_build/autoconf_build/AutogluonModels"
size_original = predictor.disk_usage()

print(model_path)




# %%
def clone_without_refitting():
    save_path_clone_opt = model_path  + "-clone-opt"
    path_clone_opt = predictor.clone_for_deployment(path=save_path_clone_opt)
    # will return the path to the cloned predictor, identical to save_path_clone_opt
    predictor_clone_opt = TabularPredictor.load(path=str(path_clone_opt))


    size_opt = predictor_clone_opt.disk_usage()
    print(f"Size Original:  {size_original} bytes")
    print(f"Size Optimized: {size_opt} bytes")
    print(
        f"Optimized predictor achieved a {round((1 - (size_opt/size_original)) * 100, 1)}% reduction in disk usage."
    )

    predictor_clone_opt.evaluate(
        test_data, silent=True
    )  # Same accuracy # from 570 MB to 182.2 MB


# %% REFITTING THE ORIGINAL ONE, 
# Reduces accuracy, in my experience (DL) of 3 percentage points, improves inference speed.
# docs at https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html

predictor.refit_full(model="best", set_best_to_refit_full=True)
save_path_refit_clone_opt = (
    model_path + "-refit-clone-opt"
)

path_refit_clone_opt = predictor.clone_for_deployment(path=save_path_refit_clone_opt)
predictor_refit_clone_opt = TabularPredictor.load(path=save_path_refit_clone_opt)
# %%
size_refit_opt = predictor_refit_clone_opt.disk_usage()
print(f"Size Original:  {size_original} bytes")
print(f"Size Optimized: {size_refit_opt} bytes")
print(
    f"Optimized predictor achieved a {round((1 - (size_refit_opt/size_original)) * 100, 1)}% reduction in disk usage."
)

predictor_refit_clone_opt.evaluate(
    test_data, silent=True
)  # Roghly Same accuracy # from 570 MB to 182.2 MB

# %% cleaning up files, keeping only the refit-opt model

if model_path and os.path.isdir(model_path):
    try:
        shutil.rmtree(model_path, ignore_errors=True)
        print(f"Deleted model directory: {model_path}")
    except Exception as e:
        print.warning(f"Could not delete model directory '{model_path}': {e}")






# %% BENCHMARK RESULTS OF OTHER PRESETS


fit_params = { 'presets':['good_quality', 'optimize_for_deployment'] , 'excluded_model_types': 'GBM'  }
# GOOD QUALITY , 30 X TRAINING TIME
# Evaluation is {'accuracy': 0.9418604651162791, 'balanced_accuracy': np.float64(0.9413340077745518), 'mcc': 0.8839195186837719, 'roc_auc': np.float64(0.9891419583278478), 'f1': 0.9390541199414919, 'precision': 0.9544103072348861, 'recall': 0.9241842610364683}
# SIZE CANNOT BE REDUCED, it is 353 MB



fit_params = { 'presets':['medium_quality', 'optimize_for_deployment'] , 'excluded_model_types': 'GBM'  }
# Evaluation is {'accuracy': 0.9218604651162791, 'balanced_accuracy': np.float64(0.9207964601769911), 'mcc': 0.8435183925882287, 'roc_auc': np.float64(0.982892156862745), 'f1': 0.9161676646706587, 'precision': 0.9329268292682927, 'recall': 0.9}
# SIZE CANNOT BE REDUCED, it is 315 MB


fit_params = { 'presets':['good_quality'] , 'excluded_model_types': 'GBM'  }
# 15:00 to 15:30
# Evaluation is {'accuracy': 0.9306976744186046, 'balanced_accuracy': np.float64(0.9304674821605297), 'mcc': 0.8610548415050774, 'roc_auc': np.float64(0.9861404207226687), 'f1': 0.9270680372001958, 'precision': 0.9284313725490196, 'recall': 0.9257086999022482}
# Size Original:  634112033 bytes # 600 MB
# Size Optimized: 259535977 bytes # 260 MB
# Optimized predictor achieved a 59.1% reduction in disk usage.

# NOTE: This 
# 'presets':['medium_quality_faster_inference_only_refit'] # in https://auto.gluon.ai/stable/tutorials/cloud_fit_deploy/cloud-aws-lambda-deployment.html
# is legacy
# ValueError: Preset 'medium_quality_faster_inference_only_refit' was not found. Valid presets: ['best_quality', 'high_quality', 'good_quality', 'medium_quality', 'optimize_for_deployment', 'ignore_text', 'ignore_text_ngrams', 'interpretable', 'best_quality_v082', 'high_quality_v082', 'good_quality_v082', 'extreme_quality', 'tabarena', 'experimental_quality_v120']


# NOTE: optimize_for_deployment is useful only for good quality, but expect ~260 mb models
# %%
