# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

# from sklearn import preprocessing
import itertools
import logging
import sys
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from paretoset import paretoset
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score, silhouette_score
from sklearn.tree import DecisionTreeClassifier

from orchestrator.utilities.naming import get_random_name_extension


def get_clusters(
    df: pd.DataFrame,
    data_columns: list[str],
    columns_to_mask: list[str],
    columns_to_unlist: list[str],
    unlist_index: int | None,
    targeted_value: str,
    max_range: int = 35,
    debug: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_np_prep = df.copy()
    translate_dicts = {}
    # TODO: automatically? but df.dtypes often has 'object'...
    #  also discrete_features could maybe be determined automatically?
    for col in columns_to_mask:
        num_unique = df_np_prep[col].nunique()
        td = dict(zip(df_np_prep[col].unique(), range(num_unique), strict=True))
        translate_dicts[col] = td

    for index, row in df_np_prep.iterrows():
        for col in columns_to_mask:
            df_np_prep.at[index, col] = translate_dicts[col][row[col]]
        for ulcol in columns_to_unlist:
            df_np_prep.at[index, ulcol] = row[ulcol][unlist_index]

    X = df_np_prep[data_columns].to_numpy(dtype="float")
    y = df_np_prep[targeted_value].to_numpy(dtype="float")

    if len(df_np_prep) <= (max_range + 2):
        max_range = len(df_np_prep) - 2
        logging.getLogger("clustering").debug(
            f"provided max_clusters is too large - adjusted to {max_range}"
        )
    cluster_trials = list(range(2, max_range))
    best_silhouette_avg = -2
    best_n_clusters = -1
    y_reshape = y.reshape(-1, 1)
    # X_norm = preprocessing.normalize(X)
    for n_clusters in cluster_trials:
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        cluster_labels = clusterer.fit_predict(y_reshape)
        silhouette_avg = silhouette_score(y_reshape, cluster_labels)
        if verbose:
            print(f"For n_clusters = {n_clusters}, silhouette_avg is {silhouette_avg}.")
        if silhouette_avg > best_silhouette_avg:
            best_silhouette_avg = silhouette_avg
            best_n_clusters = n_clusters
    if debug:
        print(
            f"Best scores found: {best_n_clusters} clusters with a silhouette score of {best_silhouette_avg}."
        )
    clusterer = KMeans(n_clusters=best_n_clusters, n_init="auto", random_state=10)
    cluster_labels = clusterer.fit_predict(y_reshape)

    return X, y, cluster_labels


class MutualInformationOutput(NamedTuple):

    mutual_information: dict[str, float]
    entropy: float
    cluster_labels: Any


def calculate_mutual_information(
    df: pd.DataFrame,
    data_columns: list[str],
    columns_to_mask: list[str],
    columns_to_unlist: list[str],
    unlist_index: int | None,
    targeted_value: str,
    discrete_features: bool = True,
    debug: bool = False,
    verbose: bool = False,
) -> MutualInformationOutput:
    "Returns a dict whose keys are data_columns and whose value is the MI for that column"

    X, _y, cluster_labels = get_clusters(
        df,
        data_columns,
        columns_to_mask,
        columns_to_unlist,
        unlist_index,
        targeted_value,
        debug=debug,
        verbose=verbose,
    )

    # The target variable are the clusters
    ig = mutual_info_classif(X, cluster_labels, discrete_features=discrete_features)

    # The entropy of the target
    entropy = mutual_info_score(cluster_labels, cluster_labels)
    logging.getLogger("mutual-information").debug(
        f"The entropy of the clusters is {entropy}"
    )

    mutual_information_labeled = dict(zip(data_columns, ig, strict=True))
    logging.getLogger("mutual-information").debug(
        f"For the target variable {targeted_value}, the mutual information scores for the clusters are:{mutual_information_labeled}"
    )
    # mutual_information_labeled['targeted_value'] = targeted_value

    return MutualInformationOutput(
        mutual_information=mutual_information_labeled,
        cluster_labels=cluster_labels,
        entropy=entropy,
    )


def mi_pareto_selection(
    mi_labeled_orig: dict[str, float],
    min_mi_threshold: float = 0.8,
    ignore_below: float = 0.0001,
    return_all_above_threshold: bool = False,
) -> list[str] | tuple[list[str], pd.DataFrame]:
    mi_labeled = {k: v for k, v in mi_labeled_orig.items() if v > ignore_below}
    l1 = list(mi_labeled.values())
    col_1 = list(mi_labeled.keys())
    pareto_pd = pd.DataFrame(
        [],
        columns=[
            "num_dimensions",
            "combined_mi",
            "selected_criteria",
            "above_threshold",
        ],
    )
    rid = 0
    mi_threshold = np.sum(l1) * min_mi_threshold
    below_threshold_since = 0
    for comb_len in range(len(l1) + 1, 1, -1):
        all_below_threshold = True
        for subset in list(set(itertools.combinations(enumerate(l1), comb_len))):
            selected_criteria = []
            sum_mi = 0.0
            num_dim = len(subset)
            # TODO take size of dimension into account
            for e in subset:
                selected_criteria.append(col_1[e[0]])
                sum_mi += e[1]
            pareto_pd.loc[rid] = [
                float(num_dim),
                float(sum_mi),
                selected_criteria,
                bool(sum_mi >= mi_threshold),
            ]
            if sum_mi >= mi_threshold:
                all_below_threshold = False
            rid += 1
        if all_below_threshold:
            below_threshold_since += 1
        if below_threshold_since > 2:
            break

    mask = paretoset(pareto_pd[["num_dimensions", "combined_mi"]], sense=["min", "max"])
    pareto_efficient = pareto_pd[mask]
    above_threshold = pareto_efficient[pareto_efficient["above_threshold"]]
    min_above = above_threshold["num_dimensions"].idxmin()
    new_dimensions = above_threshold.loc[min_above]["selected_criteria"]

    if return_all_above_threshold:
        return new_dimensions, above_threshold
    return new_dimensions


__ray_tune_keys__ = [
    "timestamp",
    "done",
    "training_iteration",
    "trial_id",
    "date",
    "time_this_iter_s",
    "time_total_s",
    "pid",
    "hostname",
    "node_ip",
    "time_since_restore",
    "iterations_since_restore",
    "checkpoint_dir_name",
]


def convert_values_of_dict(d_in: dict) -> tuple[dict, list[str]]:
    d_out = {}
    unmaskable = []
    # for k, v in d_in.items():
    #     if isinstance(v, (int, float)):
    #         d_out[k] = v
    #     elif isinstance(v, str):
    #         try:
    #             new_v = float(v)
    #             if new_v.is_integer():
    #                 d_out[k] = int(new_v)
    #             else:
    #                 d_out[k] = new_v
    #         except ValueError:
    #             d_out[k] = v
    #     else:
    #         # TODO?
    #         d_out[k] = v
    for k, v in d_in.items():
        if k == "valid_experiment" and not v:
            raise ValueError
        if (
            v is None
            or (isinstance(v, str) and v == "N/A")
            or (isinstance(v, float) and np.isnan(v))
        ):
            raise ValueError
        if isinstance(v, int):
            d_out[k] = v
        else:
            try:
                new_v = float(v)
                if new_v.is_integer():
                    d_out[k] = int(new_v)
                else:
                    d_out[k] = new_v
            except ValueError:
                d_out[k] = v
                unmaskable.append(k)
    return d_out, unmaskable


def mi_diff_over_time(
    df: pd.DataFrame,
    data_columns: list[str],
    columns_to_mask: list[str],
    columns_to_unlist: list[str],
    unlist_index: int | None,
    targeted_value: str,
    diffs_over_time: dict | None,
    last_mi: dict | None,
    threshold_diff: float,
    ranks_over_time: dict | None,
    pareto_over_time: list,
    consider_pareto_instead_ranks: bool,
    discrete_features: bool = True,
    debug: bool = False,
    verbose: bool = False,
) -> tuple[MutualInformationOutput, bool, bool, dict | None, dict | None, list]:
    # stateless calculation of diff development
    diff_ds = []
    mi_output = calculate_mutual_information(
        df,
        data_columns,
        columns_to_mask,
        columns_to_unlist,
        unlist_index,
        targeted_value,
    )
    new_mi = mi_output.mutual_information

    # Get the columns used for mi in a way that the order is the same on every call
    mi_values = np.array([new_mi[d] for d in data_columns])
    ranks = rankdata(mi_values, method="min")
    # Have the ranks such that the highest mi has lowest rank as this is what people expect
    # i.e. the rank 1 dimension is the "best"
    ranks = [len(ranks) - r + 1 for r in ranks]
    all_below_threshold = False
    change_in_ranks = True
    if last_mi is not None:
        diff_d, max_diff, _max_diff_label = diff_of_values(last_mi, new_mi)
        all_below_threshold = max_diff < threshold_diff
        diff_ds.append(diff_d)
        if diffs_over_time is None:
            diffs_over_time = {}
            ranks_over_time = {}
            for vid, (k, v) in enumerate(diff_d.items()):
                diffs_over_time[k] = [v]
                ranks_over_time[k] = [ranks[vid]]
        else:
            change_in_ranks = False
            for vid, (k, v) in enumerate(diff_d.items()):
                diffs_over_time[k].append(v)
                if ranks_over_time[k][-1] != ranks[vid]:
                    change_in_ranks = True
                ranks_over_time[k].append(ranks[vid])

    pareto_selection = mi_pareto_selection(new_mi)
    if len(pareto_over_time) > 0 and consider_pareto_instead_ranks:  # noqa: SIM102
        # >= to allow also the shrinkage of pareto sets
        if set(pareto_over_time[-1]) >= set(pareto_selection):
            change_in_ranks = False

    logging.getLogger("mutual-information").debug(
        f"new pareto selection: {pareto_selection} (change in ranks: {change_in_ranks})."
    )
    pareto_over_time.append(pareto_selection)
    return (
        mi_output,
        all_below_threshold,
        change_in_ranks,
        diffs_over_time,
        ranks_over_time,
        pareto_over_time,
    )


def diff_of_values(d1: dict, d2: dict) -> tuple[dict, float, str]:
    diff_d = {}
    max_diff = -1
    max_diff_label = "none"
    for key, value in d1.items():
        diff = abs(d2[key] - value)
        if diff > max_diff:
            max_diff = diff
            max_diff_label = key
        diff_d[key] = diff
    return diff_d, max_diff, max_diff_label


def get_valid_value_ranges(
    df: pd.DataFrame,
    data_columns: list[str],
    columns_to_mask: list[str],
    columns_to_unlist: list[str],
    unlist_index: int | None,
    targeted_value: str,
    failed_values: list | None = None,
    debug: bool = False,
    verbose: bool = False,
) -> dict[str, list]:

    if failed_values is None:
        failed_values = [
            float("nan"),
            float(sys.maxsize),
            float("-inf"),
            float("inf"),
            "N/A",
            None,
        ]

    df_np_prep = df.copy()
    translate_dicts = {}
    # TODO: automatically? but df.dtypes often has 'object'...
    #  also discrete_features could maybe be determined automatically?
    for col in columns_to_mask:
        num_unique = df_np_prep[col].nunique()
        td = dict(zip(df_np_prep[col].unique(), range(num_unique), strict=True))
        translate_dicts[col] = td

    for index, row in df_np_prep.iterrows():
        for col in columns_to_mask:
            df_np_prep.at[index, col] = translate_dicts[col][row[col]]
        for ulcol in columns_to_unlist:
            df_np_prep.at[index, ulcol] = row[ulcol][unlist_index]

    label_column_name = f"label_{get_random_name_extension()}"
    label_column_name_int = label_column_name + "_int"

    df_np_prep[label_column_name] = df_np_prep[targeted_value].apply(
        lambda x: "OK" if x not in failed_values else "FAILING"
    )
    df_np_prep[label_column_name_int] = df_np_prep[targeted_value].apply(
        lambda x: 0 if x not in failed_values else 1
    )

    X = df_np_prep[data_columns].to_numpy(dtype="float")
    y = df_np_prep[label_column_name_int].to_numpy(dtype="float")

    dt = DecisionTreeClassifier()
    dt.fit(X, y)

    # for debugging...
    # class_names = ['OK', 'failing']
    # with open("dt.dot", 'w') as f:
    #     export_graphviz(dt, out_file=f, feature_names=data_columns, class_names=class_names)
    # $ dot -Tpng dt.dot -o dt.png

    invalid_values_dict = {}
    for k, v in translate_dicts.items():
        invalid_values_dict[k] = list(v.values())
    features_touched = []

    node_indicator = dt.decision_path(X)
    sample_ids = [i for i, v in enumerate(y) if v == 1]
    if len(sample_ids) == 0:
        # all experiments are valid
        # TODO
        valid_values_dict = {}
        for col in data_columns:
            if col in columns_to_mask:
                valid_values_dict[col] = list(translate_dicts[col].keys())
            else:
                valid_values_dict[col] = [kk for kk, vv in translate_dicts[col].items()]
        return valid_values_dict
    # n_nodes = dt.tree_.node_count
    # common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
    sample_id = sample_ids[0]
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    leaf_id = dt.apply(X)
    # leaf_ids = list(set(leaf_id))
    # common_thresholds = [threshold[i] for i in common_node_id if i not in leaf_ids]
    # common_feature_ids = [feature[i] for i in common_node_id if i not in leaf_ids]
    # common_features = [data_columns[i] for i in common_feature_ids]

    for node_id in node_index:
        if leaf_id[sample_id] == node_id:
            continue
        cur_feature = feature[node_id]
        feature_name = data_columns[cur_feature]
        if feature_name not in features_touched:
            features_touched.append(feature_name)
        value_list = invalid_values_dict[feature_name]
        cur_threshold = threshold[node_id]
        if X[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
            new_value_list = [v for v in value_list if v <= cur_threshold]
        else:
            threshold_sign = ">"
            new_value_list = []
            for v in value_list:
                if v > cur_threshold:
                    new_value_list.append(v)
        invalid_values_dict[feature_name] = new_value_list

        if verbose:
            print(
                f"decision node {node_id} : (X[{sample_id}, {feature[node_id]}] = {X[sample_id, feature[node_id]]}) "
                f"{threshold_sign} {threshold[node_id]})"
            )

    # TODO
    # assert features_touched == list(set(common_features))

    valid_values_dict = {}
    for col in data_columns:
        if col not in features_touched:
            if col in columns_to_mask:
                valid_values_dict[col] = list(translate_dicts[col].keys())
            else:
                valid_values_dict[col] = [
                    kk
                    for kk, vv in translate_dicts[col].items()
                    if vv not in invalid_values_dict[col]
                ]
        else:
            valid_values_dict[col] = list(df_np_prep[col].unique())

    if debug:
        print(valid_values_dict)
    return valid_values_dict
