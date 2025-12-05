# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import asyncio
import logging
import os
import typing

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from pydantic import BaseModel

from orchestrator.core.discoveryspace.samplers import BaseSampler
from orchestrator.core.discoveryspace.space import DiscoverySpace, Entity
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager
from trim.trim_pydantic import TrimParameters
from trim.utils.compare_describe_log_sources import (
    describe_source_spaces,
    get_train_and_holdout_df_from_source_dfs_of_last_iters,
)
from trim.utils.exceptions import InsufficientDataError
from trim.utils.miscellaneous import delete_dir
from trim.utils.one_dimensional_sampling import get_index_list_nn
from trim.utils.order import get_feature_importance_order, reorder_df_by_importance
from trim.utils.space_df_connector import (
    get_list_of_entities_from_df_and_space,
    get_source_and_target,
)

logger_trim = logging.getLogger(__name__)
logger_trim.setLevel(logging.DEBUG)


# NOTE: to repeat the operation on the same space I can delete the operation if the output of this operation
# are not used by another operation
class TrimSampleSelector(BaseSampler):
    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls, remoteDiscoverySpace: DiscoverySpaceManager  # type: ignore[name-defined]
    ):
        # do you want to return False if no point has been measured?
        return True

    async def remoteEntityIterator(
        self, remoteDiscoverySpace, batchsize=1
    ) -> typing.AsyncGenerator[list[Entity], None]:
        """Returns an remoteEntityIterator that returns entities in order"""

        async def iterator_closure(
            stateHandle: DiscoverySpaceManager,  # type: ignore[name-defined]
        ) -> typing.Callable[[], typing.AsyncGenerator[list[Entity], None]]:
            logger_trim.info("Trim sampler initialized. Iterative modeling starts.\n")
            logger_trim.info(f"PARAMETERS ARE:\n{self.params}\n\n")

            # raise ValueError

            discoverySpace = await stateHandle.discoverySpace.remote()
            list_of_entities, df_ordered_to_sample = (
                self.entities_for_iterative_modeling_from_discovery_space(
                    discoverySpace=discoverySpace
                )
            )
            numberEntities = len(list_of_entities)

            async def iterator() -> typing.AsyncGenerator[list[Entity], None]:  # type: ignore[name-defined][name-defined]
                await asyncio.sleep(0.001)

                source_df, _target_df = get_source_and_target(
                    discoverySpace, self.params.targetOutput
                )
                previous_iteration_source_df = source_df
                previous_iteration_source_df.to_csv(
                    os.path.join(self.params.debugDirectory, "initial_source_df.csv")
                )

                train_cols = [
                    cp.identifier
                    for cp in discoverySpace.entitySpace.constitutiveProperties
                ]
                train_target_cols = [*train_cols, self.params.targetOutput]
                logger_trim.info(
                    f"Trim iterator will measure up to {numberEntities} entities.\
                    These entities have been ordered using {len(source_df)} measurements from the discovery space."
                )

                logger_trim.info(
                    f"Training columns are {train_cols},\nThe dependent variable (target Output) is {train_target_cols[-1]}"
                )

                metric_iteration_dict = {}
                comparison_indeces = []
                for i in range(0, numberEntities, batchsize):
                    entities = list_of_entities[i : i + batchsize]

                    if len(entities) == 0:
                        break

                    logger_trim.info(
                        "Building and evaluating a predictive model \
                                that includes one more entity in the training set.\
                                Entity is:"
                    )
                    logger_trim.info(df_ordered_to_sample[train_cols].iloc[[i]])

                    this_iteration_source_df, _this_iteration_target_df = (
                        get_source_and_target(discoverySpace, self.params.targetOutput)
                    )
                    # now source contains all sampled points, total points form the target space is i

                    describe_source_spaces(
                        this_iteration_source_df,
                        previous_iteration_source_df,
                        filter_cols=train_target_cols,
                    )

                    train_df, holdout_df = (
                        get_train_and_holdout_df_from_source_dfs_of_last_iters(
                            this_iteration_source_df,
                            previous_iteration_source_df,
                            train_cols=train_cols,
                            train_target_cols=train_target_cols,
                            holdout_size=self.params.holdoutSize,
                        )
                    )

                    # TODO: check and log warning if holdouts are different
                    if i == 0:
                        temp_holdout = holdout_df
                    else:
                        if temp_holdout.equals(holdout_df):
                            logger_trim.warning("Holdout dataframe is not changing!")
                        temp_holdout = holdout_df

                    if logger_trim.isEnabledFor(logging.DEBUG):
                        if i != 0:
                            this_iteration_source_df.to_csv(
                                os.path.join(
                                    self.params.debugDirectory,
                                    f"source_at_iter_{i}.csv",
                                ),
                                index=False,
                            )
                        train_df.to_csv(
                            os.path.join(
                                self.params.debugDirectory, f"train_at_iter_{i}.csv"
                            ),
                            index=False,
                        )
                        holdout_df.to_csv(
                            os.path.join(
                                self.params.debugDirectory, f"holdout_at_iter_{i}.csv"
                            ),
                            index=False,
                        )

                    # we rename appropriately
                    previous_iteration_source_df = this_iteration_source_df

                    ##### MODEL BUILDING AND EVALUATION #####
                    # ensure we only train on rows where the target is measured
                    if not train_df.equals(
                        train_df.dropna(subset=[str(self.params.targetOutput)])
                    ):
                        logger_trim.warning(
                            "There are rows in train df where the target is NaN! Dropping them now.\n\n"
                        )
                        train_df = train_df.dropna(subset=[self.params.targetOutput])

                    if train_df.empty:
                        logger_trim.warning(
                            "Empty training df, this means either that the operation configuration or \
                                            the inference problem is ill posed"
                        )
                        raise ValueError

                    train_data = TabularDataset(train_df)
                    holdout_data = TabularDataset(holdout_df)

                    # NOTE: assigning more weight to target space points does NOT generally improve performance
                    # due diligence has been done

                    # Now, train a model on new_source_df and get performance
                    predictor = TabularPredictor(
                        label=self.params.targetOutput,
                        **(self.params.autoGluonArgs.tabularPredictorArgs or {}),
                    )

                    fit_kwargs = (
                        getattr(
                            getattr(self.params, "autoGluonArgs", None), "fitArgs", None
                        )
                        or {}
                    )
                    logger_trim.info(
                        f"Fitting AutoGluon TabularPredictor, iteration {i}..."
                    )
                    predictor.fit(train_data=train_data, **fit_kwargs)

                    # metric metric used in training
                    training_metric = getattr(predictor, "eval_metric", None)
                    lb = predictor.leaderboard(silent=True)
                    if lb is not None and not lb.empty:
                        best_row = lb.iloc[0]
                        best_model_name = best_row.get("model", None)
                        best_score_val = best_row.get("score_val", None)
                    else:
                        best_model_name, best_score_val = None, None

                    metric_iteration_dict[i] = {
                        "metric": training_metric,
                        "best_model": best_model_name,
                        "best_score_val": best_score_val,
                        "holdout_score": predictor.evaluate(holdout_data, silent=True)[
                            predictor.eval_metric.name
                        ],
                    }

                    log_metric_string = f"""[Iteration {i}] Training metric: {training_metric};
                    Best model: {best_model_name}; score_val: {best_score_val}; holdout_score: {metric_iteration_dict[i]['holdout_score']}"""
                    logger_trim.info(log_metric_string)

                    # Capture model path and delete the folder
                    if not logger_trim.isEnabledFor(logging.DEBUG):
                        model_dir = getattr(predictor, "path", None)
                        logger_trim.info(f"AutoGluon model directory: {model_dir}")
                        del predictor
                        delete_dir(model_dir=model_dir)

                    # Use the best validation score captured earlier as the "mean ratio" proxy for stopping
                    _metric_entry = metric_iteration_dict.get(i, {})
                    _best_score_val = _metric_entry.get("holdout_score", None)
                    should_stop = 0

                    # for the first 2*iterationSize we do not have enough data to compare
                    if i < self.params.iterationSize * 2:
                        continue
                    # comparison happens at every params.iterationSize steps
                    elif comparison_indeces:
                        if max(comparison_indeces) + self.params.iterationSize > i:
                            continue  # next iteration of the for, where I will sample another point

                    else:
                        comparison_indeces.append(i)
                        prev_iter_list_range = list(
                            range(
                                i - self.params.iterationSize * 2 + 1,
                                i - self.params.iterationSize + 1,
                            )
                        )
                        this_iter_list_range = list(
                            range(i - self.params.iterationSize, i + 1)
                        )
                        logger_trim.info(
                            f"""Since iterationSize is {self.params.iterationSize}. We now
                            compare models at i-ranges\n{prev_iter_list_range}\nand\n{this_iter_list_range}"""
                        )

                        scores_previous_iteration = [
                            metric_iteration_dict[el]["best_score_val"]
                            for el in prev_iter_list_range
                        ]
                        scores_this_iteration = [
                            metric_iteration_dict[el]["best_score_val"]
                            for el in this_iter_list_range
                        ]

                        logger_trim.info(
                            f"Scores that correspond to these i-ranges are:\n{prev_iter_list_range}\nand\n{this_iter_list_range}"
                        )

                        try:
                            mean_ratio = (
                                np.array(scores_this_iteration).mean()
                                / np.array(scores_previous_iteration).mean()
                            )
                            std_ratio = (
                                np.array(scores_this_iteration).std()
                                / np.array(scores_previous_iteration).std()
                            )
                        except Exception as e:
                            logger_trim.warning(
                                f"Exception occurred: {e}, should stop will be true."
                            )
                            mean_ratio = 1
                            std_ratio = 1

                        should_stop = (
                            _best_score_val is not None
                            and stopping_bool_from_ratios(
                                mean_ratio=mean_ratio, std_ratio=std_ratio
                            )
                        )

                    if should_stop:
                        hardcoded_folder_name = "trim_models/"
                        # Stopping info
                        stop_info = f"""Stopping criteria hit after measuring {i} entities.
                                        On a iteration of batch size {self.params.iterationSize}.
                                        Performance of the model on the holdout set
                                        {training_metric}:\nmean: {_best_score_val}\t\tstd: {std_ratio}\n.
                                        """
                        logger_trim.info(stop_info)

                        # FIT ON FULL SOURCE SPACE DATA
                        source_df, _target_df = get_source_and_target(
                            discoverySpace, self.params.targetOutput
                        )
                        logger_trim.info(
                            f"Fitting AutoGluon TabularPredictor on full Source Space data,  {i}..."
                        )

                        train_df = source_df[train_target_cols]
                        # think about replicating here the guardrail about NaN in target
                        if logger_trim.isEnabledFor(logging.DEBUG):
                            train_df.to_csv(
                                os.path.join(
                                    self.params.debugDirectory,
                                    "final_model_training_data.csv",
                                ),
                                index=False,
                            )

                        train_data = TabularDataset(train_df)
                        # Now, train a model on new_source_df and get performance
                        predictor = TabularPredictor(
                            label=self.params.targetOutput,
                            # problem_type="regression", # it is inferred atm
                            **(
                                self.params.finalModelAutoGluonArgs.tabularPredictorArgs
                                or {}
                            ),
                        )
                        fit_kwargs = (
                            getattr(
                                getattr(self.params, "finalModelautoGluonArgs", None),
                                "fitArgs",
                                None,
                            )
                            or {}
                        )
                        predictor.fit(train_data=train_data, **fit_kwargs)

                        # metric metric used in training
                        # TODO: put this in the operation metadata at the end
                        final_lb = predictor.leaderboard(silent=True)
                        final_model_metric = (
                            final_lb.iloc[0].get("score_val", None)
                            if final_lb is not None and not final_lb.empty
                            else None
                        )

                        save_info = f"""Model finalized using as training set all sampled points, of len {len(train_data)}.\n
                                        Final model {training_metric}={final_model_metric}.
                                        Saving predicted model to: {hardcoded_folder_name}.
                                        """
                        logger_trim.info(save_info)
                        break

                    else:
                        yield_log_string = f"Stopping not triggered for i={i}"
                        logger_trim.info(yield_log_string)

                        # TODO: Check if this is stable without try statement
                        # try:
                        logger_trim.info("Entities yielded in this iteration are:\n")
                        for e in entities:
                            logger_trim.info(e)
                        # except:
                        #     logger_trim.info("No id")

                        yield entities

            return iterator

        # iterator_closure is an async function, so calling it returns a coroutine object.
        # To get its return value (the iterator function), you must await that coroutine.
        # Without await, retval would just be a coroutine object, not the actual function reference.
        retval = await iterator_closure(remoteDiscoverySpace)

        # NOTE:
        # Any function defined with async def is an asynchronous function.
        # It must have an await in it. In this case I have an await sleep
        # When you call that function, Python does not execute the body immediately.
        # Instead, it returns a coroutine object, which represents the pending execution of that function.

        # A coroutine object:

        # It is awaitable (you can use await on it).
        # It encapsulates the state of the async function until execution.
        # It does not run until awaited.

        # Returning an async generator object # Ready to iterate on with async for ...
        return retval()

    def entities_for_iterative_modeling_from_discovery_space(
        self,
        discoverySpace: DiscoverySpace,
    ) -> tuple[list, pd.DataFrame]:
        """
        Generate an ordered list of entities for iterative modeling from a discovery space.

        Steps:
        - Validate source data (distinct target values, minimum sampling budget).
        - Compute feature importance and reorder source-target merged dataframe.
        - Determine sampling order using nearest-neighbor strategy.
        - Return ordered entities and the corresponding dataframe.

        Parameters
        ----------
        discoverySpace : DiscoverySpace
            The discovery space containing entities and measured data.

        Returns
        -------
        tuple
            (list_of_entities, df_ordered_to_sample)

        Raises
        ------
        InsufficientDataError
            If data is insufficient for modeling.
        ValueError
            If validation checks fail.
        """

        source_df, target_df = get_source_and_target(
            discoverySpace, self.params.targetOutput
        )

        if logger_trim.isEnabledFor(logging.DEBUG):
            source_df.to_csv(
                os.path.join(self.params.debugDirectory, "Initial_source_space.csv")
            )

        distinct_count = source_df[self.params.targetOutput].nunique(dropna=False)
        if distinct_count == 1:
            unique_val = source_df[self.params.targetOutput].unique()[0]
            msg = (
                f"Target output '{self.params.targetOutput}' has only a single distinct value: {unique_val}. "
                "This is insufficient for downstream processing."
            )
            logger_trim.error(msg)
            raise InsufficientDataError

        if len(source_df) < self.params.samplingBudget.minPoints:
            info_str = """This may happen because it may be that the target variable cannot be measured for all
            the entities in the space. For example a recommender could be unable to recommend the target variables
            for some entities"""
            missing_points = self.params.samplingBudget.minPoints - len(source_df)
            logger_trim.error(
                f"Insufficient data: need {self.params.samplingBudget.minPoints}, but only {len(source_df)} available. "
                f"Consider adding {missing_points} more points or adjusting the budget."
            )
            logger_trim.info(info_str)
            if len(source_df) > 10:
                logger_trim.warning(
                    "Attempting iterative modelling with 10 source space points"
                )
            else:
                raise InsufficientDataError(
                    f"Insufficient data: need {self.params.samplingBudget.minPoints}, but only {len(source_df)} available. "
                )

        # Compute feature importance and order
        ordered_features, _importance_dict = get_feature_importance_order(
            source_df=source_df,
            target_output=self.params.targetOutput,
            min_measured_entities=self.params.minMeasuredEntities,
            autoGluonArgs=self.params.autoGluonArgs,
        )

        # TODO: see if you need to explicitly  # Merge source and target on train_cols
        # train_cols = [
        #     cp.identifier for cp in discoverySpace.entitySpace.constitutiveProperties
        # ]
        # train_target_cols = train_cols + [self.params.targetOutput]

        merged_df = source_df.merge(target_df, how="outer")

        if logger_trim.isEnabledFor(logging.DEBUG):
            merged_df.to_csv(
                os.path.join(self.params.debugDirectory, "debug_merged.csv")
            )
            source_df.to_csv(
                os.path.join(self.params.debugDirectory, "debug_source.csv")
            )
            target_df.to_csv(
                os.path.join(self.params.debugDirectory, "debug_target.csv")
            )

        # Check that rows with NaNs in train_target_cols equal len(target_df)
        nan_rows_count = merged_df[[self.params.targetOutput]].isna().any(axis=1).sum()
        if nan_rows_count != len(target_df):
            msg = (
                f"Validation failed: Expected {len(target_df)} rows with NaNs in {self.params.targetOutput}, "
                f"but found {nan_rows_count}."
            )
            logging.error(msg)
            raise ValueError(msg)

        # Order merged dataframe by source space feature importance
        merged_df_ordered_by_source_importance = reorder_df_by_importance(
            merged_df, ordered_features
        )

        # Sampled indices: rows where targetOutput is NOT NaN
        sampled_indices = merged_df_ordered_by_source_importance[
            merged_df_ordered_by_source_importance[self.params.targetOutput].notna()
        ].index.tolist()

        # Compute index order for sampling
        idx_order = get_index_list_nn(
            len(merged_df_ordered_by_source_importance),
            len(target_df),
            sampled_indices=sampled_indices,
        )

        # Filter out sampled indices while maintaining order
        idx_order_filtered = [i for i in idx_order if i not in sampled_indices]

        # Final dataframe to sample
        df_ordered_to_sample = merged_df_ordered_by_source_importance.iloc[
            idx_order_filtered
        ]

        # --------------------------
        # old
        # NOTE: If you want to apply TRIM on CCSE you may want the following instead
        # from trim.utils.order import get_df_ordered_by_source_space_importance
        # sampled_indices = []
        # df_target_ordered_by_source_importance = get_df_ordered_by_source_space_importance(discoverySpace, params = self.params)
        # idx_order = get_index_list_nn(len(df_target_ordered_by_source_importance), len(df_target_ordered_by_source_importance), sampled_indices=sampled_indices)
        # df_ordered_to_sample = df_target_ordered_by_source_importance.iloc[idx_order]
        # --------------------------

        ordered_df_path_and_name = "df_ordered_to_sample_with_id.csv"
        ordered_data_log_string = f"DataFrame successfully ordered, saving it now to {ordered_df_path_and_name}"
        logger_trim.info(ordered_data_log_string)
        list_of_entities_identifiers = df_ordered_to_sample["identifier"]
        list_of_entities_identifiers_log = f"""Ordered list of inferred entities identifiers is:\n{list_of_entities_identifiers}\n\n
        Proceeding to sample entities in this order.\n
        Valid entities are built and validated using the dataframe contained in {ordered_df_path_and_name}"""

        logger_trim.info(list_of_entities_identifiers_log)

        list_of_entities = get_list_of_entities_from_df_and_space(
            df=df_ordered_to_sample, space=discoverySpace
        )

        if logger_trim.isEnabledFor(logging.DEBUG):
            df_ordered_to_sample.to_csv(ordered_df_path_and_name)
        return list_of_entities, df_ordered_to_sample

    # NOTE: I do not know if I have to insert trim logic inside the not-remote entity iterator
    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize=1
    ) -> typing.Generator[list[Entity], None, None]:
        """Returns an remoteEntityIterator that returns entities in order"""

        def iterator_closure(
            space: DiscoverySpace,
        ) -> typing.Callable[[], typing.Generator[list[Entity], None, None]]:

            list_of_entities = [...]
            numberEntities = len(list_of_entities)

            def iterator() -> typing.Generator[list[Entity], None, None]:  # type: ignore[name-defined]
                for i in range(0, numberEntities, batchsize):
                    # batch = list_of_entities[i : i + batchsize]
                    ...

            return iterator

        retval = iterator_closure(discoverySpace)
        return retval()

    @classmethod
    def parameters_model(cls) -> type[BaseModel] | None:
        return TrimParameters

    def __init__(self, parameters: TrimParameters):
        self.params = parameters


def stopping_bool_from_ratios(
    mean_ratio: float,
    std_ratio: float,
    mean_ratio_threshold: float = 0.9,
    std_ratio_threshold: float = 0.75,
):
    """
    Determine whether sampling should stop based on mean and standard deviation ratios.

    The function evaluates whether the mean ratio lies within a symmetric threshold
    range around 1, and whether the standard deviation ratio is below its threshold.
    It returns a boolean indicating if all conditions are satisfied.

    Parameters
    ----------
    mean_ratio : float
        Ratio of the current mean compared to a reference mean.
    std_ratio : float
        Ratio of the current standard deviation compared to a reference standard deviation.
    mean_ratio_threshold : float, optional
        Lower bound threshold for the mean ratio (default is 0.9).
        The upper bound is taken as the reciprocal (1 / mean_ratio_threshold).
    std_ratio_threshold : float, optional
        Upper bound threshold for the standard deviation ratio (default is 0.75).

    Returns
    -------
    bool
        True if mean_ratio is greater than `mean_ratio_threshold` and less than
        `1 / mean_ratio_threshold`, and std_ratio is less than `1 / std_ratio_threshold`.
        False otherwise.

    Notes
    -----
    This logic works for both maximum- and minimum-based metrics, ensuring
    ratios remain within acceptable bounds before stopping.
    """
    return (
        (mean_ratio > mean_ratio_threshold)
        and (mean_ratio < 1 / mean_ratio_threshold)
        and (std_ratio < 1 / std_ratio_threshold)
    )
