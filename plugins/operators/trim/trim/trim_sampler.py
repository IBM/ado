# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import asyncio
import logging
import os
import typing
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from pydantic import BaseModel

from orchestrator.core.discoveryspace.samplers import BaseSampler
from orchestrator.core.discoveryspace.space import DiscoverySpace, Entity
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager
from trim.trim_pydantic import TrimParameters
from trim.utils.exceptions import InsufficientDataError
from trim.utils.miscellaneous import delete_dir
from trim.utils.one_dimensional_sampling import get_index_list_nn
from trim.utils.order import get_feature_importance_order, reorder_df_by_importance
from trim.utils.rowsring import RowsRing
from trim.utils.space_df_connector import (
    get_list_of_entities_from_df_and_space,
    get_source_and_target,
)
from trim.utils.split_common_and_diff import (
    split_common_and_diff,
)

logger_trim_sampler = logging.getLogger(__name__)
logger_trim_sampler.setLevel(logging.DEBUG)


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

        logger_trim_sampler.debug(f"Batchsize is {batchsize} (expected 1)")

        async def iterator_closure(
            stateHandle: DiscoverySpaceManager,  # type: ignore[name-defined]
        ) -> typing.Callable[[], typing.AsyncGenerator[list[Entity], None]]:
            logger_trim_sampler.info(
                "Trim sampler initialized. Iterative modeling starts.\n"
            )
            logger_trim_sampler.info(f"PARAMETERS ARE:\n{self.params}\n\n")

            if logger_trim_sampler.isEnabledFor(logging.DEBUG):
                # I create the folder at self.params.debugDirectory if not present
                debug_dir = Path(self.params.debugDirectory).expanduser().resolve()
                logger_trim_sampler.debug(
                    f"Creating a folder to save intermediate files:\n{debug_dir}\n\n"
                )
                debug_dir.mkdir(parents=True, exist_ok=True)  # creates if missing

            discoverySpace = await stateHandle.discoverySpace.remote()
            list_of_entities, _df_ordered_to_sample = (
                self.entities_for_iterative_modeling_from_discovery_space(
                    discoverySpace=discoverySpace
                )
            )
            numberEntities = len(list_of_entities)

            async def iterator() -> typing.AsyncGenerator[list[Entity], None]:  # type: ignore[name-defined][name-defined]
                await asyncio.sleep(0.001)

                # Recording the initial source space
                initial_source_df, _target_df = get_source_and_target(
                    discoverySpace,
                    self.params.targetOutput,
                    discoverySpaceManager=stateHandle,
                )

                if logger_trim_sampler.isEnabledFor(logging.DEBUG):
                    initial_source_df.to_csv(
                        os.path.join(
                            self.params.debugDirectory, "initial_source_df.csv"
                        )
                    )

                train_cols = [
                    cp.identifier
                    for cp in discoverySpace.entitySpace.constitutiveProperties
                ]
                train_target_cols = [*train_cols, self.params.targetOutput]
                logger_trim_sampler.info(
                    f"Trim iterator will measure up to {numberEntities} entities.\
                    These entities have been ordered using {len(initial_source_df)} measurements from the discovery space."
                )

                logger_trim_sampler.info(
                    f"Training columns are {train_cols},\nThe dependent variable (target Output) is {train_target_cols[-1]}"
                )

                ############################################################################################################
                ######################################### MAIN LOOP STARTS #################################################
                ############################################################################################################

                metric_batch_size_dict = {}
                comparison_indices = []
                previous_holdout_df = pd.DataFrame({})
                yielded_entities = deque(maxlen=self.params.holdoutSize)
                yielded_rows = RowsRing(
                    maxlen=(
                        self.params.holdoutSize
                        if self.params.holdoutSize
                        else self.params.iterationSize
                    )
                )
                # TODO: same data structured but It is made up df rows, yielded_rows.df returns a df made of of those rows
                # upon adding a new row a check is done automatically so that the rows is coherent with the others
                # rows index information is discarded, indices are from 0 to self.params.holdoutSize -1 and are automatically updated each time a
                # row is added, so that idx 0 is the oldest point, and when the ring is full, self.params.holdoutSize -1 is the latest added point

                for i in range(0, numberEntities, batchsize):
                    entities = list_of_entities[i : i + batchsize]

                    if len(entities) == 0:
                        logger_trim_sampler.warning("No Entities remaining.")
                        _ = self.finalize_model(discoverySpace)
                        break

                    logger_trim_sampler.info(
                        f"Building and evaluating a predictive model "
                        f"""that includes {batchsize} more {"entities" if batchsize>1 else "entity"} """
                        f"in the training set. Entities are:"
                    )
                    logger_trim_sampler.info(entities)

                    current_source_df, _current_batch_size_target_df = (
                        get_source_and_target(
                            discoverySpace,
                            self.params.targetOutput,
                            discoverySpaceManager=stateHandle,
                        )
                    )

                    if i == 0:
                        previous_source_df = current_source_df
                        train_df = current_source_df
                        logger_trim_sampler.debug(
                            "During the initial iterations the holdout is empty"
                        )
                        logger_trim_sampler.info(f"Yielding {len(entities)} entity")
                        yielded_entities += entities
                        yield entities

                        # NOTE I'm in iterator_closure and stateHandle appears in
                        # async def iterator_closure(
                        #     stateHandle: DiscoverySpaceManager,  # type: ignore[name-defined]
                        # )
                        # TODO: implement MJ sol:
                        # while stateHandle... != ...
                        # sleep(0.1)

                        continue

                    # since we iterate for i in range(0, numberEntities, batchsize)
                    # we know for sure that at every i!=0 we will build a model and a holdout set
                    # Initializing holdout set, -1 because i starts from zero and we know for sure that batchsize divides iteration size
                    elif i < self.params.iterationSize:
                        # TODO: separate logging logic
                        if len(current_source_df) != len(previous_source_df) + 1:
                            logger_trim_sampler.error(
                                f"ANOMALY. Initial source df has length = {len(initial_source_df)}"
                                f"While the current one, before splitting and obtaining the first holdout has length = {len(current_source_df)} "
                            )
                            raise ValueError(
                                f"The size of the source space did not increase by {batchsize}!"
                            )

                        logger_trim_sampler.debug(
                            f"longer_df_from_which_you_subtract has len = {len(current_source_df)}"
                        )
                        logger_trim_sampler.debug(
                            f"longer_df_from_which_you_subtract has len = {len(previous_source_df)}"
                        )
                        # ---------------------------

                        compare_to_previous_source_df, one_additional_row = (
                            split_common_and_diff(
                                longer_df_from_which_you_subtract=current_source_df,
                                shorter_df_that_you_subtract=previous_source_df,
                            )
                        )

                        yielded_rows += one_additional_row

                        # TODO: separate logging logic
                        if not compare_to_previous_source_df.equals(previous_source_df):
                            logger_trim_sampler.setLevel(logging.DEBUG)
                            logger_trim_sampler.error(
                                f"Unexpected behaviour of dfs, logger set to debug level, and saving data in {self.params.debugDirectory}"
                            )
                            compare_to_previous_source_df.to_csv(f"Mismatch_{i}.csv")
                            previous_source_df.to_csv(f"Mismatch_{i-1}.csv")

                        if len(one_additional_row) != 1:
                            logger_trim_sampler.setLevel(logging.DEBUG)
                            logger_trim_sampler.error(
                                f"{len(one_additional_row)} point(s) sampled (expected 1), logger set to debug level, and saving data in {self.params.debugDirectory}"
                            )
                            one_additional_row.to_csv(f"one_additional_row_{i}.csv")
                        # I STILL DO NOT BUILD MODELS

                        # _________________________

                        # TODO: implement MJ sol
                        yield entities
                        yielded_entities += entities  # TODO: data structure
                        continue
                        logger_trim_sampler.debug("Sleeping")
                        await asyncio.sleep(5)

                    elif i == self.params.holdoutSize:
                        train_df, current_holdout_df = split_common_and_diff(
                            longer_df_from_which_you_subtract=current_source_df,
                            shorter_df_that_you_subtract=initial_source_df,
                        )
                        previous_holdout_df = current_holdout_df

                        # TODO: separate logging logic
                        logger_trim_sampler.debug(
                            f"First holdout set created, it contains the following {len(current_holdout_df)} rows:"
                        )
                        logger_trim_sampler.debug(current_holdout_df)
                        if current_holdout_df.empty:
                            logger_trim_sampler.error("Empty Holdout Dataset!")
                            raise NotImplementedError
                        if len(current_holdout_df) != self.params.holdoutSize:
                            logger_trim_sampler.error(
                                f"The holdout df contains {len(current_holdout_df)} rows (expected { self.params.holdoutSize})"
                            )
                        # TODO: check that every row of yielded_rows is also in current_holdout_df
                        #  current_holdout_df
                        # Assumes both DFs have the same columns (names). Order can differ.

                        same = yielded_rows.df.columns.equals(
                            current_holdout_df.columns
                        ) and yielded_rows.df.value_counts(dropna=False).equals(
                            current_holdout_df.value_counts(dropna=False)
                        )  # True if they contain exactly the same rows (multiset equality), regardless of order
                        if not same:
                            logger_trim_sampler.error("Data mismatch")
                            logger_trim_sampler.setLevel(logging.DEBUG)
                            logger_trim_sampler.error(
                                f"Unexpected behaviour of holdout dfs, logger set to debug level, and saving data in {self.params.debugDirectory}"
                            )
                            yielded_rows.df.to_csv(f"Mismatch_yielded_rows_{i}.csv")
                            current_holdout_df.to_csv(
                                f"Mismatch_current_holdout_df_{i}.csv"
                            )

                    else:
                        train_df, one_additional_row = split_common_and_diff(
                            longer_df_from_which_you_subtract=current_source_df,
                            shorter_df_that_you_subtract=previous_source_df,
                        )
                        yielded_rows += one_additional_row
                        current_holdout_df = pd.DataFrame(yielded_rows.df)

                        if len(one_additional_row) != 1:
                            logger_trim_sampler.setLevel(logging.DEBUG)
                            logger_trim_sampler.error(
                                f"{len(one_additional_row)} point(s) sampled (expected 1), logger set to debug level, and saving data in {self.params.debugDirectory}"
                            )
                            one_additional_row.to_csv(f"one_additional_row_{i}.csv")

                        if (
                            len(current_source_df)
                            != len(previous_source_df) + batchsize
                        ):
                            logger_trim_sampler.warning(
                                f"Length of source df at iter {i}: {len(current_source_df)}"
                                f"It is NOT 1 unit greater than length of source df for {i} - {batchsize}: {len(previous_source_df)}"
                            )

                        if current_holdout_df.equals(previous_holdout_df):
                            logger_trim_sampler.warning(
                                "Holdout dataframe is not changing!"
                            )

                    # we eventually save data to debug easier
                    if logger_trim_sampler.isEnabledFor(logging.DEBUG):
                        current_source_df.to_csv(
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
                        holdout_name = (
                            f"holdout_at_iter_{i}.csv"
                            if i != 0
                            else f"empty_holdout_at_iter_{i}.csv"
                        )
                        current_holdout_df.to_csv(
                            os.path.join(self.params.debugDirectory, holdout_name),
                            index=False,
                        )

                    ##############  MODEL BUILDING AND EVALUATION  #####################
                    # ensure we only train on rows where the target is measured
                    if not train_df.equals(
                        train_df.dropna(subset=[str(self.params.targetOutput)])
                    ):
                        logger_trim_sampler.warning(
                            "There are rows in train df where the target is NaN! Dropping them now.\n\n"
                        )
                        train_df = train_df.dropna(subset=[self.params.targetOutput])

                    if train_df.empty:
                        logger_trim_sampler.warning(
                            "Empty training df, this means either that the operation configuration or \
                        the inference problem is ill posed"
                        )
                        raise ValueError

                    # we rename appropriately
                    previous_source_df = current_source_df
                    previous_holdout_df = current_holdout_df

                    train_data = TabularDataset(train_df)
                    holdout_data = TabularDataset(current_holdout_df)

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
                    logger_trim_sampler.info(
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

                    metric_batch_size_dict[i] = {
                        "metric": training_metric,
                        "best_model": best_model_name,
                        "best_score_val": best_score_val,
                        "holdout_score": predictor.evaluate(holdout_data, silent=True)[
                            predictor.eval_metric.name
                        ],
                    }

                    log_metric_string = f"""[Batch under consideration: {i}] Training metric: {training_metric};
                    Best model: {best_model_name}; score_val: {best_score_val}; holdout_score: {metric_batch_size_dict[i]['holdout_score']}"""
                    logger_trim_sampler.info(log_metric_string)

                    # Capture model path and delete the folder
                    if not logger_trim_sampler.isEnabledFor(logging.DEBUG):
                        model_dir = getattr(predictor, "path", None)
                        logger_trim_sampler.info(
                            f"AutoGluon model directory: {model_dir}"
                        )
                        del predictor
                        delete_dir(model_dir=model_dir)

                    # Use the best validation score captured earlier as the "mean ratio" proxy for stopping
                    _metric_entry = metric_batch_size_dict.get(i, {})
                    _best_score_val = _metric_entry.get("holdout_score", None)

                    should_stop = 0

                    # for the first 2*iterationSize we do not have enough data to compare
                    # remember, batchSize divides iterationSize
                    if i < self.params.iterationSize * 2:
                        continue

                    # # comparison happens at every params.iterationSize steps
                    # elif comparison_indeces:
                    #     if max(comparison_indeces) + self.params.iterationSize > i:
                    #         continue  # next iteration of the for, where I will sample another point

                    else:
                        comparison_indices.append(i)
                        # NOTE: if batchsize==iterationSize will compare just two models,
                        # one model from prev_iter_list_range, whose len would be 1, and
                        # one model from this_iter_list_range, whose len would be 1
                        _prev_iter_list_range = list(
                            range(
                                i
                                - self.params.iterationSize * 2
                                + 1,  # this index might be included
                                i
                                - self.params.iterationSize
                                + 1,  # this index cannot be included
                            )
                        )
                        _this_iter_list_range = list(
                            range(
                                i - self.params.iterationSize + 1,
                                i
                                + 1,  # this index cannot be included, but i can be included (this is desired)
                            )
                        )
                        # I filter these to keep only points that I know correspond to models
                        prev_iter_list_range = [
                            i
                            for i in _prev_iter_list_range
                            if i in list(range(0, numberEntities, batchsize))
                        ]
                        this_iter_list_range = [
                            i
                            for i in _this_iter_list_range
                            if i in list(range(0, numberEntities, batchsize))
                        ]

                        logger_trim_sampler.info(
                            f"""Since iterationSize is {self.params.iterationSize}. We now
                            compare models at the following batch indices\n{prev_iter_list_range}\nand\n{this_iter_list_range}"""
                        )

                        scores_previous_iteration = [
                            metric_batch_size_dict[el]["best_score_val"]
                            for el in prev_iter_list_range
                        ]
                        scores_this_iteration = [
                            metric_batch_size_dict[el]["best_score_val"]
                            for el in this_iter_list_range
                        ]

                        logger_trim_sampler.info(
                            f"Scores that correspond to these i-ranges are:\n{prev_iter_list_range}\nand\n{this_iter_list_range}"
                        )

                        try:
                            mean_ratio = (
                                np.array(scores_this_iteration).mean()
                                / np.array(scores_previous_iteration).mean()
                            )
                            if (
                                np.array(scores_previous_iteration).std()
                                * np.array(scores_this_iteration).std()
                                != 0
                            ):
                                logger_trim_sampler.info(
                                    "Product of standard deviation of the scores across batches is 0."
                                    "Setting the ratio to 0"
                                )
                                std_ratio = 0
                                if self.params.batchSize != self.params.iterationSize:
                                    logger_trim_sampler.warning(
                                        "This is a suspicious behavior since the iteration size is differenet from the batch size"
                                    )

                            else:
                                std_ratio = (
                                    np.array(scores_this_iteration).std()
                                    / np.array(scores_previous_iteration).std()
                                )
                        except Exception as e:
                            logger_trim_sampler.warning(
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
                        # Stopping info
                        self.params.finalModelAutoGluonArgs.tabularPredictorArgs[
                            "path"
                        ] = (
                            self.params.finalModelAutoGluonArgs.tabularPredictorArgs.get(
                                "path", self.params.outputDirectory
                            )
                            + "_finalized"
                        )
                        final_model_path = (
                            self.params.finalModelAutoGluonArgs.tabularPredictorArgs[
                                "path"
                            ]
                        )

                        stop_info = f"""Stopping criteria hit after measuring {i} entities.
                                        On a iteration of batch size {self.params.iterationSize}.
                                        Performance of the model on the holdout set
                                        {final_model_path}:\nmean: {_best_score_val}\t\tstd: {std_ratio}\n.
                                        """
                        logger_trim_sampler.info(stop_info)
                        _predictor = self.finalize_model(discoverySpace=discoverySpace)
                        break

                    else:
                        yield_log_string = f"Stopping not triggered for i={i}"
                        logger_trim_sampler.info(yield_log_string)

                        # TODO: Check if this is stable without try statement
                        # try:
                        logger_trim_sampler.info(
                            "Entities yielded in this iteration are:\n"
                        )
                        for e in entities:
                            logger_trim_sampler.info(e)

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

    def finalize_model(self, discoverySpace):
        # FIT ON FULL SOURCE SPACE DATA
        source_df, _target_df = get_source_and_target(
            discoverySpace, self.params.targetOutput
        )
        logger_trim_sampler.info(
            f"Finalizing the predictive model:"
            f"Fitting AutoGluon TabularPredictor on full Source Space data."
            f"Model will be saved in: {self.params.finalModelAutoGluonArgs.tabularPredictorArgs['path']}"
        )

        train_cols = [
            cp.identifier for cp in discoverySpace.entitySpace.constitutiveProperties
        ]
        train_target_cols = [*train_cols, self.params.targetOutput]

        train_df = source_df[train_target_cols]
        # think about replicating here the guardrail about NaN in target
        if logger_trim_sampler.isEnabledFor(logging.DEBUG):
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
            **(self.params.finalModelAutoGluonArgs.tabularPredictorArgs or {}),
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
        training_metric = getattr(predictor, "eval_metric", None)
        save_info = f"""Model finalized using as training set all sampled points, of len {len(train_data)}.\n
                        Final model {training_metric}={final_model_metric}.
                        Saving predicted model to: {self.params.finalModelAutoGluonArgs.tabularPredictorArgs['path']}.
                        """
        logger_trim_sampler.info(save_info)
        return predictor

    def entities_for_iterative_modeling_from_discovery_space(
        self, discoverySpace: DiscoverySpace, discoverySpaceManager=None
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
            discoverySpace,
            self.params.targetOutput,
            discoverySpaceManager=discoverySpaceManager,
        )

        if logger_trim_sampler.isEnabledFor(logging.DEBUG):
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
            logger_trim_sampler.error(msg)
            raise InsufficientDataError(msg)

        if len(source_df) < self.params.samplingBudget.minPoints:
            info_str = """This may happen because it may be that the target variable cannot be measured for all
            the entities in the space. For example a recommender could be unable to recommend the target variables
            for some entities"""
            missing_points = self.params.samplingBudget.minPoints - len(source_df)
            logger_trim_sampler.error(
                f"Insufficient data: need {self.params.samplingBudget.minPoints}, but only {len(source_df)} available. "
                f"Consider adding {missing_points} more points or adjusting the budget."
            )
            logger_trim_sampler.info(info_str)
            if len(source_df) > 10:
                logger_trim_sampler.warning(
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

        if logger_trim_sampler.isEnabledFor(logging.DEBUG):
            merged_df.to_csv(
                os.path.join(self.params.debugDirectory, "initial_debug_merged.csv")
            )
            source_df.to_csv(
                os.path.join(self.params.debugDirectory, "initial_debug_source.csv")
            )
            target_df.to_csv(
                os.path.join(self.params.debugDirectory, "initial_debug_target.csv")
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
        list_of_entities_identifiers = df_ordered_to_sample["identifier"]
        list_of_entities = get_list_of_entities_from_df_and_space(
            df=df_ordered_to_sample, space=discoverySpace
        )

        if logger_trim_sampler.isEnabledFor(logging.DEBUG):
            ordered_df_path_and_name = os.path.join(
                self.params.debugDirectory, "df_ordered_to_sample_with_id.csv"
            )
            ordered_data_log_string = f"DataFrame successfully ordered, saving it now to {ordered_df_path_and_name}"
            logger_trim_sampler.info(ordered_data_log_string)
            list_of_entities_identifiers_log = f"""Ordered list of inferred entities identifiers is:\n{list_of_entities_identifiers}\n\n
            Proceeding to sample entities in this order.\n
            Valid entities are built and validated using the dataframe contained in {ordered_df_path_and_name}"""
            logger_trim_sampler.info(list_of_entities_identifiers_log)
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
