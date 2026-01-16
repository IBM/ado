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
from trim.utils.logging_utils import (
    log_after_first_holdout_creation,
    log_after_split_common_and_diff,
    log_before_first_holdout_update,
    save_source_train_holdout_dfs,
    training_guardrail,
)
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
from trim.utils.stopping_criterion import stopping_bool_from_ratios

logger_trim_sampler = logging.getLogger(__name__)
# logger_trim_sampler.setLevel(logging.DEBUG)


# NOTE: to repeat the operation on the same space you can delete the operation
# but first make sure that the output of this operation is not used by another operation
class TrimSampleSelector(BaseSampler):
    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls, remoteDiscoverySpace: DiscoverySpaceManager  # type: ignore[name-defined]
    ) -> bool:
        # do you want to return False if no point has been measured?
        return True

    async def remoteEntityIterator(
        self, remoteDiscoverySpace: DiscoverySpaceManager, batchsize: int = 1  # type: ignore[name-defined]
    ) -> typing.AsyncGenerator[list[Entity], None]:
        """Returns an remoteEntityIterator that returns entities in order"""

        logger_trim_sampler.debug(f"Batchsize is {batchsize} (expected 1)")

        async def iterator_closure(
            stateHandle: DiscoverySpaceManager,  # type: ignore[name-defined]
        ) -> typing.Callable[[], typing.AsyncGenerator[list[Entity], None]]:

            logger_trim_sampler.debug(
                f"Trim starts with parameters:\n{self.params}\n\n"
            )

            if logger_trim_sampler.isEnabledFor(logging.DEBUG):
                debug_dir = Path(self.params.debugDirectory).expanduser().resolve()
                logger_trim_sampler.debug(
                    f"Creating a folder to save intermediate files:\n{debug_dir}\n\n"
                )
                debug_dir.mkdir(parents=True, exist_ok=True)

            discoverySpace = await stateHandle.discoverySpace.remote()
            list_of_entities, _df_ordered_to_sample = (
                self.entities_for_iterative_modeling_from_discovery_space(
                    discoverySpace=discoverySpace
                )
            )
            numberEntities = len(list_of_entities)

            async def iterator() -> typing.AsyncGenerator[list[Entity], None]:  # type: ignore[name-defined][name-defined]
                await asyncio.sleep(0.001)

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
                    f"Trim iterator will measure up to {numberEntities} entities.\n"
                    f"These entities have been ordered using {len(initial_source_df)} measurements from the discovery space."
                )

                logger_trim_sampler.info(
                    f"Training columns are {train_cols},\nThe dependent variable (target Output) is {train_target_cols[-1]}"
                )

                ############################################################################################################
                ######################################### MAIN LOOP STARTS #################################################
                ############################################################################################################

                metric_dict = {}
                comparison_indices = []
                previous_holdout_df = pd.DataFrame({})
                # Ring-like data structures
                yielded_entities = deque(maxlen=self.params.holdoutSize)
                yielded_rows = RowsRing(
                    maxlen=(
                        self.params.holdoutSize
                        if self.params.holdoutSize
                        else self.params.iterationSize
                    )
                )
                for i in range(0, numberEntities, batchsize):
                    entity = list_of_entities[i : i + batchsize]

                    if len(entity) == 0:
                        logger_trim_sampler.warning("No Entities remaining.")
                        _ = self.finalize_model(
                            discoverySpace, discoverySpaceManager=stateHandle
                        )
                        break

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
                        logger_trim_sampler.info(
                            f"Yielding {len(entity)} entity, which is {entity}"
                        )
                        yielded_entities += entity
                        yield entity
                        continue

                    # TODO: the first holdout set can also be obtained from the source space
                    # atm we sample new points from the target and put these into the holdout
                    # we can instead look at the source at iter=0 and select within this set the best
                    # source and holdout df, the rationale here would be selecting the holdout set first
                    # to prioritize representativeness in the OOS set, and put the remaining points in
                    # the test set
                    elif i < self.params.iterationSize:
                        compare_to_previous_source_df, one_additional_row = (
                            split_common_and_diff(
                                longer_df_from_which_you_subtract=current_source_df,
                                shorter_df_that_you_subtract=previous_source_df,
                            )
                        )
                        log_after_split_common_and_diff(
                            i,
                            compare_to_previous_source_df,
                            previous_source_df,
                            one_additional_row,
                            directory=self.params.debugDirectory,
                        )
                        yielded_rows += one_additional_row
                        yielded_entities += entity
                        previous_source_df = current_source_df
                        logger_trim_sampler.info(
                            f"Yielding {len(entity)} entity, which is {entity}"
                        )
                        yield entity
                        continue

                    elif (
                        i == self.params.iterationSize
                    ):  # at this point we build the first model
                        train_df, current_holdout_df = split_common_and_diff(
                            longer_df_from_which_you_subtract=current_source_df,
                            shorter_df_that_you_subtract=initial_source_df,
                        )
                        _, one_additional_row = split_common_and_diff(
                            longer_df_from_which_you_subtract=current_source_df,
                            shorter_df_that_you_subtract=previous_source_df,
                        )
                        yielded_rows += one_additional_row
                        previous_holdout_df = current_holdout_df

                        log_after_first_holdout_creation(
                            current_holdout_df,
                            yielded_rows,
                            iter_index=i,
                            params=self.params,
                        )

                    else:  # i > self.params.iterationSize
                        train_df, one_additional_row = split_common_and_diff(
                            longer_df_from_which_you_subtract=current_source_df,
                            shorter_df_that_you_subtract=previous_source_df,
                        )

                        log_before_first_holdout_update(
                            one_additional_row,
                            current_source_df,
                            previous_source_df,
                            iter_index=i,
                            debugDirectory=self.params.debugDirectory,
                            batchsize=batchsize,
                        )

                        yielded_rows += one_additional_row
                        current_holdout_df = pd.DataFrame(yielded_rows.df)

                        if current_holdout_df.equals(previous_holdout_df):
                            logger_trim_sampler.warning(
                                "Holdout dataframe is not changing!"
                            )

                    # we rename appropriately
                    previous_source_df = current_source_df
                    previous_holdout_df = current_holdout_df
                    if logger_trim_sampler.isEnabledFor(logging.DEBUG):
                        save_source_train_holdout_dfs(
                            current_source_df=current_source_df,
                            train_df=train_df,
                            current_holdout_df=current_holdout_df,
                            iter=i,
                            directory=self.params.debugDirectory,
                        )

                    ##############  MODEL BUILDING AND EVALUATION  #####################

                    logger_trim_sampler.info(
                        f"Building and evaluating a predictive model "
                        f"""that includes {batchsize} more {"entities" if batchsize>1 else "entity"} """
                        f"in the training set:\n {entity}"
                    )
                    # ensures we only train on rows where the target is measured
                    # TODO: monitor if this is needed
                    train_df = training_guardrail(
                        train_df, targetOutput=self.params.targetOutput
                    )

                    train_data = TabularDataset(train_df)
                    holdout_data = TabularDataset(current_holdout_df)

                    # NOTE: assigning more weight to target space points does NOT generally improve performance
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

                    metric_dict[i] = {
                        "metric": training_metric,
                        "best_model": best_model_name,
                        "best_score_val": best_score_val,
                        "holdout_score": predictor.evaluate(holdout_data, silent=True)[
                            predictor.eval_metric.name
                        ],
                    }

                    logger_trim_sampler.info(
                        f"[Batch under consideration: {i}] Training metric: {training_metric};\n"
                        f"Best model: {best_model_name}; score_val: {best_score_val:.2f}; holdout_score: {metric_dict[i]['holdout_score']:.2f}",
                    )

                    # Capture model path and delete the folder
                    if not logger_trim_sampler.isEnabledFor(logging.DEBUG):
                        model_dir = getattr(predictor, "path", None)
                        logger_trim_sampler.info(
                            f"AutoGluon model directory: {model_dir}"
                        )
                        del predictor
                        delete_dir(model_dir=model_dir)

                    # Use the best validation score captured earlier as the "mean ratio" proxy for stopping
                    _metric_entry = metric_dict.get(i, {})
                    _best_score_val = _metric_entry.get("holdout_score", None)

                    should_stop = 0

                    # for the first 2*iterationSize we do not have enough data to compare
                    # i need to go up to self.params.iterationSize * 3
                    # if I want that I have one iteration size of models already measured:
                    # i<iter_size: no models
                    # itersize =< i< itersize *2 : 1st iter of models
                    # itersize*2 =< i< itersize *3 : 2nd iter of models
                    if i < self.params.iterationSize * 3 - 1:
                        yield entity
                        yielded_entities += entity
                        continue

                    # NOTE: at the moment comparison does NOT happen at every params.iterationSize steps
                    # instead, it happens at every batchsize=1 step, in a rolling fashion,
                    # thus the parameters set for the threshold in trim parameters are not the same as in the paper!
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
                            f"Since iterationSize is {self.params.iterationSize}, "
                            f"We now compare models at the following batch indices\n{prev_iter_list_range}\nand\n{this_iter_list_range}"
                        )

                        scores_previous_iteration = [
                            float(metric_dict[el]["holdout_score"])
                            for el in prev_iter_list_range
                        ]
                        scores_this_iteration = [
                            float(metric_dict[el]["holdout_score"])
                            for el in this_iter_list_range
                        ]

                        logger_trim_sampler.info(
                            f"Scores that correspond to these i-ranges are:\n{scores_previous_iteration}\nand\n{scores_this_iteration}"
                        )

                        try:
                            mean_ratio = (
                                np.array(scores_this_iteration).mean()
                                / np.array(scores_previous_iteration).mean()
                            )
                            if (
                                np.array(scores_previous_iteration).std()
                                * np.array(scores_this_iteration).std()
                                == 0
                            ):
                                logger_trim_sampler.info(
                                    "Product of standard deviation of the scores across batches is 0."
                                    "Setting the ratio to 0"
                                )
                                std_ratio = 0

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
                        logger_trim_sampler.info(
                            f"Testing stopping criterion after measuring {i} points, "
                            "mean_ratio={mean_ratio} and std_ratio={std_ratio}"
                        )
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

                        logger_trim_sampler.info(
                            f"Stopping criteria hit after measuring {i} entities.\n"
                            f"On a iteration of batch size {self.params.iterationSize}.\n"
                            "Performance of the model on the holdout set"
                            f"{final_model_path}:\nmean: {_best_score_val}\t\tstd: {std_ratio}\n."
                        )
                        _predictor = self.finalize_model(
                            discoverySpace=discoverySpace,
                            discoverySpaceManager=stateHandle,
                        )
                        break

                    else:
                        yield_log_string = f"Stopping not triggered for i={i}"
                        logger_trim_sampler.info(yield_log_string)
                        yield entity

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

    def finalize_model(
        self, discoverySpace: DiscoverySpace, discoverySpaceManager: DiscoverySpaceManager  # type: ignore[name-defined]
    ) -> TabularPredictor:
        # FIT ON FULL SOURCE SPACE DATA
        source_df, _target_df = get_source_and_target(
            discoverySpace,
            self.params.targetOutput,
            discoverySpaceManager=discoverySpaceManager,
        )

        # TODO: check why len(source_df) is minor than max(i) of the iterative modeling phase
        logger_trim_sampler.info(
            f"Finalizing the predictive model:"
            f"Fitting AutoGluon TabularPredictor on full Source Space data of {len(source_df)} rows."
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
        logger_trim_sampler.info(
            f"Model finalized using as training set all sampled points, of cardinality {len(train_data)}.\n"
            f"Final model {training_metric}={final_model_metric}."
            f"Saving predicted model to: {self.params.finalModelAutoGluonArgs.tabularPredictorArgs['path']}."
        )
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
            logger_trim_sampler.info(
                f"Ordered list of inferred entities identifiers is:\n{list_of_entities_identifiers}\n"
                "Proceeding to sample entities in this order.\n"
                f"Valid entities are built and validated using the dataframe contained in {ordered_df_path_and_name}"
            )
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
                for _i in range(0, numberEntities, batchsize):
                    # batch = list_of_entities[i : i + batchsize]
                    ...

            return iterator

        retval = iterator_closure(discoverySpace)
        return retval()

    @classmethod
    def parameters_model(cls) -> type[BaseModel] | None:
        return TrimParameters

    def __init__(self, parameters: TrimParameters) -> None:
        self.params = parameters
