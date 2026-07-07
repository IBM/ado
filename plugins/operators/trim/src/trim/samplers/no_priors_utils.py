# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""
Utility functions for no-priors sampling, including:
- High-dimensional sampling strategies (CLHS, Sobol, random)
- DataFrame ordering and index mapping
- Entity/point conversion and validation
- Discovery space data extraction
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.schema.virtual_property import PropertyAggregationMethodEnum

if TYPE_CHECKING:
    from collections.abc import Hashable

    from orchestrator.metastore.project import ProjectContext
    from orchestrator.schema.entity import Entity

logger = logging.getLogger(__name__)


# ============================================================================
# 1D Sampling Functions
# ============================================================================


def get_index_list_van_der_corput(
    length_segment: int,
    tot_points_to_sample: int,
    sampled_indices: list[int] | None = None,
    sort: bool = False,
    verbose: bool = False,
) -> list[int]:
    """
    Selects indices from a 1D segment using a modified Van der Corput sequence.

    Args:
        length_segment: Total number of units in the 1D segment
        tot_points_to_sample: Total number of indices to sample
        sampled_indices: List of indices already sampled
        sort: If True, returns the final list sorted
        verbose: If True, prints debug information

    Returns:
        List of sampled indices

    Raises:
        ValueError: If tot_points_to_sample exceeds length_segment
    """
    if tot_points_to_sample == 0:
        return []

    if tot_points_to_sample > length_segment:
        raise ValueError(
            "ValueError: You are trying to sample more points than those that are available"
        )

    if sampled_indices is None:
        sampled_indices = []

    if len(sampled_indices) == length_segment:
        maximal_indices_list = list(range(length_segment))
        if sorted(sampled_indices) != maximal_indices_list:
            logging.error(
                "Sampled indices do not correspond to [0,..., max_n_indices -1]. "
                "Returning list(range(max_n_indices))"
            )
        return maximal_indices_list

    if len(sampled_indices) > tot_points_to_sample:
        logging.warning(
            "Number of sampled indices is greater than the number of indices you want to sample"
            "Returning sampled indices"
        )
        return sampled_indices

    index_list = list(sampled_indices)
    sampled_set = set(index_list)

    for point in [0, length_segment - 1]:
        if point not in sampled_set:
            index_list.append(point)
            sampled_set.add(point)
            if len(index_list) == tot_points_to_sample:
                return sorted(index_list)

    def build_prefix_and_len(index_list: list[int]) -> tuple[list[int], int]:
        if not index_list:
            return [0], 0

        M = max(index_list) + 1
        sampled_set = set(index_list)
        prefix = [0] * (M + 1)
        s = 0

        for i in range(M):
            s += 1 if i in sampled_set else 0
            prefix[i + 1] = s

        return prefix, M

    def get_list_min_weight(
        prefix: list[int], M: int, d: int, selectable_indices: list[int]
    ) -> list[int]:
        vals = {}
        for i in selectable_indices:
            if i >= M:
                break
            left = max(0, i - d)
            right = min(M - 1, i + d)
            total = prefix[right + 1] - prefix[left]
            denom = right - left + 1
            mean = total / denom
            vals[i] = mean

        if not vals:
            return []

        min_val = min(vals.values())
        out = []
        for i in selectable_indices:
            if i >= M:
                break
            if vals.get(i) == min_val:
                out.append(i)
        return out

    def get_selectable_indices() -> list[int]:
        return [i for i in range(length_segment) if i not in sampled_set]

    max_d = length_segment

    while len(index_list) < tot_points_to_sample:
        selection = 0
        selectable_indices = get_selectable_indices()
        prefix, M = build_prefix_and_len(index_list=index_list)
        d = 1
        previous_set = selectable_indices

        while selection == 0:
            indices = get_list_min_weight(prefix, M, d, selectable_indices)

            if not indices:
                if not previous_set:
                    raise ValueError(
                        "Previous candidate set should not be empty or None"
                    )
                if verbose:
                    logger.info(
                        f"No intersection found with d={d}. Using the previous set "
                        f"Appending to {index_list} the first element of {previous_set}"
                    )
                chosen = previous_set[0]
                index_list.append(chosen)
                sampled_set.add(chosen)
                selection = 1
            else:
                previous_set = selectable_indices
                selectable_indices = indices

                if len(selectable_indices) == 1 or d == max_d:
                    if verbose:
                        logger.info(
                            f"Appending to {index_list} the first element of {selectable_indices}"
                        )
                    chosen = selectable_indices[0]
                    index_list.append(chosen)
                    sampled_set.add(chosen)
                    selection = 1

            d += 1

    if sort:
        return sorted(index_list)
    return index_list


# ============================================================================
# High-Dimensional Sampling Functions
# ============================================================================


def concatenated_latin_hypercube_sampling(
    dimensions: list[int],
    final_sample_size: int,
    seed: int | None = None,
) -> list[list[int]]:
    """
    Generates samples using Concatenated Latin Hypercube Sampling.

    Args:
        dimensions: Cardinality (size) of each dimension
        final_sample_size: Total number of points to sample
        seed: Optional PRNG seed for reproducibility

    Returns:
        List of sampled points

    Raises:
        ValueError: If any dimension size is less than 1
    """
    if any(d <= 0 for d in dimensions):
        raise ValueError(
            f"All dimensions must be >= 1, received dimensions={dimensions}"
        )

    if final_sample_size <= 0:
        return []

    rng = random.Random() if seed is None else random.Random(seed)  # noqa: S311
    pools: list[list[int]] = [list(range(d)) for d in dimensions]
    samples: list[list[int]] = []

    for _ in range(final_sample_size):
        point: list[int] = []
        for j, d in enumerate(dimensions):
            if not pools[j]:
                pools[j] = list(range(d))
            k = rng.randrange(len(pools[j]))
            value = pools[j].pop(k)
            point.append(value)
        samples.append(point)

    return samples


def sobol_sampling(
    dimensions: list[int], final_sample_size: int, seed: int | None = None
) -> list[list[int]]:
    """
    Generates Sobol sampled points scaled to integer dimensions.

    Falls back to CLHS if collisions are detected.

    Args:
        dimensions: Size of each dimension
        final_sample_size: Number of points to sample
        seed: Random seed for the Sobol scrambler

    Returns:
        List of sampled points
    """
    sampler = Sobol(d=len(dimensions), scramble=True, rng=seed)
    points = sampler.random(final_sample_size)

    discrete_points = [
        [int(val * d) for val, d in zip(p, dimensions, strict=True)] for p in points
    ]

    unique_points = {tuple(p) for p in discrete_points}
    n_collisions = final_sample_size - len(unique_points)

    if n_collisions > 0:
        logger.error(
            f"Sobol sampling failed, {n_collisions} collisions detected, defaulting to clhs sampling"
        )
        return concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=final_sample_size, seed=seed
        )

    return discrete_points


def random_high_dimensional_sampling(
    dimensions: list[int], final_sample_size: int, seed: int | None = None
) -> list[list[int]]:
    """
    Generate unique random samples from a high-dimensional space.

    Args:
        dimensions: Cardinality of each dimension
        final_sample_size: Total number of points to sample
        seed: Optional PRNG seed

    Returns:
        List of sampled points

    Raises:
        ValueError: If final_sample_size exceeds total configurations
    """
    if seed is not None:
        random.seed(seed)

    num_configs = math.prod(dimensions)
    if final_sample_size > num_configs:
        raise ValueError(
            f"Cannot generate {final_sample_size} unique samples. "
            f"The sample space only contains {num_configs} possibilities."
        )

    configs = list(itertools.product(*[range(d) for d in dimensions]))
    actual_sample_size = min(final_sample_size, len(configs))

    if actual_sample_size < final_sample_size:
        logger.warning(
            f"Requested {final_sample_size} samples but only {len(configs)} unique "
            f"configurations available. Sampling {actual_sample_size} instead."
        )

    samples = random.sample(configs, actual_sample_size)
    return [list(s) for s in samples]


def get_sampling_indices_multi_dimensional(
    dimensions: list[int],
    n: int | Literal["all", "max"],
    space: dict[str, int] | None = None,
    strategy: Literal["random", "clhs", "sobol"] = "clhs",
    seed: int | None = None,
) -> list[list[int]]:
    """
    Generate sampling indices for a high-dimensional space.

    Args:
        dimensions: Sizes of each dimension
        n: Number of points to sample ('all', 'max', or integer)
        space: Optional mapping of dimension names to sizes
        strategy: Sampling strategy ('random', 'clhs', or 'sobol')
        seed: Controls randomness

    Returns:
        List of sampled multi-dimensional coordinates
    """
    if seed is not None:
        random.seed(seed)

    if space:
        indices_dict = {
            k: get_index_list_van_der_corput(v, v) for k, v in space.items()
        }
        if [len(indices) for indices in list(indices_dict.values())] != dimensions:
            logger.error(
                f"A space dict has been provided ->{space}. It is inconsistent with dimensions={dimensions}"
            )
            raise ValueError("Space has inconsistent dimensions!")
        logger.info(
            "Sampling indices for each named dimension (ordered low to high): %s",
            indices_dict,
        )

    orders = [get_index_list_van_der_corput(v, v) for v in dimensions]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Dimensions: %s", dimensions)
        logger.debug("Sampling orders for each dimension:")
        for i, o in enumerate(orders):
            logger.debug("Dimension %d order: %s", i, o)

    maximum_n = math.prod(dimensions)
    lcm = math.lcm(*dimensions)

    if lcm != maximum_n:
        logger.debug(
            "Periodicity detected, the sampling subroutine will ensure that you will not sample"
            "the same configuration more than once."
        )

    if isinstance(n, str):
        if n == "all":
            n = maximum_n
        elif n == "max":
            n = max(dimensions)
        else:
            raise ValueError(f"Unrecognized string for n: {n}")

    if n > maximum_n:
        logger.warning(
            f"Maximal sample size is {maximum_n}, you requested {n} sampling prescriptions."
            f"Elaborating prescription for n_samples = {maximum_n}"
        )

    logger.debug("Preparing to sample %d out of %d possible points.", n, maximum_n)

    match strategy:
        case "random":
            return random_high_dimensional_sampling(dimensions, n, seed=seed)
        case "clhs":
            return concatenated_latin_hypercube_sampling(
                dimensions=dimensions, final_sample_size=n, seed=seed
            )
        case "sobol":
            return sobol_sampling(dimensions=dimensions, final_sample_size=n, seed=seed)
        case _:
            raise NotImplementedError(f"Strategy {strategy} is unknown")


# ============================================================================
# DataFrame Ordering and Index Mapping
# ============================================================================


def get_index_list_nn_high_dimensional(
    orders_to_sample: list[list[int]], dimensions: list[int]
) -> list[int]:
    """
    Map high-dimensional sampling orders to linear (flattened) indices.

    Args:
        orders_to_sample: List of multi-dimensional coordinates
        dimensions: Size of each dimension

    Returns:
        List of linear indices

    Warns:
        If duplicate or out-of-bounds indices are detected
    """
    indices = []
    cprod = np.cumprod(np.array(dimensions), dtype=int).tolist()
    maximum_n = cprod[-1]

    for order in orders_to_sample:
        index = 0
        multiplier = 1
        for i in reversed(range(len(dimensions))):
            index += order[i] * multiplier
            multiplier *= dimensions[i]

        if index > maximum_n:
            logging.warning(
                f"Out of bound index {index} computed from order {order}, dimensions are {dimensions}"
            )
        indices.append(index)

    if len(set(indices)) != len(indices):
        logger.error(f"{len(indices) - len(set(indices))} Duplicated indices!")

    out_of_bounds_list = [i for i in indices if i > maximum_n]
    if out_of_bounds_list:
        logger.error(
            f"The following indices are out of bound: {out_of_bounds_list}, maximum admissible value is {maximum_n - 1}"
        )

    return indices


def order_df_for_get_index_list_nn_high_dimensional(
    df: pd.DataFrame, constitutive_properties: list[str], dimensions: list[int]
) -> pd.DataFrame:
    """
    Ensure DataFrame is ordered and complete for high-dimensional index generation.

    Args:
        df: Input DataFrame
        constitutive_properties: Column names defining the space
        dimensions: Expected cardinality for each property

    Returns:
        DataFrame sorted and augmented with missing combinations
    """
    df = df.sort_values(by=constitutive_properties).reset_index(drop=True)
    expected_len = math.prod(dimensions)

    if len(df) == expected_len:
        return df

    unique_values = [
        sorted(df[prop].dropna().unique()) for prop in constitutive_properties
    ]
    all_combinations = list(itertools.product(*unique_values))
    actual_expected_len = len(all_combinations)

    logger.warning(
        f"DataFrame length mismatch: expected {expected_len} (product of {dimensions}), "
        f"but got {len(df)}. Actual unique combinations: {actual_expected_len}."
    )

    existing_combinations = {
        tuple(row[prop] for prop in constitutive_properties) for _, row in df.iterrows()
    }

    missing_combinations = [
        comb for comb in all_combinations if comb not in existing_combinations
    ]

    if missing_combinations:
        logger.info(
            f"Injecting {len(missing_combinations)} missing rows to satisfy the property."
        )
        injected_rows = []
        for comb in missing_combinations:
            row_data = dict(zip(constitutive_properties, comb, strict=False))
            for col in df.columns:
                if col not in constitutive_properties:
                    row_data[col] = pd.NA
            injected_rows.append(row_data)

        df = pd.concat([df, pd.DataFrame(injected_rows)], ignore_index=True)
        df = df.sort_values(by=constitutive_properties).reset_index(drop=True)
        logger.info(f"Injected rows: {injected_rows}")

    return df


def order_df_for_sampling_with_no_priors(
    df: pd.DataFrame,
    constitutive_properties: list[str],
    n: int,
    strategy: Literal["random", "clhs", "sobol"],
) -> pd.DataFrame:
    """
    Orders a DataFrame for high-dimensional sampling without prior knowledge.

    Args:
        df: Input dataset
        constitutive_properties: Column names defining the configuration space
        n: Number of samples to generate
        strategy: Sampling strategy

    Returns:
        DataFrame with n sampled rows

    Raises:
        ValueError: If n <= 0 after adjustment or no samples available
    """
    len_original = len(df)
    df_unique = df.drop_duplicates(subset=constitutive_properties).reset_index(
        drop=True
    )
    delta_len = len_original - len(df_unique)
    if delta_len > 0:
        logging.warning(
            f"Removing {delta_len} duplicate configurations."
            f"They are characterized by the same combination of constitutive properties = {constitutive_properties}"
        )

    if n > len(df_unique):
        logging.warning(
            f"Requested {n} samples, but DataFrame has only {len(df_unique)} rows. Adjusting n to {len(df_unique)}."
        )
        n = len(df_unique)

    if n <= 0:
        logging.error(
            f"No samples available to select. DataFrame has {len(df_unique)} rows and {n} samples were requested."
        )
        return pd.DataFrame(columns=df_unique.columns)

    def _get_sorted_uniques(prop: str) -> list:
        vals = df_unique[prop].unique()
        try:
            return sorted(vals)
        except TypeError:
            logging.warning(
                f"Cannot sort mixed types for property '{prop}'. "
                "Keeping original order."
            )
            return list(vals)

    value_dict = {prop: _get_sorted_uniques(prop) for prop in constitutive_properties}
    space_dict = {prop: len(vals) for prop, vals in value_dict.items()}
    dimensions = list(space_dict.values())

    df_unique = order_df_for_get_index_list_nn_high_dimensional(
        df_unique, constitutive_properties, dimensions=dimensions
    ).reset_index(drop=True)

    orders_to_sample = get_sampling_indices_multi_dimensional(
        dimensions=dimensions, space=space_dict, n=n, strategy=strategy
    )

    indices_to_sample = get_index_list_nn_high_dimensional(orders_to_sample, dimensions)

    logger.info(f"Indexes are:\n {indices_to_sample}")
    try:
        return df_unique.iloc[indices_to_sample]
    except IndexError:
        logging.error(
            f"Index Error detected. Length of the dataframe is {len(df_unique)}."
            "The indices that cause the error are:"
        )
        max_len = len(df_unique)
        out_of_bounds_list = [i for i in indices_to_sample if i < 0 or i >= max_len]
        logging.error(out_of_bounds_list)
        logging.error("Returning empty dataset")
        return pd.DataFrame({})


# ============================================================================
# Discovery Space Data Extraction
# ============================================================================


def get_project_context() -> ProjectContext:
    """Retrieve the current ADO project context from configuration."""
    import orchestrator.cli.core.config

    ado_configuration = orchestrator.cli.core.config.AdoConfiguration.load()
    return ado_configuration.project_context  # type: ignore[name-defined]


def get_space(
    space_or_space_id: DiscoverySpace | str,
) -> DiscoverySpace:
    """Get a DiscoverySpace object from either a space object or identifier string."""
    if isinstance(space_or_space_id, DiscoverySpace):
        return space_or_space_id

    return DiscoverySpace.from_stored_configuration(
        project_context=get_project_context(),
        space_identifier=space_or_space_id,
    )


def get_df_all_entities_no_measurements(
    discoverySpace: DiscoverySpace | str,
) -> pd.DataFrame:
    """
    Return a DataFrame of all entities in the Discovery Space.

    Returns:
        DataFrame with columns: ['identifier', <constitutive properties>]
    """
    space = get_space(space_or_space_id=discoverySpace)
    entity_space = space.entitySpace
    cp_ids = [cp.identifier for cp in entity_space.constitutiveProperties]

    list_of_dicts_to_convert = []
    for point_values in entity_space.sequential_point_iterator():
        point_dict = dict(zip(cp_ids, point_values, strict=True))
        entity = entity_space.entity_for_point(point_dict)
        ed = {"identifier": entity.identifier}
        ed.update(point_dict)
        list_of_dicts_to_convert.append(ed)

    return pd.DataFrame(list_of_dicts_to_convert)


def get_df_at_least_one_measured_value(
    discoverySpace: DiscoverySpace | str,
    targetOutput_list: list[str] | None = None,
    add_measurement_id: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of entities with at least one measured target output.

    Returns:
        DataFrame with columns: ['identifier' (optional), <constitutive properties>, <target outputs>]
    """
    if not targetOutput_list:
        targetOutput_list = []
    space = get_space(space_or_space_id=discoverySpace)
    col_list = [cp.identifier for cp in space.entitySpace.constitutiveProperties]
    if add_measurement_id:
        col_list = ["identifier", *col_list]

    discoverySpace.sample_store.refresh()

    df = pd.DataFrame(
        space.matchingEntitiesTable(
            property_type="target",
            aggregationMethod=PropertyAggregationMethodEnum.mean,
        )
    )

    if df.empty:
        logger.warning(
            "No measured properties found in the discovery space\nReturning empty DataFrame\n "
        )
        return df

    all_df_cols = list(df.columns)
    valid_targetOutput_list = []
    for el in targetOutput_list:
        if el in all_df_cols:
            valid_targetOutput_list.append(el)
        elif f"{el}-mean" in all_df_cols and el not in all_df_cols:
            logger.warning(
                f"Column named '{el}-mean' (instead of '{el}', which is not present)"
                "found in the DataFrame obtained through matchingEntitiesTable. "
                f"Renaming it to '{el}'."
            )
            df.rename(columns={f"{el}-mean": el}, inplace=True)
            valid_targetOutput_list += [el]
        elif f"{el}-mean" in all_df_cols and el in all_df_cols:
            logger.warning(
                f"Columns named '{el}-mean' and '{el}'"
                "found in the DataFrame obtained through matchingEntitiesTable. "
                f"Renaming it to '{el}'."
            )
            logger.error("Unexpected behavior can happen!")
            df.rename(columns={f"{el}-mean": el}, inplace=True)
            valid_targetOutput_list += [el]
    col_list += valid_targetOutput_list

    if valid_targetOutput_list != targetOutput_list:
        if len(valid_targetOutput_list) == 0:
            logger.error(
                "No valid target in the columns of the DataFrame."
                f"columns are:\t{list(df.columns)}."
                f"First rows are:\n{df.head(5)}"
            )
        else:
            not_found = [
                t for t in targetOutput_list if t not in valid_targetOutput_list
            ]
            logger.error(
                f"Found measurements for the following valid targets:\t{valid_targetOutput_list}"
            )
            logger.error(
                f"No measurement found for the following valid targets:\t{not_found}"
            )

    removed_cols = [c for c in list(df.columns) if c not in col_list]
    logger.debug(
        "Obtaining df with at least one measured target."
        f"Removed columns: {removed_cols}"
    )

    df = df[col_list]
    df.dropna(inplace=True)

    if df.empty:
        logger.warning(
            "Although there were some measured properties in the discovery space."
        )
        logger.warning(
            "All measured properties in the discovery space"
            f"are different from the desired outputs {targetOutput_list}.Returning empty DataFrame\n "
        )

    return df


def get_source_and_target(
    discoverySpace: DiscoverySpace | str,
    targetOutput: str,
    log_string: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build source (labeled) and target (unlabeled) DataFrames for a target output.

    Returns:
        Tuple of (source_df, target_df)
    """
    dfm = get_df_at_least_one_measured_value(discoverySpace, [targetOutput])
    dfu = get_df_all_entities_no_measurements(discoverySpace)
    keys = [c for c in dfu.columns if c in dfm.columns and c != "identifier"]

    if dfm.empty:
        logger.warning("The source space is empty")
        return dfm, dfu

    df = dfu.merge(dfm, on=keys, how="left")

    if targetOutput not in list(df.columns):
        logger.info(
            f"""The target output was not present in the columns of the measured+unmeasured DataFrame,' \
                        meaning that '{targetOutput}' has never been measured in this space.
                        dfm.empty = {df.empty}. Adding an empty column to the DataFrame.
                    """
        )
        logger.debug("Adding an empty column to the DataFrame.")
        df[targetOutput] = pd.NA

    if targetOutput in list(df.columns):
        df_measured_drop_na = df.dropna(subset=[targetOutput])
        df_unmeasured_drop_na = df[df[targetOutput].isna()].drop(columns=[targetOutput])
        n_rows_dropped = len(df) - len(df_measured_drop_na)
        logger.debug(
            f"Dropped {n_rows_dropped} rows. Function called with log_string={log_string}"
        )
        if df_measured_drop_na.empty:
            logger.warning(
                f"Empty source after dropping rows that contain Nan in {targetOutput} column"
            )
        if df_unmeasured_drop_na.empty:
            logger.warning(
                f"Empty target after filtering rows that contain Nan in {targetOutput} column"
            )
        return df_measured_drop_na, df_unmeasured_drop_na

    save_path = "df_with_no_targetOutput_columns.csv"
    logger.error(
        f"'{targetOutput}' column is missing, saving df in {save_path}, returning unmerged DataFrames"
    )
    df.to_csv(save_path)
    return dfm, dfu


# ============================================================================
# Entity/Point Conversion
# ============================================================================


def validate_points_in_space(
    points: list[dict],
    space: DiscoverySpace,
) -> tuple[list[dict], list[int]]:
    """
    Validate point dictionaries against a Discovery Space.

    Returns:
        Tuple of (valid_points, invalid_indices)
    """
    valid_points: list[dict] = []
    invalid_indices: list[int] = []

    for i, p in enumerate(points):
        if space.entitySpace.isPointInSpace(p):
            valid_points.append(p)
        else:
            invalid_indices.append(i)
    return valid_points, invalid_indices


def df_to_points(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    dropna: bool = True,
    drop_duplicates: bool = False,
) -> list[dict[Hashable, Any]]:
    """
    Convert DataFrame rows to list of point dictionaries.

    Args:
        df: Input DataFrame
        cols: Columns to include
        dropna: If True, drop rows containing NaN
        drop_duplicates: If True, drop duplicate rows

    Returns:
        List of point dictionaries
    """
    if cols is None:
        cols = list(df.columns)
    missing = set(cols) - set(df.columns)
    if missing:
        raise KeyError(f"Requested columns not present in DataFrame: {missing}")

    sub = df[cols].copy()
    if dropna:
        sub = sub.dropna(how="any")
    if drop_duplicates:
        sub = sub.drop_duplicates()

    def to_py(x: object) -> object:
        if isinstance(x, (np.generic)):
            return x.item()
        return x

    for c in sub.columns:
        sub[c] = sub[c].map(to_py)

    return sub.to_dict(orient="records")


def df_to_points_parsing(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    dropna: bool = True,
    parse_values: bool = False,
) -> list[dict]:
    """Convert DataFrame to points with optional string value parsing."""
    import ast

    points = df_to_points(df, cols=cols, dropna=dropna)
    if not parse_values:
        return points

    parsed = []
    for p in points:
        newp = {}
        for k, v in p.items():
            if isinstance(v, str):
                try:
                    newp[k] = ast.literal_eval(v)
                except Exception:
                    newp[k] = v
            else:
                newp[k] = v
        parsed.append(newp)
    return parsed


def make_points_from_df(
    df: pd.DataFrame,
    space: DiscoverySpace,
    cols: list[str] | None = None,
    dropna: bool = True,
    parse_values: bool = True,
) -> list[dict]:
    """
    Convert DataFrame of constitutive properties into point dictionaries.

    Args:
        df: Input DataFrame
        space: Discovery Space providing canonical order
        cols: Explicit list of columns to use
        dropna: If True, drop rows with NaN
        parse_values: If True, parse string values

    Returns:
        List of point dictionaries
    """
    if cols is None:
        cols = [cp.identifier for cp in space.entitySpace.constitutiveProperties]

    missing = set(cols) - set(df.columns)
    if missing:
        raise KeyError(f"Requested columns not present in DataFrame: {missing}")

    return df_to_points_parsing(df, cols=cols, dropna=dropna, parse_values=parse_values)


def get_list_of_entities_from_df_and_space(
    df: pd.DataFrame, space: DiscoverySpace
) -> list[Entity]:
    """
    Convert DataFrame rows to Entity objects validated against a discovery space.

    Args:
        df: DataFrame containing constitutive property values
        space: DiscoverySpace defining the entity space constraints

    Returns:
        List of valid Entity objects
    """
    points = make_points_from_df(df=df, space=space)
    valid_points, __ = validate_points_in_space(points, space)

    list_of_entities = []
    from orchestrator.schema.point import SpacePoint

    for p in valid_points:
        sp = SpacePoint(entity=p)
        entity = sp.to_entity(generatorid="no_priors_characterization")
        list_of_entities.append(entity)

    numberEntities = len(list_of_entities)
    if numberEntities != len(df):
        numberEntities_log = f"""Warning: number of valid entities {numberEntities} is different from the number of rows in the ordered df {len(df)}.
        This means that some rows in the ordered df did not correspond to valid entities in the discovery space.
        """
        logging.warning(numberEntities_log)
    return list_of_entities


# Made with Bob
