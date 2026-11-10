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

from ado.core.discoveryspace.space import DiscoverySpace
from ado.schema.entity import entity_identifier_from_properties_and_values
from ado.schema.virtual_property import PropertyAggregationMethodEnum

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ado.metastore.project import ProjectContext
    from ado.schema.entity import Entity

logger = logging.getLogger(__name__)


# ============================================================================
# 1D Sampling Functions
# ============================================================================


def _select_least_dense_index(
    occupancy: np.ndarray,
    length_segment: int,
    max_half_width: int,
) -> int:
    """Return the unsampled index lying in the least densely sampled neighbourhood.

    For each unsampled index ``i`` in ``[0, length_segment)`` the density at
    half-width ``d`` is the fraction of sampled positions in the window
    ``[i - d, i + d]`` clipped to ``[0, len(occupancy) - 1]``. Candidates are
    narrowed to those of minimal density for ``d = 1, 2, ...`` until one
    candidate is left or ``d`` reaches ``max_half_width``; the smallest
    remaining index is returned.

    Args:
        occupancy: Array with 1 at sampled positions and 0 elsewhere.
        length_segment: Number of positions eligible for selection.
        max_half_width: Largest window half-width considered.

    Returns:
        The selected index.
    """
    last = len(occupancy) - 1
    prefix = np.concatenate(([0], np.cumsum(occupancy)))
    candidates = np.flatnonzero(occupancy[:length_segment] == 0)

    d = 1
    while True:
        left = np.maximum(candidates - d, 0)
        right = np.minimum(candidates + d, last)
        density = (prefix[right + 1] - prefix[left]) / (right - left + 1)
        candidates = candidates[density == density.min()]
        if len(candidates) == 1 or d >= max_half_width:
            return int(candidates[0])
        d += 1


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

    # Both extremities are sampled here, so the occupancy array spans at least
    # the whole segment; it also covers any out-of-segment sampled index.
    occupancy_length = max(index_list) + 1
    occupancy = np.zeros(occupancy_length, dtype=np.int64)
    occupancy[[i for i in sampled_set if 0 <= i < occupancy_length]] = 1

    while len(index_list) < tot_points_to_sample:
        chosen = _select_least_dense_index(
            occupancy=occupancy,
            length_segment=length_segment,
            max_half_width=length_segment,
        )
        if verbose:
            logger.info(f"Appending {chosen} to {index_list}")
        index_list.append(chosen)
        sampled_set.add(chosen)
        occupancy[chosen] = 1

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
    import ado.cli.core.config

    ado_configuration = ado.cli.core.config.AdoConfiguration.load()
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

    Args:
        discoverySpace: Discovery space, or the identifier of a stored one.

    Returns:
        DataFrame with columns: ['identifier', <constitutive properties>]

    Raises:
        ValueError: If the entity space has a non-discrete dimension.
    """
    space = get_space(space_or_space_id=discoverySpace)
    entity_space = space.entitySpace
    cp_ids = [cp.identifier for cp in entity_space.constitutiveProperties]

    rows: list[dict[str, Any]] = []
    for point_values in entity_space.sequential_point_iterator():
        point = dict(zip(cp_ids, point_values, strict=True))
        rows.append(
            {"identifier": entity_identifier_from_properties_and_values(point), **point}
        )

    return pd.DataFrame(rows, columns=["identifier", *cp_ids])


def _resolve_target_columns(
    df: pd.DataFrame,
    target_outputs: list[str],
    aggregation_method: PropertyAggregationMethodEnum,
) -> tuple[pd.DataFrame, list[str]]:
    """Expose each target output of a target-property table under its bare identifier.

    In a table built with an aggregation method, a target is stored in the
    column ``"<target>-<method>"`` for entities whose values were aggregated and
    in the column ``"<target>"`` for entities whose values could not be
    aggregated. The two columns are merged into ``"<target>"``, taking the
    aggregated value where present.

    Args:
        df: Target-property table.
        target_outputs: Bare target property identifiers to expose.
        aggregation_method: Aggregation method used to build ``df``.

    Returns:
        Tuple of the DataFrame with merged target columns and the target
        outputs present in it, in the order of ``target_outputs``.
    """
    df = df.copy()
    found: list[str] = []
    for target in target_outputs:
        aggregated_column = f"{target}-{aggregation_method.value}"
        if aggregated_column in df.columns:
            aggregated = df.pop(aggregated_column)
            df[target] = (
                aggregated.combine_first(df[target])
                if target in df.columns
                else aggregated
            )
        if target in df.columns:
            found.append(target)
    return df, found


def get_df_at_least_one_measured_value(
    discoverySpace: DiscoverySpace | str,
    targetOutput_list: list[str] | None = None,
    add_measurement_id: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of entities with at least one measured target output.

    Args:
        discoverySpace: Discovery space, or the identifier of a stored one.
        targetOutput_list: Bare target property identifiers to include.
        add_measurement_id: If True, include the entity 'identifier' column.

    Returns:
        DataFrame with columns: ['identifier' (optional), <constitutive properties>, <target outputs>]
    """
    if not targetOutput_list:
        targetOutput_list = []
    space = get_space(space_or_space_id=discoverySpace)
    col_list = [cp.identifier for cp in space.entitySpace.constitutiveProperties]
    if add_measurement_id:
        col_list = ["identifier", *col_list]

    # The sample store caches entities; refresh picks up measurements added
    # since the last query.
    space.sample_store.refresh()

    aggregation_method = PropertyAggregationMethodEnum.mean
    df = pd.DataFrame(
        space.matchingEntitiesTable(
            property_type="target",
            aggregationMethod=aggregation_method,
        )
    )

    if df.empty:
        logger.warning(
            "No measured properties found in the discovery space\nReturning empty DataFrame\n "
        )
        return df

    df, valid_targetOutput_list = _resolve_target_columns(
        df=df,
        target_outputs=targetOutput_list,
        aggregation_method=aggregation_method,
    )
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

    Args:
        discoverySpace: Discovery space, or the identifier of a stored one.
        targetOutput: Bare target property identifier.
        log_string: Text added to the debug log message.

    Returns:
        Tuple of (source_df, target_df). source_df holds the entities with a
        measured targetOutput; target_df holds the remaining entities of the
        space, without the targetOutput column.
    """
    dfm = get_df_at_least_one_measured_value(discoverySpace, [targetOutput])
    dfu = get_df_all_entities_no_measurements(discoverySpace)
    keys = [c for c in dfu.columns if c in dfm.columns and c != "identifier"]

    if dfm.empty:
        logger.warning("The source space is empty")
        return dfm, dfu

    df = dfu.merge(dfm, on=keys, how="left")

    if targetOutput not in df.columns:
        logger.info(
            f"'{targetOutput}' has never been measured in this space. "
            "Adding an empty column to the DataFrame."
        )
        df[targetOutput] = pd.NA

    df_measured = df.dropna(subset=[targetOutput])
    df_unmeasured = df[df[targetOutput].isna()].drop(columns=[targetOutput])
    logger.debug(
        f"Dropped {len(df) - len(df_measured)} rows. "
        f"Function called with log_string={log_string}"
    )
    if df_measured.empty:
        logger.warning(
            f"Empty source after dropping rows that contain Nan in {targetOutput} column"
        )
    if df_unmeasured.empty:
        logger.warning(
            f"Empty target after filtering rows that contain Nan in {targetOutput} column"
        )
    return df_measured, df_unmeasured


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
    from ado.schema.point import SpacePoint

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
