# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import operator
import typing
from functools import reduce

import pandas as pd
import ray

from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.metastore.project import ProjectContext
from orchestrator.schema.virtual_property import PropertyAggregationMethodEnum

logger = logging.getLogger(__name__)


def get_project_context() -> "ProjectContext":
    import orchestrator.cli.core.config

    ado_configuration = orchestrator.cli.core.config.AdoConfiguration.load()
    return ado_configuration.project_context  # type: ignore[name-defined]


def get_space(
    space_or_space_id: typing.Union["DiscoverySpace", str],
) -> DiscoverySpace:

    if isinstance(space_or_space_id, DiscoverySpace):
        return space_or_space_id

    return DiscoverySpace.from_stored_configuration(
        project_context=get_project_context(),
        space_identifier=space_or_space_id,
    )


# %%


def get_df_all_entities_no_measurements(
    discoverySpace: typing.Union["DiscoverySpace", str],
    discoverySpaceManager=None,
) -> pd.DataFrame:
    """
    Return a DataFrame of all entities in the given Discovery Space, regardless of whether
    they have any measured target outputs.

    - Each row represents an entity from the entity space.
    - Includes the entity identifier and all constitutive property values.
    - Does NOT include any measured target outputs (only features).
    - Useful for generating the full feature set for prediction or backfilling missing measurements.

    Parameters
    ----------
    discoverySpace : DiscoverySpace | str
        The Discovery Space object or its identifier.
    targetOutput_list : list, optional
        List of target output names (ignored in this function, included for API consistency).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['identifier', <constitutive properties>].
    """

    from orchestrator.core.discoveryspace.samplers import (
        ExplicitEntitySpaceGridSampleGenerator,
        WalkModeEnum,
    )

    space = get_space(space_or_space_id=discoverySpace)

    gen = ExplicitEntitySpaceGridSampleGenerator(WalkModeEnum.SEQUENTIAL)
    # VV: We need a list that contains all entities associated with this Discovery Space in its entity source
    # regardless of whether an experiment (on this space or others) have measured any properties using the
    # experiments that this Discovery Space references.
    # To this end, we find all matching entities in the entity source which have at least 1 measurement from at least
    # one of the experiments that this discovery space defines. We then fill in any missing entities by iteratively
    # generating all the entities that this discovery space contains. These "backfill" entities will not be associated
    # with any observed property

    if discoverySpaceManager:
        logger.info("Using manager in getting entities no measurement")
        all_entities = reduce(
            # operator.add, gen.entitySpaceIterator(entitySpace=discoverySpaceManager.discoverySpace.remote()), []
            operator.add,
            gen.entitySpaceIterator(
                entitySpace=ray.get(discoverySpaceManager.entitySpace.remote())
            ),
            [],
        )
    else:
        all_entities = reduce(
            operator.add, gen.entitySpaceIterator(entitySpace=space.entitySpace), []
        )

    # same as:
    # all_entities = sum(list(gen.entitySpaceIterator(entitySpace=space.entitySpace)), [])

    cp_ids = [cp.identifier for cp in space.entitySpace.constitutiveProperties]

    list_of_dicts_to_convert = []
    for e in all_entities:
        ed = {}
        ed["identifier"] = e.identifier
        for cp in cp_ids:
            obj = e.valueForConstitutivePropertyIdentifier(identifier=str(cp))
            ed[str(cp)] = obj.value
        list_of_dicts_to_convert.append(ed)

    return pd.DataFrame(list_of_dicts_to_convert)


def get_df_at_least_one_measured_value(
    discoverySpace: typing.Union["DiscoverySpace", str],
    targetOutput_list=None,
    # discoverySpaceManager: DiscoverySpaceManager | None = None, This makes the operator disappear
    discoverySpaceManager=None,
    add_measurement_id=False,
) -> pd.DataFrame:
    """
    Return a DataFrame of entities that have at least one measured target output from the
    provided list, aggregated across all experiments in the Discovery Space.

    - Each row represents an entity with measurements.
    - Includes identifier (optional), constitutive properties, and the requested target outputs.
    - Drops rows with missing values for the selected targets.
    - May Return an empty df

    Parameters
    ----------
    discoverySpace : DiscoverySpace | str
        The Discovery Space object or its identifier.
    targetOutput_list : list
        List of target output names to include in the DataFrame.
    add_measurement_id : bool, optional
        If True, include the entity identifier column in the output.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['identifier' (optional), <constitutive properties>, <target outputs>].
    """

    if not targetOutput_list:
        targetOutput_list = []
    space = get_space(space_or_space_id=discoverySpace)
    col_list = [cp.identifier for cp in space.entitySpace.constitutiveProperties]
    if add_measurement_id:
        col_list = ["identifier", *col_list]

    if discoverySpaceManager:
        df = pd.DataFrame(
            ray.get(
                discoverySpaceManager.matchingEntitiesTable.remote(
                    property_type="target",
                    aggregationMethod=PropertyAggregationMethodEnum.mean,
                )
            )
        )

        # logger.debug("Using manager in getting entities one measurement")
        # logger.warning(f"len source with manager is {len(df)}")
        # df = pd.DataFrame(
        #     discoverySpaceManager.matchingEntitiesTable.remote(
        #         property_type="target",
        #         aggregationMethod=PropertyAggregationMethodEnum.mean,
        #     )
        # )
    #
    #      line 153, in get_df_at_least_one_measured_value\n    df = pd.DataFrame(\n  \
    #     \       ^^^^^^^^^^^^^\n  File \"/Users/danielelotito/Documents/github/ad-orchestrator/.venv/lib/python3.12/site-packages/pandas/core/frame.py\"\
    #     , line 890, in __init__\n    raise ValueError(\"DataFrame constructor not properly\
    #     \ called!\")\nValueError: DataFrame constructor not properly called!."
    #   recorded_at: '2025-12-15T16:35:55.915448Z'
    # using async def in the class method does not change the result

    #     without '.remote' I have the following error

    #     discoverySpaceManager.matchingEntitiesTable(\n\
    # \  File \"/Users/danielelotito/Documents/github/ad-orchestrator/.venv/lib/python3.12/site-packages/ray/actor.py\"\
    # , line 659, in __call__\n    raise TypeError(\nTypeError: Actor methods cannot\
    # \ be called directly. Instead of running 'object.matchingEntitiesTable()', try\
    # \ 'object.matchingEntitiesTable.remote()'.."

    else:
        df = pd.DataFrame(
            space.matchingEntitiesTable(
                property_type="target",
                aggregationMethod=PropertyAggregationMethodEnum.mean,
            )
        )

    if df.empty:
        # NOTE: this condition is hit when there are no measurements at all existing in the space
        logger.warning(
            "No measured properties found in the discovery space\nReturning empty df\n "
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
            # Rename the column in the DataFrame
            df.rename(columns={f"{el}-mean": el}, inplace=True)
            valid_targetOutput_list += [el]
        elif f"{el}-mean" in all_df_cols and el in all_df_cols:
            logger.warning(
                f"Columns named '{el}-mean' and '{el}'"
                "found in the DataFrame obtained through matchingEntitiesTable. "
                f"Renaming it to '{el}'."
            )
            logger.error("Unexpected behavior can happen!")
            # Rename the column in the DataFrame
            df.rename(columns={f"{el}-mean": el}, inplace=True)
            valid_targetOutput_list += [el]
    col_list += valid_targetOutput_list

    # Something unexpected happened: log here about it
    if valid_targetOutput_list != targetOutput_list:
        if len(valid_targetOutput_list) == 0:
            logger.error(
                "No valid target in the columns of the df."
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

    # I can still have Nans here for cols in targetOutput_list,
    # because I am taking points for which I have at least one of the measured properties of the experiment
    df.dropna(inplace=True)

    # The resulting dataframe can be empty
    if df.empty:
        logger.warning(
            "Although there were some measured properties in the discovery space."
        )
        logger.warning(
            "All measured properties in the discovery space"
            f"are different from the desired outputs {targetOutput_list}.Returning empty df\n "
        )

    return df


# TODO: in case we keep the retrieval with ray get, this function will accept discoverySpaceManager as the first argument
def get_source_and_target(
    discoverySpace: typing.Union["DiscoverySpace", str],
    targetOutput: str,
    discoverySpaceManager=None,
    log_string: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build source (labeled) and target (unlabeled) DataFrames for a given target output `t`.
    Note, source can be empty

    - Retrieves measured entities for `t` and all entities without measurements.
    - Merges on common feature columns (excluding 'identifier').
    - Splits into:
        source_df: rows with non-null `t` (features + target).
        target_df: rows with null `t` (features only).

    Parameters
    ----------
    discoverySpace : str
        Discovery Space identifier (e.g., 'space-1a2469-6a3ed5').
    t : str
        Target output column name.

    Returns
    -------
    tuple
        (source_df, target_df)
    """

    dfm = get_df_at_least_one_measured_value(
        discoverySpace, [targetOutput], discoverySpaceManager=discoverySpaceManager
    )
    dfu = get_df_all_entities_no_measurements(
        discoverySpace, discoverySpaceManager=discoverySpaceManager
    )
    keys = [c for c in dfu.columns if c in dfm.columns and c != "identifier"]

    if dfm.empty:
        logger.warning("Empty source")
        return dfm, dfu

    df = dfu.merge(dfm, on=keys, how="left")

    # If nothing is measured you do not have the columns, so I add the column as empty to run the
    # following logic safely
    if targetOutput not in list(df.columns):
        logger.info(
            f"""The target output was not present in the columns of the measured+unmeasured dataframe,' \
                        meaning that '{targetOutput}' has never been measured in this space.
                        dfm.empty = {df.empty}. Adding an empty column to the dataframe.
                    """
        )
        logger.debug("Adding an empty column to the dataframe.")
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
        f"'{targetOutput}' column is missing, saving df in {save_path}, returning unmerged dataframes"
    )
    df.to_csv(save_path)
    return dfm, dfu


def validate_points_in_space(
    points: list[dict],
    space: DiscoverySpace,
) -> tuple[list[dict], list[int]]:
    """
    Validate a list of point dictionaries against a Discovery Space entity space.

    A point is considered valid if `space.entitySpace.isPointInSpace(point)` returns True.
    This function returns both the subset of valid points (in original order) and
    the indices of invalid points for diagnostics.

    Parameters
    ----------
    points : list[dict]
        List of point dicts `{constitutive_property_id: value}` to validate.
    space : DiscoverySpace
        The Discovery Space whose entity space defines the validity constraints.

    Returns
    -------
    (valid_points, invalid_indices) : tuple[list[dict], list[int]]
        valid_points :
            The points that are valid under `space.entitySpace.isPointInSpace`.
        invalid_indices :
            The zero-based indices (relative to the input `points`) that were invalid.

    Examples
    --------
    >>> points = make_points_from_df(df, space)
    >>> valid_points, invalid_idx = validate_points_in_space(points, space)
    >>> if invalid_idx:
    ...     print(f"Warning: {len(invalid_idx)} invalid rows at indices {invalid_idx}")
    """
    valid_points: list[dict] = []
    invalid_indices: list[int] = []

    for i, p in enumerate(points):
        if space.entitySpace.isPointInSpace(p):
            valid_points.append(p)
        else:
            invalid_indices.append(i)
    return valid_points, invalid_indices


def df_to_points(df, cols=None, dropna=True, drop_duplicates=False):
    """
    Convert DataFrame rows to list of "point" dicts {prop_id: value}.
    - cols: list of columns to use. If None use all df.columns.
    - dropna: drop rows containing any NaN (default True).
    - drop_duplicates: drop duplicate rows (default False).
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

    # Convert numpy scalars to python builtins for safety
    def to_py(x):
        import numpy as np

        if isinstance(x, (np.generic)):
            return x.item()
        return x

    # apply conversion (only if needed)
    for c in sub.columns:
        sub[c] = sub[c].map(to_py)

    return sub.to_dict(orient="records")


# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

# TODO: check if these are actually needed


def df_to_points_parsing(df, cols=None, dropna=True, parse_values=False):
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
    Convert a DataFrame of constitutive properties into a list of point dictionaries,
    using the entity-space canonical column order by default.

    Each point is a mapping {constitutive_property_id: value}. By default, rows with
    any NaN across the selected columns are dropped, and string values are parsed
    into Python literals where possible (e.g., "[1, 2]" -> [1, 2]) via `ast.literal_eval`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame whose columns correspond to constitutive property identifiers.
    space : DiscoverySpace
        The Discovery Space providing the canonical order of constitutive properties.
    cols : list[str], optional
        Explicit list of columns to use. If None, uses the canonical order:
        `[cp.identifier for cp in space.entitySpace.constitutiveProperties]`.
    dropna : bool, default True
        If True, drop rows containing any NaN in the selected columns.
    parse_values : bool, default True
        If True, attempt to parse string values into Python objects using `ast.literal_eval`.

    Returns
    -------
    list[dict]
        A list of point dicts, one per retained row: `[{prop_id: value, ...}, ...]`.

    Raises
    ------
    KeyError
        If any of the requested `cols` are not present in `df`.

    Examples
    --------
    >>> space_cols = [cp.identifier for cp in space.entitySpace.constitutiveProperties]
    >>> points = make_points_from_df(df, space, cols=space_cols, dropna=True, parse_values=True)
    """
    # Determine canonical order if cols not provided
    if cols is None:
        cols = [cp.identifier for cp in space.entitySpace.constitutiveProperties]

    # Validate requested columns exist
    missing = set(cols) - set(df.columns)
    if missing:
        raise KeyError(f"Requested columns not present in DataFrame: {missing}")

    # Convert rows -> point dicts, with optional parsing
    return df_to_points_parsing(df, cols=cols, dropna=dropna, parse_values=parse_values)


def get_list_of_entities_from_df_and_space(df: pd.DataFrame, space):
    """
    A utility to start from a df and a space and obtain a list of entities.
    """
    points = make_points_from_df(df=df, space=space)
    valid_points, __ = validate_points_in_space(points, space)

    list_of_entities = []
    from orchestrator.schema.point import SpacePoint

    for p in valid_points:
        # p is a dict mapping constitutive property id -> value
        sp = SpacePoint(entity=p)
        entity = (
            sp.to_entity()
        )  # builds an Entity from the dict without touching the sample store
        list_of_entities.append(entity)

    numberEntities = len(list_of_entities)
    if numberEntities != len(df):
        numberEntities_log = f"""Warning: number of valid entities {numberEntities} is different from the number of rows in the ordered df {len(df)}.
        This means that some rows in the ordered df did not correspond to valid entities in the discovery space.
        """
        logging.warning(numberEntities_log)
    return list_of_entities


# %%
