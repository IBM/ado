# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
#
import logging

logger = logging.getLogger(__name__)


# TRIM- Optimized
def get_index_list_nn(
    length_segment: int,
    tot_points_to_sample: int,
    sampled_indices: list[int] | None = None,
    sort: bool = False,
    verbose: bool = False,
) -> list[int]:
    """
    Selects a set of indices from a 1D segment using a deterministic sampling strategy.
    This function assumes that the data has been projected into a 1D segment based on feature importance,
    making it isomorphic to a 1d segment. The goal is to sample `tot_points_to_sample` indices from this segment,
    optionally considering a set of already sampled indices (`sampled_indices`). The strategy ensures that the
    selected points are well-distributed and structurally balanced, akin to placing support ropes on a beam to
    prevent collapse.

    The metaphor used is that of a beam suspended by ropes. Initially, ropes are placed at the extremities (indices 0 and `length_segment - 1`)
    to ensure boundary support. Additional ropes (sampled points) are added iteratively at the midpoint of the longest unsampled intervals.
    In cases of symmetry or multiple equally sparse regions, the algorithm evaluates local neighborhood density to prioritize selection.


    For example, consider a segment of 14 elements (get_index_list_nn(14,8)):

    ::

        Index:     0  1  2  3  4  5  6  7  8  9 10 11 12 13
        Sample:    1  -  8  5  -  7  3  -  -  4  -  6  -  2

    Here, numbers represent the order in which points were added, and `-` indicates unsampled positions.
    The algorithm ensures that each new point is placed where it maximally improves the balance of the structure,
    often targeting the midpoint of the largest gaps.

    :param length_segment: Total number of units in the 1D segment.
    :type length_segment: int
    :param tot_points_to_sample: Total number of indices to sample.
    :type tot_points_to_sample: int
    :param sampled_indices: List of indices already sampled. Defaults to an empty list.
    :type sampled_indices: list[int], optional
    :param sort: If True, returns the final list sorted in ascending order. Defaults to False.
    :type sort: bool, optional
    :param verbose: If True, prints debug information during sampling. Defaults to False.
    :type verbose: bool, optional

    :raises ValueError: If `tot_points_to_sample` exceeds `length_segment`.

    :return: A list of sampled indices satisfying the distribution strategy.
    :rtype: list[int]

    :example:

    >>> get_index_list_nn(5, 3, sampled_indices=[0, 4])
    [0, 2, 4]

    >>> get_index_list_nn(10, 4, sampled_indices=[0, 4, 9])
    [0, 4, 6, 9]

    This strategy is particularly useful in optimization settings where boundary coverage and balanced sampling are important.
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
        return sampled_indices

    index_list = list(sampled_indices)
    sampled_set = set(index_list)

    if 0 not in sampled_set:
        index_list.append(0)
        sampled_set.add(0)
        if len(index_list) == tot_points_to_sample:
            return sorted(index_list)

    if (length_segment - 1) not in sampled_set:
        index_list.append(length_segment - 1)
        sampled_set.add(length_segment - 1)
        if len(index_list) == tot_points_to_sample:
            return sorted(index_list)

    def build_prefix_and_len():
        """
        Builds prefix sums over a truncated mask: M = max(index_list)+1.
        prefix[j] = sum(mask[0:j]) with prefix length M+1.
        """
        M = max(index_list) + 1
        prefix = [0] * (M + 1)
        s = 0

        for i in range(M):
            s += 1 if i in sampled_set else 0
            prefix[i + 1] = s
        return prefix, M

    def get_list_min_weight(prefix, M, d, selectable_indices):
        """
        uses prefix sums instead of numpy.mean.
        Only considers indices i in selectable_indices intersected with [0, M-1],
        and preserves ascending order for ties exactly like the OG.
        """
        # cmpute mean densities and track min
        # We must preserve order: OG loops i = 0..M-1 and filters by membership.
        # Achieve the same by iterating selectable_indices (which we build in ascending order)
        # but breaking when i >= M.
        vals = {}
        for i in selectable_indices:
            if i >= M:
                break
            left = i - d
            right = i + d
            if left < 0:
                left = 0
            if right >= M:
                right = M - 1
            total = prefix[right + 1] - prefix[left]
            denom = right - left + 1
            mean = total / denom  # float64-equivalent - matches numpy.mean on booleans
            vals[i] = mean

        if not vals:
            return []

        min_val = min(vals.values())
        # preserving order of candidates as OG: ascending index order
        out = []
        for i in selectable_indices:
            if i >= M:
                break
            if vals.get(i) == min_val:
                out.append(i)
        return out

    # Secret sauce - electable indices function: same order as OG (ascending)
    def get_selectable_indices():
        # OG did O(N*m) with "i not in list", but we do O(N) with a set, but order identical.
        return [i for i in range(length_segment) if i not in sampled_set]

    max_d = length_segment

    # main loop
    while len(index_list) < tot_points_to_sample:
        selection = 0
        selectable_indices = get_selectable_indices()

        # prefix sums for the current (truncated) mask once per outer iteration
        prefix, M = build_prefix_and_len()

        d = 1
        # keeping "previous set" semantics exactly (used when l becomes empty)
        previous_set = selectable_indices

        while selection == 0:
            indices = get_list_min_weight(prefix, M, d, selectable_indices)

            if not indices:
                # Exact OG behavior: pick first element of the previous set
                # when the intersection is empty at this d.
                assert previous_set, "Previous candidate set should not be empty"
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
                # narrowing minimal-density set
                previous_set = selectable_indices
                selectable_indices = indices

                if len(selectable_indices) == 1 or d == max_d:
                    # pick the first element (ascending order preserved)
                    if verbose:
                        logger.info(
                            f"Appending to {index_list} the first element of {selectable_indices}"
                        )
                    chosen = selectable_indices[0]
                    index_list.append(chosen)
                    sampled_set.add(chosen)
                    selection = 1

            # OG increments d regardless it's immaterial after selection, but we mirror it lol
            d += 1

    if sort:
        return sorted(index_list)
    return index_list


def get_index_list_ordered_partitions(n: int, tot_points: int) -> list[int]:
    """
    The sampling strategy is the following one.
    You make the data isomorph to a 1d segment by ordering it according to source space feature importance
    and then the problem becomes selecting points in this segment.
    More specifically, the length of the dataframe is equal to the length of the segment + 1.
    And points can be selected on units, i.e. starting from 0 and going up to len(df)-1.
    When no point is selected I use
    Some points may have been already selected.

    NOTE: n is the len(df), greatest acceptable index will be n-1"

    """
    if tot_points == 0:
        logger.debug("No points selected from the list, return empty list")
        return []
    if tot_points > n:
        raise ValueError
    if tot_points == 1:
        return [0]
    index_list = [n - 1, 0]
    f_count = 0
    while f_count + 2 < tot_points:
        l_copy_sorted = index_list.copy()
        l_copy_sorted.sort()
        l_copy = index_list.copy()
        for i, el in enumerate(l_copy[1:]):
            start = el
            index_seen = l_copy_sorted.index(el)
            end = l_copy_sorted[index_seen + 1]
            mid = midpoint(start=start, end=end)
            if mid in index_list:
                continue
            f_count += 1
            index_list.append(mid)
            if f_count + 2 == tot_points:
                break
    index_list.sort()
    return index_list


# %%


def sorting_and_check(target_metric, source_space_df):
    """NOTE: this is the old function
    This function is responsible of the ordering in the 1-D sampling"""

    cols = list(source_space_df.columns)
    valid_cols = [col for col in cols if not col.startswith(target_metric)]

    valid_cols = list(set(valid_cols))

    if "total_tokens_per_batch" in valid_cols:
        valid_cols = ["total_tokens_per_batch"] + [
            c for c in valid_cols if c != "total_tokens_per_batch"
        ]
    df_copy = source_space_df.copy()

    return df_copy.sort_values(by=valid_cols).reset_index(drop=True)


def generate_df_and_point_mask(df, k):
    """Do not shuffle here or after, and the index of the df must be reset!"""
    selected_points = get_index_list_ordered_partitions(len(df), k)
    mask = [i in selected_points for i in range(len(df))]
    temp_df = df[mask].copy()

    return temp_df, mask


def midpoint(start: int, end: int) -> int:
    """n is a valid index for your row/array"""
    assert end - start >= 0
    return start + ((end - start) // 2)


# %% Here the 'weakpoint' of ordered partition that I am trying to address

# IMPORT LIKE THIS
# from trim.utils.sample import get_index_list_nn, get_index_list_ordered_partitions
# get_index_list_ordered_partitions(14,6) == [0, 1, 3, 6, 9, 13] # I do not now if I like this because I fill the 3-steps gap at the beginning before the four step gap at the end

# %% TESTING HAPPENS HERE

# The sampling strategy is the following one.
# You make the data isomorph to a 1d segment by ordering it according to source space feature importance
# and then the problem becomes selecting points in this segment.
# More specifically, the length of the dataframe is equal to the length of the segment + 1.
# And points can be selected on units, i.e. starting from 0 and going up to len(df)-1.
# Some points may have been already been sampled, these corresponds to points in the 1d segment that are in the set 'sampled_indices'
# When no point is selected this function (sampled_indices=[]) this function is NOT equivalent to get_index_list, which follows a slightly different logic (based on creating ordered partitions).
# An intuitive picture behind the function is that I am holding a 1d beam with indestructible ropes, and I want to be sure that the beam does not collapse under
# its own weight, so I try to add a rope at the midpoint of the longest portion of the beam without ropes.
# The first ropes, however, must be at the extremities of the beam because this could be interesting to an optimization setting.
# Let me introduce our notation: r represent initial ropes, numbers are for the newly added points, in r3-r it means that the third point has been added at index 1 [indices start from zero]

# Moreover, for cases where I have multiple instances of an interval I need to tell you the whole story:
# eg. r---r---r---r---r I first notice that I will fill 4 gaps of the same length.
# I will now assign to each point a number, that is based on the average number of neighbours in distance 1
# r-o-r-o-r-o-r-o-r
# the 4 o have 0 neighbours so I go to distance 2, even again, then go to distance 3, If I meet boundaries I do not include them in the averages
# The os in the middle have a lower number, so they go to the next step
# Since they are symmetric I select one of them and repeat the procedure


# Also note that If I need to place an additional point here r--r I am biased toward the beginning so I will have r3-r.
# The len(index_list) = the greatest integer in the 1d representation.

# r-3-r  # 5 elements segment,  get_index_list_nn(5,3, sampled_indices=[0,4]) = [0,2,4]
# r-3-2  # 5 elements segment,  get_index_list_nn(5,3, sampled_indices=[0,]) = [0,2,4]
# 2---r  # 5 elements segment,  get_index_list_nn(5,2, sampled_indices=[4]) = [0,4]
# r---r----r # 10 elements segment,  get_index_list_nn(10,2, sampled_indices=[0,4]) = VALUEERROR (cannot include extrema in the list)
# r---r----r # 10 elements segment,  get_index_list_nn(10,3, sampled_indices=[0,4,9]) = [0,4,9]
# r---r-4--r # 10 elements segment,  get_index_list_nn(10,4, sampled_indices=[0,4,9]) = [0,4,6,9]
# r---r---r---r---r # 17 elements segment,  get_index_list_nn(17, 6, sampled_indices=[0,4,8,12,16]) = [0,2,4,8,12,16]
# r---r---r---r---r # 17 elements segment,  get_index_list_nn(17, 7, sampled_indices=[0,4,8,12,16]) = [0,2,4,8,10,12,16]
# To sum up:
# This is result for a segment of 14 elements (commas are for clarity)

# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13  # indices row, just for visual aid
# 1, 7, -, 5, -, -, 3, 8, -, 4, -, 6, -, 2


# %%


def get_index_list_nn_not_optimized(
    length_segment: int,
    tot_points_to_sample: int,
    sampled_indices: list[int] = [],
    sort: bool = False,
    verbose: bool = False,
) -> list[int]:
    """SAME AS ABOVE BUT NOT OPTIMIZED
    Selects a set of indices from a 1D segment using a deterministic sampling strategy.
    This function assumes that the data has been projected into a 1D segment based on feature importance,
    making it isomorphic to a 1d segment. The goal is to sample `tot_points_to_sample` indices from this segment,
    optionally considering a set of already sampled indices (`sampled_indices`). The strategy ensures that the
    selected points are well-distributed and structurally balanced, akin to placing support ropes on a beam to
    prevent collapse.

    The metaphor used is that of a beam suspended by ropes. Initially, ropes are placed at the extremities (indices 0 and `length_segment - 1`)
    to ensure boundary support. Additional ropes (sampled points) are added iteratively at the midpoint of the longest unsampled intervals.
    In cases of symmetry or multiple equally sparse regions, the algorithm evaluates local neighborhood density to prioritize selection.


    For example, consider a segment of 14 elements (get_index_list_nn(14,8)):

    ::

        Index:     0  1  2  3  4  5  6  7  8  9 10 11 12 13
        Sample:    1  -  8  5  -  7  3  -  -  4  -  6  -  2

    Here, numbers represent the order in which points were added, and `-` indicates unsampled positions.
    The algorithm ensures that each new point is placed where it maximally improves the balance of the structure,
    often targeting the midpoint of the largest gaps.

    :param length_segment: Total number of units in the 1D segment.
    :type length_segment: int
    :param tot_points_to_sample: Total number of indices to sample.
    :type tot_points_to_sample: int
    :param sampled_indices: List of indices already sampled. Defaults to an empty list.
    :type sampled_indices: list[int], optional
    :param sort: If True, returns the final list sorted in ascending order. Defaults to False.
    :type sort: bool, optional
    :param verbose: If True, prints debug information during sampling. Defaults to False.
    :type verbose: bool, optional

    :raises ValueError: If `tot_points_to_sample` exceeds `length_segment`.

    :return: A list of sampled indices satisfying the distribution strategy.
    :rtype: list[int]

    :example:

    >>> get_index_list_nn(5, 3, sampled_indices=[0, 4])
    [0, 2, 4]

    >>> get_index_list_nn(10, 4, sampled_indices=[0, 4, 9])
    [0, 4, 6, 9]

    This strategy is particularly useful in optimization settings where boundary coverage and balanced sampling are important.
    """

    # CHECKS
    if tot_points_to_sample == 0:
        return []

    if tot_points_to_sample > length_segment:
        raise ValueError(
            "You are trying to sample more points than those that are available"
        )

    if len(sampled_indices) == length_segment:
        return sampled_indices

    index_list = sampled_indices.copy()

    if 0 not in index_list:
        index_list.append(0)
        if len(index_list) == tot_points_to_sample:
            return sorted(index_list)

    if length_segment - 1 not in index_list:
        index_list.append(length_segment - 1)
        if len(index_list) == tot_points_to_sample:
            return sorted(index_list)

    def get_list_min_weight(
        l_for_min_weight: list[bool], d: int, selectable_indices: list[int]
    ) -> list[int]:
        import numpy as np

        res = {}
        for i in range(len(l_for_min_weight)):
            if i not in selectable_indices:
                continue
            min_index = max(i - d, 0)
            max_index = min(i + d, len(l_for_min_weight) - 1)
            neighborhood = l_for_min_weight[min_index : max_index + 1]
            res[i] = np.array(neighborhood).mean()

        # Find the minimum value
        if res:
            min_val = min(res.values())
            return [k for k, v in res.items() if v == min_val]
        return []

    def get_binary_array(index_list):
        return [i in index_list for i in range(max(index_list) + 1)]

    def get_selectable_indices(index_list, length_segment):
        return [i for i in range(length_segment) if i not in index_list]

    max_d = length_segment
    while len(index_list) < tot_points_to_sample:
        selection = 0
        selectable_indices = get_selectable_indices(
            index_list, length_segment=length_segment
        )
        d = 1
        indices = []
        while selection == 0:
            # in cases of symmetry can be that the intersection selectable indices is empty so, you need to pre filter this step, it the
            # intersection is going to be empty we chose the first index of the previous list

            indices = get_list_min_weight(
                get_binary_array(index_list), d, selectable_indices=selectable_indices
            )
            # If I cannot filter further
            if not indices:
                # I use the previously set, asserting is not empty
                assert selectable_indices
                selection = 1
                if verbose:
                    logger.info(
                        f"No intersection found with d={d}. Using the previous set Appending to {index_list} the first element of {selectable_indices}"
                    )
                index_list.append(selectable_indices[0])
            else:
                selectable_indices = indices
                if len(selectable_indices) == 1 or d == max_d:
                    selection = 1
                    if verbose:
                        logger.info(
                            f"Appending to {index_list} the first element of {selectable_indices}"
                        )
                    index_list.append(selectable_indices[0])
            d += 1

    if sort:
        return sorted(index_list)
    return index_list
