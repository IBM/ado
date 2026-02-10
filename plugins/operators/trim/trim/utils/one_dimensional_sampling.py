# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
#
import logging

logger = logging.getLogger(__name__)


def get_index_list_van_der_corput(
    length_segment: int,
    tot_points_to_sample: int,
    sampled_indices: list[int] | None = None,
    sort: bool = False,
    verbose: bool = False,
) -> list[int]:
    """
    Selects a set of indices from a 1D segment using a deterministic sampling strategy.
    It is a modified Van der Corput Sequence

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

    ## Additional Observations and examples
    This function assumes that the data has been projected into a 1D segment based on feature importance,
    making it isomorphic to a 1d segment. The goal is to sample `tot_points_to_sample` indices from this segment,
    optionally considering a set of already sampled indices (`sampled_indices`). The strategy ensures that the
    selected points are well-distributed and structurally balanced, akin to placing support ropes on a beam to
    prevent collapse.

    The metaphor used is that of a beam suspended by ropes. Initially, ropes are placed at the extremities (indices 0 and `length_segment - 1`)
    to ensure boundary support. Additional ropes (sampled points) are added iteratively at the midpoint of the longest unsampled intervals.
    In cases of symmetry or multiple equally sparse regions, the algorithm evaluates local neighborhood density to prioritize selection.


    For example, consider a segment of 14 elements (get_index_list_van_der_corput(14,8)):

    ::

        Index:     0  1  2  3  4  5  6  7  8  9 10 11 12 13
        Sample:    1  -  8  5  -  7  3  -  -  4  -  6  -  2

    Here, numbers in the bottom row represent the order in which each point is added, and `-` indicates unsampled positions.
    The algorithm ensures that each new point is placed where it maximally improves the balance of the structure,
    often targeting the midpoint of the largest gaps.

    :examples:

    >>> get_index_list_van_der_corput(5, 3, sampled_indices=[0, 4])
    [0, 2, 4]

    >>> get_index_list_van_der_corput(10, 4, sampled_indices=[0, 4, 9])
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
        maximal_indices_list = list(range(length_segment))
        if sampled_indices.sort() != maximal_indices_list:
            logging.error(
                "Sampled indices do not correspond to [0,..., max_n_indices -1]"
                "Returning list(range(max_n_indices)"
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
        """
        Builds prefix sums over a truncated mask: M = max(index_list)+1.
        prefix[j] = sum(mask[0:j]) with prefix length M+1.
        """
        if not index_list:
            return [0], 0

        M = max(index_list) + 1

        # You must define sampled_set based on the input list
        sampled_set = set(index_list)

        prefix = [0] * (M + 1)
        s = 0

        for i in range(M):
            # i represents the current index in the imaginary mask array
            s += 1 if i in sampled_set else 0
            prefix[i + 1] = s

        return prefix, M

    def get_list_min_weight(
        prefix: list[int], M: int, d: int, selectable_indices: list[int]
    ) -> list[int]:
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

    def get_selectable_indices() -> list[int]:
        # OG did O(N*m) with "i not in list", but we do O(N) with a set, but order identical.
        return [i for i in range(length_segment) if i not in sampled_set]

    max_d = length_segment

    # main loop
    while len(index_list) < tot_points_to_sample:
        selection = 0
        selectable_indices = get_selectable_indices()

        # prefix sums for the current (truncated) mask once per outer iteration
        prefix, M = build_prefix_and_len(index_list=index_list)

        d = 1
        # keeping "previous set" semantics exactly (used when l becomes empty)
        previous_set = selectable_indices

        while selection == 0:
            indices = get_list_min_weight(prefix, M, d, selectable_indices)

            if not indices:
                # Exact OG behavior: pick first element of the previous set
                # when the intersection is empty at this d.
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

            # OG increments d regardless it's immaterial after selection, but we mirror it
            d += 1

    if sort:
        return sorted(index_list)
    return index_list


def get_index_list_ordered_partitions(n: int, tot_points: int) -> list[int]:
    """
    Select indices from a 1D segment using a partition-based sampling strategy.

    The data is treated as isomorphic to a 1D segment ordered by feature importance.
    Points are selected by iteratively finding midpoints of the largest gaps.

    Args:
        n: Total length of the segment (len(df)), valid indices are 0 to n-1
        tot_points: Number of points to sample

    Returns:
        Sorted list of sampled indices

    Raises:
        ValueError: If tot_points exceeds n
    """
    if tot_points == 0:
        logger.debug("No points selected from the list, return empty list")
        return []
    if tot_points > n:
        raise ValueError
    if tot_points == 1:
        return [0]
    index_list = [n - 1, 0]
    number_of_inner_points_sampled = 0
    while number_of_inner_points_sampled + 2 < tot_points:
        l_copy_sorted = index_list.copy()
        l_copy_sorted.sort()
        l_copy = index_list.copy()
        for _i, el in enumerate(l_copy[1:]):
            start = el
            index_seen = l_copy_sorted.index(el)
            end = l_copy_sorted[index_seen + 1]
            mid = midpoint(start=start, end=end)
            if mid in index_list:
                continue
            number_of_inner_points_sampled += 1
            index_list.append(mid)
            if number_of_inner_points_sampled + 2 == tot_points:
                break
    index_list.sort()
    return index_list


def midpoint(start: int, end: int) -> int:
    """
    Calculate the midpoint between two indices.

    Args:
        start: Starting index
        end: Ending index

    Returns:
        Integer midpoint index

    Raises:
        ValueError: If start is greater than end
    """
    if end - start < 0:
        raise ValueError("Start is greater than end!")
    return start + ((end - start) // 2)
