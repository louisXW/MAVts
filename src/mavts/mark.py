import numpy.typing as npt
import numpy as np
import pandas as pd


def all_periods(series: pd.Series,
                min_diff: float | None = None,
                max_drawdown: float | None = None) -> pd.DataFrame:
    """ Add boolean series as flags when a observation is part of a special
    period. The columns added are:
   "is_interpolated", "is_up_period", "is_dn_period", "is_top", "is_bottom"

   `series` is a pd.Series of observations

   `min_diff` is the minimum difference between observations so that the period
   is considered as an up or down period

   `max_drawdown` is the maximum drawdown (i.e. updown) between observations in
   order to end an up or down period.


   """

    ups, dns = up_dn_periods(series.values, min_diff, max_drawdown)
    peaks, bottoms = peaks_bottoms(ups, dns)
    interpolated_data = interpolated(series)

    all_df = pd.DataFrame(np.zeros((series.shape[0], 6)),
                          index=series.index,
                          columns=[series.name,
                                   "is_interpolated",
                                   "is_up_period",
                                   "is_dn_period",
                                   "is_top",
                                   "is_bottom"])
    all_df[series.name] = series

    for col, period in zip(all_df.columns[1:],
                           (interpolated_data, ups, dns, peaks, bottoms)):
        for pos in period:
            if isinstance(pos, tuple):
                left, right = pos
                all_df[col].iloc[left:right] = 1
            else:
                all_df[col].iloc[pos] = 1

    for bool_col in all_df.filter(regex="is_*").columns:
        all_df[bool_col] = all_df[bool_col].astype(bool)

    return all_df


def up_dn_periods(series: npt.NDArray,
                  min_diff_arg: float | None = None,
                  max_drawdown_arg: float | None = None) -> tuple:
    """
    Return a tuple of lists (ups, dns). Each list of ups and dns contains
    a tuple (start, end) of start index and end index of an up in (ups) and 
    down period in (dns). There is no overlap between periods.
    
   `series` is a pd.Series of observations

   `min_diff` is the minimum difference between observations so that the period
   is considered as an up or down period

   `max_drawdown` is the maximum drawdown (i.e. updown) between observations in
   order to end an up or down period.
    """

    min_diff = series.std() if min_diff_arg is None else min_diff_arg
    max_drawdown = min_diff/4 if max_drawdown_arg is None else max_drawdown_arg

    left, right = None, None
    left_dn, right_dn = None, None
    idx: int = 1
    last_top: float = 0
    last_bot: float = 1e12
    cur_mdd: float = 0
    cur_mdu: float = 0

    ups, dns = [], []

    while idx < series.shape[0]:

        cur_val = series[idx]
        change = cur_val - series[idx-1]

        if change > 0:
            last_top = max(last_top, cur_val)
            if left is None:
                left = idx - 1
            elif cur_val - series[left] >= min_diff:
                right = idx

            cur_mdu = max(cur_mdu, max(cur_val - last_bot, 0))
            if cur_mdu > max_drawdown:
                if left_dn and right_dn:
                    dns.append((left_dn, right_dn))
                right_dn = None
                left_dn = None
                last_bot = 1e12
                cur_mdu = 0

        if change < 0:
            cur_mdd = max(cur_mdd, max(last_top - cur_val, 0))
            if cur_mdd > max_drawdown:
                if left and right:
                    ups.append((left, right))
                right = None
                left = None
                last_top = 0
                cur_mdd = 0

            last_bot = min(last_bot, cur_val)
            if left_dn is None:
                left_dn = idx - 1
            elif series[left_dn] - cur_val >= min_diff:
                right_dn = idx

        idx += 1

    exclusive_ups, exclusive_dns, _ = _resolve_overlaps(ups, dns, series)

    return exclusive_ups, exclusive_dns


def peaks_bottoms(ups: list, dns: list) -> tuple:
    """ 
    Return a tuple of lists (peaks, bottoms) containing the indices where 
    there is a peak or bottom.

    Each list of ups and dns contains a tuple (start, end) of start index and 
    end index of an up in (ups) and down period in (dns).
    """
    idx = 0
    jdx = 0
    peaks, bottoms = [], []

    while True:
        up_start, up_end = ups[idx]
        dn_start, dn_end = dns[jdx]

        if up_start == dn_end:
            bottoms.append(dn_end)
        if dn_start == up_end:
            peaks.append(up_end)

        if up_start < dn_start:
            idx += 1 if idx + 1 < len(ups) else 0
        else:
            jdx += 1 if jdx + 1 < len(dns) else 0

        if idx + 1 == len(ups) or jdx + 1 == len(dns):
            break

    return peaks, bottoms


def interpolated(series: pd.Series) -> list:
    """ Mark the interpolated observations in the series """

    lvl_range = series.max() - series.min()  # type:ignore

    # make them from 0 to 100
    levels = 100 * (series - series.min()) / lvl_range
    changes = round(levels.diff(), 0)  # rounded daily changes
    # True if the rounded daily change repeats exactly
    repeats = (~changes.diff().astype(bool))
    # diff here makes everything False except where there is a change
    # cumsum count how many changes there
    # multiply with repeats to set to zero the counts where there is no repeat
    counts = repeats * repeats.diff().cumsum()
    # finally get value counts of grouping by the number of repeats
    # that are greater than 2
    # without the first as the first is the zero group when there is no repeats
    # the repeat_groups now has index of "group_id"
    # and the value is the number of repeats for that group
    repeat_groups = counts.value_counts()[counts.value_counts().gt(2)].iloc[1:]
    # now plot the groups with 30 days before and 30 days after

    interpolated = []

    for group_id in repeat_groups.to_dict().keys():
        hl_range = levels.index[counts.eq(group_id)]
        # the 2 that already happened because we diff twice
        left = max(levels.index.get_loc(hl_range[0]) - 2, 0)  # type:ignore
        right = levels.index.get_loc(hl_range[-1])  # type:ignore
        interpolated.append((left, right))

    return interpolated


def _resolve_overlaps(ups: list, dns: list, series: npt.NDArray) -> tuple:
    """ 
    Find and remove overlaps between two lists of segments
    Each list of ups and dns contains
    a tuple (start, end) of start index and end index of an up in (ups) and 
    down period in (dns).
    
    The returned list is in the same format
    """
    n = len(ups)
    m = len(dns)

    if n == 0 or m == 0:
        return ups, dns, []

    idx = 0
    jdx = 0

    overlaps = []
    ex_ups, ex_dns = [[l, r] for l, r in ups], [[l, r] for l, r in dns]
    while True:
        up_start, up_end = ups[idx]
        dn_start, dn_end = dns[jdx]

        if up_start < dn_start:
            if up_end > dn_start:
                l, r = dn_start, min(up_end, dn_end)
                t = series[l:r].argmax()
                overlaps.append((l, r))
                ex_ups[idx][1] = l + t
                ex_dns[jdx][0] = l + t
            if dns[min(jdx+1, m-1)][0] > up_end:
                idx += 1
            else:
                jdx += 1
        elif up_start >= dn_start:
            if up_start < dn_end:
                l, r = up_start, min(up_end, dn_end)
                b = series[l:r].argmin()
                overlaps.append((l, r))
                ex_dns[jdx][1] = l + b
                ex_ups[idx][0] = l + b
            if ups[min(idx+1, n-1)][0] > dn_end:
                jdx += 1
            else:
                idx += 1

        if idx == n or jdx == m:
            break

    return ex_ups, ex_dns, overlaps
