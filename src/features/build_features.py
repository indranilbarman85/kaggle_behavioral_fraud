"""Feature definitions for behavioral fraud scoring.

Each feature is a plain function ``f(df) -> Series``, self-contained: it computes
whatever it needs from the raw columns, so features never depend on each other.

To switch a feature on, add it to :data:`FEATURES` at the bottom of this file.
To try one without editing the file::

    build_feature_frame(df, features=[amount_log, amount_zscore_vs_user])

Leakage rule followed throughout: any per-user statistic describing "normal
behavior" is computed from *strictly prior* transactions of that user, so a
transaction never contributes to its own baseline. The first transaction of a
user therefore has NaN for those features — the median imputer in
`src.data.preprocess` fills them.

Frequency/count encodings are unsupervised, so they are meant to be computed on
the combined train+test frame (`data/processed/combined.csv`, see notes.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

USER_COL = "user_id"
AMOUNT_COL = "amount"
GAP_COL = "hours_since_prev_txn"
LABEL_COL = "label"
ID_COLS = ["transaction_id", "user_id", "timestamp"]
CATEGORICAL_COLS = ["merchant_category", "country", "device_id", "channel"]

# `timestamp` is a per-user sequence number (see notes.md), so every "so far"
# feature depends on this ordering.
SORT_COLUMNS = ["user_id", "timestamp"]

# Look-back window, in hours, for the recent-activity count features.
RECENT_WINDOW_HOURS = 57_000

# Gap below which a transaction counts as "hard on the heels" of the previous
# one. Set to the 5th percentile of real (non-first) gaps: the median gap is
# ~39,000h, so a human-scale threshold like 24h would fire on 0.03% of rows.
SHORT_GAP_HOURS = 30_000

# Bound on `gap_zscore_vs_user`, in std devs. Near-identical prior gaps make the
# denominator vanish and the raw z-score reach six figures; see that function.
GAP_ZSCORE_CLIP = 10.0

FeatureFunc = Callable[[pd.DataFrame], pd.Series]


# --------------------------------------------------------------------------
# helpers — shared maths used by several feature functions
# --------------------------------------------------------------------------
def _past_count(df: pd.DataFrame) -> pd.Series:
    """Number of transactions the user had before this one."""
    return df.groupby(USER_COL).cumcount()


def _past_sum(df: pd.DataFrame, col: str) -> pd.Series:
    """Running sum of `col` over the user's strictly prior transactions."""
    return df.groupby(USER_COL)[col].cumsum() - df[col]


def _past_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Mean of `col` over the user's strictly prior transactions."""
    count = _past_count(df)
    return (_past_sum(df, col) / count).where(count > 0)


def _past_std(df: pd.DataFrame, col: str) -> pd.Series:
    """Population std over prior transactions; NaN until 2 of them exist."""
    count = _past_count(df)
    mean = _past_mean(df, col)
    squares = df[col] ** 2
    past_sum_sq = squares.groupby(df[USER_COL]).cumsum() - squares
    variance = (past_sum_sq / count - mean**2).clip(lower=0)
    return np.sqrt(variance).where(count > 1)


def _prior_mean_std(
    values: pd.Series, users: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Mean and population std over each user's strictly prior *non-null* values.

    The NaN-tolerant counterpart to :func:`_past_mean` / :func:`_past_std`: nulls
    are skipped rather than propagated, so a column that deliberately blanks out
    unusable rows (see :func:`_real_gap`) still gets a baseline from the rest.
    Std is NaN until two prior values exist.
    """
    valid = values.notna()
    filled = values.fillna(0.0)

    count = valid.astype(int).groupby(users).cumsum() - valid.astype(int)
    total = filled.groupby(users).cumsum() - filled
    mean = (total / count).where(count > 0)

    squares = filled**2
    total_sq = squares.groupby(users).cumsum() - squares
    variance = (total_sq / count - mean**2).clip(lower=0)
    return mean, np.sqrt(variance).where(count > 1)


def _real_gap(df: pd.DataFrame) -> pd.Series:
    """`hours_since_prev_txn` with the first-transaction sentinel blanked out.

    A user's first transaction has no predecessor, so its gap is a sentinel
    (999 for most users, see notes.md) rather than a measured interval. Left in,
    it drags the user's early rhythm baseline toward 999 against a real median
    gap of ~39,000h. NaN here means "no interval", which
    :func:`_prior_mean_std` skips.
    """
    return df[GAP_COL].clip(lower=0).where(_past_count(df) > 0)


def _is_first_occurrence(df: pd.DataFrame, col: str) -> pd.Series:
    """True when this user has never used this value of `col` before."""
    return df.groupby([USER_COL, col]).cumcount().eq(0)


def _is_new_for_user(df: pd.DataFrame, col: str) -> pd.Series:
    """Value of `col` never seen before for this user (first txn excluded)."""
    return (_is_first_occurrence(df, col) & _past_count(df).gt(0)).astype(int)


def _prev_value(df: pd.DataFrame, col: str) -> pd.Series:
    """Value of `col` on the user's previous transaction; NaN on the first."""
    return df.groupby(USER_COL)[col].shift(1)


def _changed_from_prev(df: pd.DataFrame, col: str) -> pd.Series:
    """Value of `col` differs from the user's previous transaction."""
    prev = df.groupby(USER_COL)[col].shift(1)
    return prev.notna().mul(df[col].ne(prev)).astype(int)


def _prior_share(df: pd.DataFrame, col: str) -> pd.Series:
    """Share of the user's prior transactions that had this value of `col`."""
    count = _past_count(df)
    prior_same = df.groupby([USER_COL, col]).cumcount()
    return (prior_same / count).where(count > 0)


def _hours_elapsed(df: pd.DataFrame) -> pd.Series:
    """Hours between the user's first transaction and this one.

    `timestamp` only orders transactions (notes.md), so the user's clock is
    rebuilt by accumulating `hours_since_prev_txn` along that order. A user's
    first transaction has no predecessor — its gap is a sentinel (999 for most
    users) — so it is zeroed and the clock starts at 0.
    """
    gap = df[GAP_COL].clip(lower=0).where(_past_count(df) > 0, 0.0)
    return gap.groupby(df[USER_COL]).cumsum()


def _past_count_within(df: pd.DataFrame, window_hours: float) -> pd.Series:
    """Strictly prior transactions of the user within `window_hours` of this one.

    Elapsed time is non-decreasing within a user, so each user's prior window
    is a contiguous slice found by binary search.
    """
    elapsed = _hours_elapsed(df).to_numpy(dtype=float)
    counts = np.empty(len(df), dtype=float)
    for positions in df.groupby(USER_COL, sort=False).indices.values():
        user_elapsed = elapsed[positions]
        window_start = np.searchsorted(
            user_elapsed, user_elapsed - window_hours, side="left"
        )
        counts[positions] = np.arange(len(positions)) - window_start
    return pd.Series(counts, index=df.index)


def _zscore(value: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
    """Standardise, leaving NaN where the baseline is undefined or degenerate."""
    return (value - mean) / std.where(std > 0)


# --------------------------------------------------------------------------
# features — amount deviation
# --------------------------------------------------------------------------
def amount_log(df: pd.DataFrame) -> pd.Series:
    """log1p(amount) — amount is heavily right-skewed (notes.md)."""
    return np.log1p(df[AMOUNT_COL])


def amount_zscore_vs_user(df: pd.DataFrame) -> pd.Series:
    """How many of the user's own std devs this amount sits from their mean."""
    return _zscore(
        df[AMOUNT_COL], _past_mean(df, AMOUNT_COL), _past_std(df, AMOUNT_COL)
    )


def user_avg_amount_so_far(df: pd.DataFrame) -> pd.Series:
    """The user's average amount over their strictly prior transactions."""
    return _past_mean(df, AMOUNT_COL)


def amount_ratio_to_user_mean(df: pd.DataFrame) -> pd.Series:
    """Amount relative to the user's historical average spend."""
    mean = _past_mean(df, AMOUNT_COL)
    return df[AMOUNT_COL] / mean.where(mean > 0)


def amount_diff_from_user_mean(df: pd.DataFrame) -> pd.Series:
    """Amount minus the user's average over their strictly prior transactions.

    The signed sibling of :func:`amount_ratio_to_user_mean`: in currency units
    rather than as a multiple, so a €500 overshoot reads the same for a big
    spender and a small one.
    """
    return df[AMOUNT_COL] - _past_mean(df, AMOUNT_COL)


def amount_abs_diff_from_user_mean(df: pd.DataFrame) -> pd.Series:
    """Distance from the user's historical average, ignoring direction.

    Use when unusually *small* amounts (card testing) matter as much as
    unusually large ones; the signed version above keeps the direction.
    """
    return amount_diff_from_user_mean(df).abs()


def amount_over_user_max(df: pd.DataFrame) -> pd.Series:
    """1 when this is the largest amount the user has ever spent."""
    past_max = df.groupby(USER_COL)[AMOUNT_COL].transform(
        lambda s: s.shift(1).cummax()
    )
    return (df[AMOUNT_COL] > past_max).astype(float).where(past_max.notna())


def amount_diff_from_prev(df: pd.DataFrame) -> pd.Series:
    """Change in amount versus the user's immediately preceding transaction."""
    return df[AMOUNT_COL] - df.groupby(USER_COL)[AMOUNT_COL].shift(1)


def amount_pct_rank_in_user_history(df: pd.DataFrame) -> pd.Series:
    """Percentile of this amount within the user's *whole* history.

    Not causal: it looks at future transactions of the same user, which is
    acceptable for this competition (test users are known up front). Use only
    if the CV lift justifies it.
    """
    return df.groupby(USER_COL)[AMOUNT_COL].rank(pct=True)


# --------------------------------------------------------------------------
# features — timing deviation
# --------------------------------------------------------------------------
def gap_log(df: pd.DataFrame) -> pd.Series:
    """log1p(hours_since_prev_txn) — also right-skewed."""
    return np.log1p(df[GAP_COL].clip(lower=0))


def gap_zscore_vs_user(df: pd.DataFrame) -> pd.Series:
    """Deviation of this gap from the user's usual transaction rhythm.

    Both the gap and its baseline come from :func:`_real_gap`, so the
    first-transaction sentinel never poses as an interval. NaN until the user
    has two measured gaps behind them.

    Clipped to +/-:data:`GAP_ZSCORE_CLIP`. A user whose prior gaps happened to be
    near-identical has a near-zero std, and dividing by it sends the z-score
    into the hundreds of thousands -- which would leave `StandardScaler`
    squashing every ordinary row toward zero. Past the clip the exact magnitude
    says more about the denominator than about the transaction.
    """
    gap = _real_gap(df)
    mean, std = _prior_mean_std(gap, df[USER_COL])
    return _zscore(gap, mean, std).clip(-GAP_ZSCORE_CLIP, GAP_ZSCORE_CLIP)


def gap_ratio_to_user_mean(df: pd.DataFrame) -> pd.Series:
    """Gap relative to the user's average gap; <1 means a burst of activity."""
    mean = _past_mean(df, GAP_COL)
    return df[GAP_COL] / mean.where(mean > 0)


def is_short_gap_since_prev(df: pd.DataFrame) -> pd.Series:
    """1 when this transaction follows the previous one unusually quickly.

    "Quickly" means within :data:`SHORT_GAP_HOURS` — see that constant for why
    the threshold is not a human-scale number of hours.

    A user's first transaction has no predecessor, and its `hours_since_prev_txn`
    sentinel (999 for most users, see notes.md) sits *below* the threshold, so a
    naive comparison would flag all 5,000 of them as rapid repeats. They are
    excluded and left at 0: no predecessor means no interval, not a short one.
    """
    has_prev = _past_count(df) > 0
    return (has_prev & df[GAP_COL].clip(lower=0).lt(SHORT_GAP_HOURS)).astype(int)


def user_txn_count_last_57000h(df: pd.DataFrame) -> pd.Series:
    """How many of the user's own transactions fall in the preceding 57000h.

    A burst count rather than a single gap: it sees three transactions crammed
    into a window that the user normally fills with one. The current
    transaction is excluded, so this is 0 for a user's first one (and for
    anyone whose previous transaction is older than the window) — a genuine 0,
    not a missing baseline, so it is left as 0 rather than NaN.
    """
    return _past_count_within(df, RECENT_WINDOW_HOURS)


# --------------------------------------------------------------------------
# features — behavioral novelty
# --------------------------------------------------------------------------
def is_new_country_for_user(df: pd.DataFrame) -> pd.Series:
    """Country never seen before for this user (first txn excluded)."""
    return _is_new_for_user(df, "country")


def is_new_device_for_user(df: pd.DataFrame) -> pd.Series:
    """Device never seen before for this user (first txn excluded)."""
    return _is_new_for_user(df, "device_id")


def is_new_merchant_for_user(df: pd.DataFrame) -> pd.Series:
    """Merchant category never seen before for this user (first txn excluded)."""
    return _is_new_for_user(df, "merchant_category")


def is_new_channel_for_user(df: pd.DataFrame) -> pd.Series:
    """Channel never seen before for this user (first txn excluded)."""
    return _is_new_for_user(df, "channel")


def country_changed_from_prev(df: pd.DataFrame) -> pd.Series:
    """Country differs from the user's previous transaction."""
    return _changed_from_prev(df, "country")


def device_changed_from_prev(df: pd.DataFrame) -> pd.Series:
    """Device differs from the user's previous transaction."""
    return _changed_from_prev(df, "device_id")


def merchant_changed_from_prev(df: pd.DataFrame) -> pd.Series:
    """Merchant category differs from the user's previous transaction."""
    return _changed_from_prev(df, "merchant_category")


def channel_changed_from_prev(df: pd.DataFrame) -> pd.Series:
    """Channel differs from the user's previous transaction."""
    return _changed_from_prev(df, "channel")


def prev_country(df: pd.DataFrame) -> pd.Series:
    """Country of the user's previous transaction; NaN on their first.

    A nominal code, not a magnitude (30 countries, arbitrarily numbered), so it
    is only meaningful to a model that can split on it — a tree. Under the
    current logistic-regression baseline the scaler treats "country 23" as
    twice "country 11", which it is not.
    """
    return _prev_value(df, "country")


def prev_merchant_category(df: pd.DataFrame) -> pd.Series:
    """Merchant category of the user's previous transaction; NaN on their first.

    Nominal, with the same caveat as :func:`prev_country`.
    """
    return _prev_value(df, "merchant_category")


def user_distinct_devices_so_far(df: pd.DataFrame) -> pd.Series:
    """Running count of distinct devices used by the user, incl. this txn."""
    first = _is_first_occurrence(df, "device_id")
    return first.astype(int).groupby(df[USER_COL]).cumsum()


def user_distinct_countries_so_far(df: pd.DataFrame) -> pd.Series:
    """Running count of distinct countries used by the user, incl. this txn."""
    first = _is_first_occurrence(df, "country")
    return first.astype(int).groupby(df[USER_COL]).cumsum()


def user_country_share_so_far(df: pd.DataFrame) -> pd.Series:
    """Share of the user's prior transactions that came from this country."""
    return _prior_share(df, "country")


def user_merchant_share_so_far(df: pd.DataFrame) -> pd.Series:
    """Share of the user's prior transactions with this merchant category."""
    return _prior_share(df, "merchant_category")


# --------------------------------------------------------------------------
# features — history depth and global frequency encodings
# --------------------------------------------------------------------------
def user_history_length(df: pd.DataFrame) -> pd.Series:
    """How much history backs the deviation features for this row."""
    return _past_count(df)


def device_txn_count(df: pd.DataFrame) -> pd.Series:
    """How many transactions in the dataset share this device."""
    return df.groupby("device_id")["device_id"].transform("size")


def device_user_count(df: pd.DataFrame) -> pd.Series:
    """How many distinct users share this device — shared devices are suspect."""
    return df.groupby("device_id")[USER_COL].transform("nunique")


def merchant_freq(df: pd.DataFrame) -> pd.Series:
    """Global share of transactions for this merchant category.

    Merchants are near-uniformly distributed (notes.md), so this carries little
    signal on its own.
    """
    return df.groupby("merchant_category")["merchant_category"].transform("size") / len(df)


def country_freq(df: pd.DataFrame) -> pd.Series:
    """Global share of transactions for this country (near-uniform, see notes.md)."""
    return df.groupby("country")["country"].transform("size") / len(df)


# --------------------------------------------------------------------------
# active features — add the ones you want here
# --------------------------------------------------------------------------
FEATURES: list[FeatureFunc] = [
    merchant_changed_from_prev,
    country_changed_from_prev,
    device_changed_from_prev,
    channel_changed_from_prev,
    is_short_gap_since_prev,
    amount_diff_from_prev,
    amount_diff_from_user_mean,
    prev_country,
    prev_merchant_category,
    gap_zscore_vs_user,
]


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------
def build_feature_frame(
    df: pd.DataFrame,
    features: Sequence[FeatureFunc] | None = None,
) -> pd.DataFrame:
    """Return a copy of `df` with one column per feature appended.

    Rows are sorted per user while the features are computed, then put back in
    the input order.

    Args:
        df: Combined train+test frame (features are engineered before the split,
            see notes.md) or a single split.
        features: Feature functions to compute; defaults to :data:`FEATURES`.
            Each column is named after its function.
    """
    funcs = list(FEATURES if features is None else features)

    work = df if df.index.is_unique else df.reset_index(drop=True)
    original_index = work.index
    work = work.sort_values(SORT_COLUMNS, kind="mergesort").copy()

    for func in funcs:
        work[func.__name__] = func(work)

    return work.reindex(original_index)


def feature_columns(features: Sequence[FeatureFunc] | None = None) -> list[str]:
    """Column names that :func:`build_feature_frame` will produce."""
    return [func.__name__ for func in (FEATURES if features is None else features)]


def combined_path() -> Path:
    """Locate the combined train+test frame that features are engineered on.

    The file currently sits under `src/data/processed/` rather than the
    repo-root `data/processed/` that CLAUDE.md documents, so both are tried.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "data" / "processed" / "combined.csv",
        repo_root / "src" / "data" / "processed" / "combined.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "combined.csv not found in: " + ", ".join(str(p) for p in candidates)
    )


def load_combined() -> pd.DataFrame:
    """Load the combined train+test frame that features are engineered on."""
    return pd.read_csv(combined_path())


if __name__ == "__main__":
    featured = build_feature_frame(load_combined())
    cols = feature_columns()
    print(f"rows: {len(featured)}   features: {len(cols)}")
    if cols:
        print(featured[cols].describe().T.to_string())
    else:
        print("FEATURES is empty — add feature functions to it to build columns.")
