"""Univariate analysis of engineered features against the fraud label.

Meant to be driven from `notebooks/exploration.ipynb`::

    from src.features.analysis import univariate, univariate_summary
    from src.features import build_features as bf

    df = bf.load_combined()

    table = univariate(df, bf.amount_zscore_vs_user)     # one feature, with plots
    summary = univariate_summary(df, [bf.amount_log, bf.gap_log])  # rank many

A feature is passed either as the function itself (it gets computed on the fly
via :func:`src.features.build_features.build_feature_frame`, so the per-user
sorting and the strictly-prior rule are respected) or as the name of a column
that is already on the frame.

Rows with a missing label are dropped — `combined.csv` holds the test rows too,
and those have no label. Missing *feature* values are kept and reported as their
own bin, because "no history yet" is itself informative.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features.build_features import LABEL_COL, FeatureFunc, build_feature_frame

MISSING_LABEL = "(missing)"

# Categorical slots 1 and 2 of the reference palette, plus recessive ink.
COLOR_LEGIT = "#2a78d6"
COLOR_FRAUD = "#eb6834"
COLOR_INK = "#52514e"
COLOR_GRID = "#d8d7d2"

FeatureRef = str | FeatureFunc


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------
def _resolve(df: pd.DataFrame, feature: FeatureRef) -> tuple[pd.Series, str]:
    """Return the feature's values and its name, computing it if needed."""
    if isinstance(feature, str):
        if feature not in df.columns:
            raise KeyError(f"{feature!r} is not a column of the frame")
        return df[feature], feature

    name = feature.__name__
    if name in df.columns:
        return df[name], name
    return build_feature_frame(df, features=[feature])[name], name


def _bin_values(
    values: pd.Series, bins: int, max_discrete: int
) -> tuple[pd.Series, list, bool]:
    """Bucket `values`; returns the buckets, their order, and if it's discrete.

    Few distinct values (flags, small counts) are kept as-is; anything else is
    cut into quantile bins so every bucket carries a comparable row count.
    NaNs become their own `(missing)` bucket, kept last.
    """
    present = values.dropna()
    discrete = present.nunique() <= max_discrete

    if discrete:
        buckets = present.astype("object")
    else:
        try:
            buckets = pd.qcut(present, q=bins, duplicates="drop").astype("object")
        except ValueError:  # too few distinct values for any quantile cut
            buckets = present.astype("object")
            discrete = True

    order = sorted(pd.unique(buckets), key=lambda b: getattr(b, "left", b))
    buckets = buckets.reindex(values.index)
    if buckets.isna().any():
        buckets = buckets.fillna(MISSING_LABEL)
        order.append(MISSING_LABEL)
    return buckets, order, discrete


def _bin_table(buckets: pd.Series, y: pd.Series, order: list) -> pd.DataFrame:
    """Fraud rate, lift and weight-of-evidence per bin."""
    grouped = y.groupby(buckets, observed=True)
    table = pd.DataFrame({"n": grouped.size(), "frauds": grouped.sum()})
    table = table.reindex(order)
    table["fraud_rate"] = table["frauds"] / table["n"]
    table["share_of_rows"] = table["n"] / len(y)
    table["lift"] = table["fraud_rate"] / y.mean()

    # 0.5 smoothing keeps WOE finite in bins with no frauds (1.64% base rate).
    pos = (table["frauds"] + 0.5) / (y.sum() + 0.5 * len(table))
    neg = (table["n"] - table["frauds"] + 0.5) / ((1 - y).sum() + 0.5 * len(table))
    table["woe"] = np.log(pos / neg)
    table["iv"] = (pos - neg) * table["woe"]
    return table


def _auc(values: pd.Series, y: pd.Series) -> float:
    """ROC-AUC of the raw feature, over the rows where it is defined."""
    ok = values.notna()
    if ok.sum() == 0 or y[ok].nunique() < 2:
        return float("nan")
    return roc_auc_score(y[ok], values[ok])


def _labelled(df: pd.DataFrame, label_col: str) -> pd.Index:
    if label_col not in df.columns:
        raise KeyError(f"no {label_col!r} column — pass the frame that has labels")
    return df.index[df[label_col].notna()]


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def _plot(
    name: str,
    table: pd.DataFrame,
    values: pd.Series,
    y: pd.Series,
    discrete: bool,
    base_rate: float,
    figsize: tuple[float, float],
) -> None:
    fig, (ax_rate, ax_dist) = plt.subplots(1, 2, figsize=figsize)
    labels = [str(b) for b in table.index]

    # --- fraud rate per bin -------------------------------------------------
    ax_rate.bar(labels, table["fraud_rate"] * 100, color=COLOR_LEGIT, width=0.68)
    ax_rate.axhline(base_rate * 100, color=COLOR_INK, linestyle="--", linewidth=1)
    ax_rate.text(
        1.0, base_rate * 100, f"  base {base_rate:.2%}",
        transform=ax_rate.get_yaxis_transform(), va="bottom", ha="right",
        fontsize=8, color=COLOR_INK,
    )
    for x, (rate, n) in enumerate(zip(table["fraud_rate"], table["n"])):
        if np.isfinite(rate):
            ax_rate.annotate(
                f"n={n:,}", (x, rate * 100), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=7, color=COLOR_INK,
            )
    ax_rate.set_title(f"{name} — fraud rate by bin", fontsize=10, loc="left")
    ax_rate.set_ylabel("fraud rate (%)")

    # --- distribution by class ---------------------------------------------
    if discrete:
        shares = (
            pd.crosstab(values.fillna(MISSING_LABEL), y, normalize="columns")
            .reindex(table.index)
            .fillna(0)
        )
        pos = np.arange(len(shares))
        ax_dist.bar(pos - 0.19, shares.get(0, 0) * 100, width=0.36,
                    color=COLOR_LEGIT, label="legit")
        ax_dist.bar(pos + 0.19, shares.get(1, 0) * 100, width=0.36,
                    color=COLOR_FRAUD, label="fraud")
        ax_dist.set_xticks(pos, [str(b) for b in shares.index])
        ax_dist.set_ylabel("share of class (%)")
    else:
        present = values.dropna()
        lo, hi = present.quantile([0.005, 0.995])
        edges = np.linspace(lo, hi, 41) if hi > lo else 41
        for cls, color, lab in ((0, COLOR_LEGIT, "legit"), (1, COLOR_FRAUD, "fraud")):
            subset = present[y.reindex(present.index) == cls]
            if len(subset):
                ax_dist.hist(subset.clip(lo, hi), bins=edges, density=True,
                             histtype="step", linewidth=2, color=color, label=lab)
        ax_dist.set_ylabel("density")
        ax_dist.set_xlabel(f"{name} (clipped to 0.5–99.5 pct)")
    ax_dist.set_title("distribution by class", fontsize=10, loc="left")
    ax_dist.legend(frameon=False, fontsize=8)

    for ax in (ax_rate, ax_dist):
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8, colors=COLOR_INK)
    plt.setp(ax_rate.get_xticklabels(), rotation=45, ha="right")
    if discrete:
        plt.setp(ax_dist.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------
def univariate(
    df: pd.DataFrame,
    feature: FeatureRef,
    label_col: str = LABEL_COL,
    bins: int = 10,
    max_discrete: int = 12,
    plot: bool = True,
    figsize: tuple[float, float] = (12.0, 4.2),
    verbose: bool = True,
) -> pd.DataFrame:
    """Univariate view of one feature against the label.

    Prints headline numbers (coverage, ROC-AUC, information value), draws the
    fraud rate per bin next to the class-conditional distribution, and returns
    the per-bin table.

    Args:
        df: Frame holding the raw columns; the feature is computed if absent.
        feature: A feature function from `src.features.build_features`, or the
            name of a column already on `df`.
        label_col: Binary target column. Rows where it is null are ignored.
        bins: Quantile bins for continuous features.
        max_discrete: At or below this many distinct values the feature is
            treated as categorical and each value gets its own bin.
        plot: Set False for the table only.
        figsize: Figure size passed to matplotlib.
        verbose: Print the headline numbers.

    Returns:
        One row per bin (plus a `(missing)` row when the feature has NaNs) with
        `n`, `frauds`, `fraud_rate`, `share_of_rows`, `lift`, `woe`, `iv`.
    """
    idx = _labelled(df, label_col)
    values, name = _resolve(df, feature)
    values, y = values.loc[idx], df.loc[idx, label_col].astype(int)

    buckets, order, discrete = _bin_values(values, bins=bins, max_discrete=max_discrete)
    table = _bin_table(buckets, y, order)
    auc = _auc(values, y)
    base_rate = y.mean()

    if verbose:
        print(
            f"{name}\n"
            f"  rows {len(y):,}   missing {values.isna().mean():.1%}   "
            f"distinct {values.nunique():,}\n"
            f"  base fraud rate {base_rate:.3%}   AUC {auc:.4f} "
            f"(|AUC-0.5| = {abs(auc - 0.5):.4f})   IV {table['iv'].sum():.4f}"
        )

    if plot:
        _plot(name, table, values, y, discrete, base_rate, figsize)
    return table


def univariate_summary(
    df: pd.DataFrame,
    features: Sequence[FeatureRef],
    label_col: str = LABEL_COL,
    bins: int = 10,
    max_discrete: int = 12,
) -> pd.DataFrame:
    """Rank many features by univariate strength — no plots.

    Returns one row per feature with `auc`, `auc_gap` (|AUC - 0.5|, the sortable
    strength since a feature can point either way), `iv`, `missing_rate`,
    `max_lift` and `n_bins`, sorted by `auc_gap`.
    """
    rows = []
    for feature in features:
        table = univariate(
            df, feature, label_col=label_col, bins=bins,
            max_discrete=max_discrete, plot=False, verbose=False,
        )
        values, name = _resolve(df, feature)
        idx = _labelled(df, label_col)
        values, y = values.loc[idx], df.loc[idx, label_col].astype(int)
        auc = _auc(values, y)
        rows.append(
            {
                "feature": name,
                "auc": auc,
                "auc_gap": abs(auc - 0.5),
                "iv": table["iv"].sum(),
                "missing_rate": values.isna().mean(),
                "max_lift": table["lift"].max(),
                "n_bins": len(table),
            }
        )
    return (
        pd.DataFrame(rows)
        .set_index("feature")
        .sort_values("auc_gap", ascending=False)
    )


def all_feature_funcs() -> list[FeatureFunc]:
    """Every feature function defined in `src.features.build_features`.

    Handy for a first sweep: ``univariate_summary(df, all_feature_funcs())``.
    """
    import inspect

    from src.features import build_features as bf

    return [
        obj
        for name, obj in vars(bf).items()
        if inspect.isfunction(obj)
        and obj.__module__ == bf.__name__
        and not name.startswith("_")
        and name not in {"build_feature_frame", "feature_columns", "load_combined"}
    ]
