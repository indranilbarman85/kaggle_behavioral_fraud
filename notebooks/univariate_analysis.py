# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

DATA_PATH = Path(__file__).resolve().parents[1] / "src" / "data" / "processed" / "combined.csv"

df = pd.read_csv(DATA_PATH)

# label only exists for the train split (test rows are unlabeled holdout), so all
# label-conditioned univariate stats below are computed on the train subset only.
train = df[df["dataset"] == "train"].copy()
train["label"] = train["label"].astype(int)

CONTINUOUS = ["amount", "hours_since_prev_txn"]
CATEGORICAL = ["merchant_category", "country", "channel"]

# %%
# ---------------------------------------------------------------------------
# Continuous variables: describe, tail quantiles, histograms (linear + log)
# ---------------------------------------------------------------------------


def describe_continuous(data: pd.DataFrame, col: str) -> None:
    print(f"\n{'=' * 70}\n{col} — overall\n{'=' * 70}")
    print(data[col].describe(percentiles=[0.25, 0.5, 0.75]).to_string())

    print(f"\n{col} — by label")
    print(data.groupby("label")[col].describe(percentiles=[0.25, 0.5, 0.75]).to_string())

    print(f"\n{col} — tail quantiles")
    tail_q = [0.95, 0.99, 0.999]
    overall_tail = data[col].quantile(tail_q)
    by_label_tail = data.groupby("label")[col].quantile(tail_q).unstack(level=0)
    by_label_tail.columns = [f"label={c}" for c in by_label_tail.columns]
    tail_table = pd.concat([overall_tail.rename("overall"), by_label_tail], axis=1)
    print(tail_table.to_string())


def plot_continuous(data: pd.DataFrame, col: str, bins: int = 60) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for label, group in data.groupby("label"):
        axes[0].hist(group[col], bins=bins, alpha=0.5, label=f"label={label}", density=True)
    axes[0].set_title(f"{col} — histogram (linear)")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("density")
    axes[0].legend()

    positive = data[data[col] > 0]
    for label, group in positive.groupby("label"):
        axes[1].hist(np.log1p(group[col]), bins=bins, alpha=0.5, label=f"label={label}", density=True)
    axes[1].set_title(f"{col} — histogram (log1p scale)")
    axes[1].set_xlabel(f"log1p({col})")
    axes[1].set_ylabel("density")
    axes[1].legend()

    fig.tight_layout()
    plt.show()


for col in CONTINUOUS:
    describe_continuous(train, col)
    plot_continuous(train, col)

# %%
# ---------------------------------------------------------------------------
# Categorical variables: value counts / frequency bars, fraud rate per category
# ---------------------------------------------------------------------------


def describe_categorical(data: pd.DataFrame, col: str) -> None:
    counts = data[col].value_counts().sort_index()
    freq = (counts / counts.sum() * 100).round(2)
    fraud_rate = (data.groupby(col)["label"].mean() * 100).round(3)

    summary = pd.concat(
        [counts.rename("count"), freq.rename("freq_pct"), fraud_rate.rename("fraud_rate_pct")],
        axis=1,
    )
    print(f"\n{'=' * 70}\n{col} ({data[col].nunique()} distinct values)\n{'=' * 70}")
    print(summary.to_string())


def plot_categorical(data: pd.DataFrame, col: str) -> None:
    counts = data[col].value_counts().sort_index()
    fraud_rate = data.groupby(col)["label"].mean() * 100
    overall_rate = data["label"].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(counts) * 0.4), 4.5))

    axes[0].bar(counts.index.astype(str), counts.values, color="steelblue")
    axes[0].set_title(f"{col} — frequency")
    axes[0].set_ylabel("count")
    axes[0].tick_params(axis="x", rotation=90 if len(counts) > 10 else 0)

    axes[1].bar(fraud_rate.index.astype(str), fraud_rate.values, color="indianred")
    axes[1].axhline(overall_rate, color="black", linestyle="--", linewidth=1, label="overall rate")
    axes[1].set_title(f"{col} — fraud rate (%)")
    axes[1].set_ylabel("fraud rate (%)")
    axes[1].tick_params(axis="x", rotation=90 if len(counts) > 10 else 0)
    axes[1].legend()

    fig.tight_layout()
    plt.show()


for col in CATEGORICAL:
    describe_categorical(train, col)
    plot_categorical(train, col)

# %%
