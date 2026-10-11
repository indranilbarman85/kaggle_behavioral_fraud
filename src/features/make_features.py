"""Write `combined.csv` back out with the active feature columns attached.

Run from the repo root after changing
:data:`src.features.build_features.FEATURES`::

    python -m src.features.make_features

The output lands next to the input as `combined_new_features.csv`, keeping the
raw columns (and the `dataset` flag that splits train from test) so downstream
code can slice it straight back into the two sets.

Rerunning is safe: feature columns left over from a previous run are dropped
before the current :data:`FEATURES` are computed, so the output always reflects
exactly what is switched on right now.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.build_features import (
    FEATURES,
    build_feature_frame,
    combined_path,
    feature_columns,
)

OUTPUT_NAME = "combined_new_features.csv"


def build_combined_features() -> pd.DataFrame:
    """Return the combined frame with one column per active feature."""
    df = pd.read_csv(combined_path())

    # A previous run may have written feature columns into the input; drop any
    # so a renamed or switched-off feature cannot linger in the output.
    stale = [col for col in feature_columns() if col in df.columns]
    if stale:
        df = df.drop(columns=stale)

    return build_feature_frame(df)


def write_combined_features(output_path: Path | None = None) -> Path:
    """Build the feature frame and write it as CSV. Returns the path written."""
    path = output_path or combined_path().with_name(OUTPUT_NAME)
    build_combined_features().to_csv(path, index=False)
    return path


if __name__ == "__main__":
    if not FEATURES:
        raise SystemExit(
            "FEATURES is empty — add feature functions to it in "
            "src/features/build_features.py before building the dataset."
        )

    featured = build_combined_features()
    cols = feature_columns()
    out = combined_path().with_name(OUTPUT_NAME)
    featured.to_csv(out, index=False)

    print(f"rows: {len(featured)}   features: {len(cols)}")
    print(featured[cols].describe().T.to_string())
    print(f"\nwrote {out} ({out.stat().st_size:,} bytes)")
