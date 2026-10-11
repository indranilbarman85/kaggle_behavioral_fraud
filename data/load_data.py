import pandas as pd
from pathlib import Path


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Load a CSV dataset from the raw data directory."""
    if path is None:
        base_dir = Path(__file__).resolve().parents[2]
        path = str(base_dir / "data" / "raw" / "train.csv")

    df = pd.read_csv(path)
    return df


def report_missing_data(path: str | None = None) -> pd.DataFrame:
    """Return missing-data counts and percentages for a CSV dataset."""
    df = load_dataset(path)
    missing_count = df.isna().sum()
    missing_pct = df.isna().mean() * 100
    report = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "dtype": df.dtypes,
        }
    )
    return report.sort_values(by="missing_pct", ascending=False)


def report_missing_data_for_raw_files() -> dict[str, pd.DataFrame]:
    """Report missing data for train.csv and test.csv in data/raw."""
    base_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    reports = {}

    for name in ["train", "test"]:
        file_path = base_dir / f"{name}.csv"
        if file_path.exists():
            reports[name] = report_missing_data(str(file_path))

    return reports


if __name__ == "__main__":
    reports = report_missing_data_for_raw_files()
    for name, report in reports.items():
        print(f"Missing data report for {name}.csv")
        print(report.to_string())
        print("\n")
