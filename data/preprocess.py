import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocess_pipeline() -> Pipeline:
    """Create a simple preprocessing pipeline for numeric features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def prepare_features(df: pd.DataFrame, target_col: str = "Class") -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for training."""
    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col] if target_col in df.columns else df.iloc[:, -1]

    numeric_cols = X.select_dtypes(include=["number"]).columns
    X_num = X[numeric_cols]
    return X_num, y
