from pathlib import Path
import pandas as pd

def ensure_dir(path: Path):
    """Creates a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercases, strips whitespace and replaces spaces with underscores in column names."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def add_lag_features(df: pd.DataFrame, target: str = "quantity", lags=[1, 7, 14]) -> pd.DataFrame:
    """Append lag and rolling mean features for the specified target column.

    The function returns a new frame with lag_{lag} columns and rolling_mean_3/7/14.
    """
    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)
    # rolling means windows fixed for convenience
    df["rolling_mean_3"] = df[target].rolling(3).mean()
    df["rolling_mean_7"] = df[target].rolling(7).mean()
    df["rolling_mean_14"] = df[target].rolling(14).mean()
    return df



def add_intermittent_features(df: pd.DataFrame, target: str = "quantity") -> pd.DataFrame:

    out = df.copy()
    y = out[target].fillna(0).astype(float)

    out["is_zero"] = (y <= 0).astype(int)

    zero_run = []
    streak = 0
    for val in out["is_zero"].values:
        if val == 1:
            streak += 1
        else:
            streak = 0
        zero_run.append(streak)
    out["zero_run_length"] = pd.Series(zero_run, index=out.index).shift(1).fillna(0)

    occ = (y > 0).astype(float)
    out["occurrence_rate_7"] = occ.shift(1).rolling(7, min_periods=1).mean()
    out["occurrence_rate_14"] = occ.shift(1).rolling(14, min_periods=1).mean()

    pos = y.where(y > 0)
    out["positive_mean_7"] = pos.shift(1).rolling(7, min_periods=1).mean()
    out["positive_mean_7"] = out["positive_mean_7"].fillna(0)

    return out