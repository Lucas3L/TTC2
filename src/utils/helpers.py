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


def add_lag_features(
    df: pd.DataFrame,
    target: str = "quantity",
    lags=None,
    rolling_windows=None,
) -> pd.DataFrame:
    """Append lag and rolling mean features for the specified target column.

    The function returns a new frame with lag_{lag} columns and
    rolling_mean_{window} columns. Lags/rollings are computed por produto
    (quando a coluna ``product_id`` existir) para evitar vazamento entre séries.
    As médias móveis usam apenas histórico passado (``shift(1)``).
    """
    if lags is None:
        lags = [1, 7, 14]
    if rolling_windows is None:
        rolling_windows = [3, 7, 14]

    out = df.copy()
    group_cols = ["product_id"] if "product_id" in out.columns else None

    if group_cols is not None:
        out = out.sort_values(group_cols + (["date"] if "date" in out.columns else []))
        grouped_target = out.groupby(group_cols, sort=False)[target]
        for lag in lags:
            out[f"lag_{lag}"] = grouped_target.shift(lag)
        for w in rolling_windows:
            out[f"rolling_mean_{w}"] = grouped_target.transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
    else:
        for lag in lags:
            out[f"lag_{lag}"] = out[target].shift(lag)
        for w in rolling_windows:
            out[f"rolling_mean_{w}"] = out[target].shift(1).rolling(w, min_periods=1).mean()

    return out



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
