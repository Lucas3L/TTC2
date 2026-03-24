
# --- sys.path bootstrap: ensure project root is in sys.path before any src import ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import gc
import argparse


# --- (Optional) Debug sys.path ---
print(f"[DEBUG][baseline] sys.path: {sys.path}")

# imports utilitários
from src.utils.reproducibility import set_global_seed
from src.utils.helpers import ensure_dir, normalize_columns, add_lag_features
from src.utils.metrics import naive_market_smape
from src.models.evaluate import evaluate
from src.config.model_params import COMMON_MODEL_PARAMS

# Caminhos base usando raiz do projeto para evitar dependência de cwd

PROJECT_ROOT = ROOT
INPUT_BASE = PROJECT_ROOT / "Dados" / "preprocessed"
OUTPUT_BASE = PROJECT_ROOT / "Resultados" / "baseline_zero_aware"
ensure_dir(OUTPUT_BASE)

TARGET = "quantity"

WINDOW = COMMON_MODEL_PARAMS["window_size"]
TRAIN_RATIO = COMMON_MODEL_PARAMS["train_ratio"]
VAL_RATIO = COMMON_MODEL_PARAMS["val_ratio"]
LAGS = COMMON_MODEL_PARAMS["lags"]
ROLLING_WINDOWS = COMMON_MODEL_PARAMS["rolling_windows"]
ROLLING_FEATURES = [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
def naive_lag_forecast(df, feature="lag_1"):
    return df[feature].fillna(0)

def process_file(csv_file, scenario=None, date_from=None, date_to=None):
    df = pd.read_csv(csv_file, parse_dates=["date"])
    df = normalize_columns(df)
    df = df.dropna(subset=[TARGET])
    if date_from is not None:
        df = df[df["date"] >= pd.to_datetime(date_from)]
    if date_to is not None:
        df = df[df["date"] <= pd.to_datetime(date_to)]
    df = df.sort_values(["product_id", "date"])
    results = []
    pred_rows = []

    for pid, g in df.groupby("product_id"):
        g = g.copy().sort_values("date")
        
        g = add_lag_features(g, TARGET, lags=LAGS, rolling_windows=ROLLING_WINDOWS)
        lag_cols = [f"lag_{lag}" for lag in LAGS]
        feat_cols = lag_cols + ROLLING_FEATURES + [TARGET]
        g = g.dropna(subset=feat_cols)
        
        n = len(g)
        if n < 20:
            continue

        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        test_df = g.iloc[val_end:].copy()
        
        if len(test_df) < 3:
            continue

        y_pred = test_df['lag_1'].fillna(0)
        y_true = test_df[TARGET]
        metrics = evaluate(y_true, y_pred)

        results.append({
            "model": "baseline_zero_aware",
            "arquivo": csv_file.name,
            "product_id": pid,
            "mae": metrics["MAE"],
            "rmse": metrics["RMSE"],
            "smape": metrics["sMAPE"]
        })
        for dt, y_t, y_p in zip(test_df["date"], y_true, y_pred):
            pred_rows.append({
                "model": "baseline_zero_aware",
                "arquivo": csv_file.name,
                "product_id": pid,
                "date": dt,
                "y_true": float(y_t),
                "y_pred": float(y_p),
                "scenario": scenario
            })

        gc.collect()

    return pd.DataFrame(results), pd.DataFrame(pred_rows)

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=None,
                        help="Cenário a ser aplicado (volume, price, kmeans)")
    parser.add_argument("--date-from", type=str, default=None,
                        help="Data inicial (YYYY-MM-DD) para filtrar histórico")
    parser.add_argument("--date-to", type=str, default=None,
                        help="Data final (YYYY-MM-DD) para filtrar histórico")
    args = parser.parse_args()
    
    set_global_seed(args.seed)
    
    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir():
            continue

        market_name = market_path.name
        print(f"\nRodando baseline_zero_aware em: {market_name}")

        all_results = []
        all_predictions = []
        for csv_file in market_path.glob("cat*.csv"):
            df_res, df_pred = process_file(
                csv_file, scenario=args.scenario,
                date_from=args.date_from, date_to=args.date_to
            )
            if not df_res.empty:
                all_results.append(df_res)
            if not df_pred.empty:
                all_predictions.append(df_pred)

        if all_results:
            final = pd.concat(all_results, ignore_index=True)
            out_file = OUTPUT_BASE / f"{market_name}_baseline_zero_aware.csv"
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")
            if all_predictions:
                pred_file = OUTPUT_BASE / f"{market_name}_baseline_zero_aware_predictions.csv"
                pd.concat(all_predictions, ignore_index=True).to_csv(pred_file, index=False)
                print(f"  Curva real vs predito salva em {pred_file}")
            
            mean_smape = final['smape'].mean()
            print(f"FINAL sMAPE: {mean_smape:.4f}")
        else:
            market_smap = naive_market_smape(market_path)
            print(f"FALLBACK market sMAPE: {market_smap:.4f}")
            print(f"FINAL sMAPE: {market_smap:.4f}")

# PONTO DE ENTRADA
if __name__ == "__main__":
    main()
