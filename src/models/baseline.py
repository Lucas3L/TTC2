from pathlib import Path
import os
import sys
import gc
import pandas as pd
import numpy as np
import argparse

# Configurar path ANTES de importar módulos locais
file_path = Path(__file__).resolve()
root = file_path.parents[2]
if str(root) not in sys.path:
    sys.path.append(str(root))

# imports utilitários
from src.utils.reproducibility import set_global_seed
from src.utils.helpers import ensure_dir, normalize_columns, add_lag_features
from src.utils.metrics import naive_market_smape
from src.models.evaluate import evaluate

# Caminhos base usando raiz do projeto para evitar dependência de cwd
INPUT_BASE = root / "Dados" / "preprocessed"
OUTPUT_BASE = root / "Resultados" / "baseline_zero_aware"
ensure_dir(OUTPUT_BASE)

TARGET = "quantity"

WINDOW = 7  # opcional, para gerar lags




# Forecast simples: usa último valor do lag mais recente
def naive_lag_forecast(df, feature="lag_1"):
    return df[feature]

def zero_aware_forecast(row):
    zero_run = row.get("zero_run_length", 0)
    occ_rate = row.get("occurrence_rate_14", 1.0)
    rolling_mean = row.get("rolling_mean_7", 0)
    pos_mean = row.get("positive_mean_7", 0)

    # regra intermitente
    if zero_run >= 3 and occ_rate <= 0.4:
        return 0.0

    if pos_mean > 0:
        return pos_mean

    return rolling_mean

# Processa cada arquivo
def process_file(csv_file):
    df = pd.read_csv(csv_file, parse_dates=["date"])
    df = normalize_columns(df)
    df = df.dropna(subset=[TARGET])
    df = df.sort_values(["product_id", "date"])
    results = []

    for pid, g in df.groupby("product_id"):
        g = g.copy().sort_values("date")
        g["zero_run_length"] = (
            (g[TARGET] == 0)
            .astype(int)
            .groupby((g[TARGET] != 0).cumsum())
            .cumsum()
        )

        g["occurrence_rate_14"] = (
            (g[TARGET] > 0)
            .rolling(14, min_periods=1)
            .mean()
        )

        g["positive_mean_7"] = (
            g[TARGET]
            .where(g[TARGET] > 0)
            .rolling(7, min_periods=1)
            .mean()
        )

        g["rolling_mean_7"] = (
            g[TARGET]
            .rolling(7, min_periods=1)
            .mean()
        )
        g = add_lag_features(g, TARGET, lags=[1, 7, 14])
        # baseline only uses lag columns and target, ignore others when dropping
        feat_cols = ['lag_1','lag_7','lag_14', TARGET]
        g = g.dropna(subset=feat_cols)
        
        n = len(g)

        if n < 20:
            continue

        split_idx = int(n * 0.8)
        test_df  = g.iloc[split_idx:].copy()
        
        if len(test_df) < 3:
            continue

        y_pred = test_df.apply(zero_aware_forecast, axis=1)
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

        gc.collect()

    return pd.DataFrame(results)

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=None,
                        help="Cenário a ser aplicado (volume, price, kmeans)")
    args = parser.parse_args()
    
    set_global_seed(args.seed)
    
    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir():
            continue

        market_name = market_path.name
        print(f"\nRodando baseline_zero_aware em: {market_name}")

        all_results = []
        for csv_file in market_path.glob("cat*.csv"):
            df_res = process_file(csv_file)
            if not df_res.empty:
                all_results.append(df_res)

        if all_results:
            final = pd.concat(all_results, ignore_index=True)
            out_file = OUTPUT_BASE / f"{market_name}_baseline_zero_aware.csv"
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")
            
            mean_smape = final['smape'].mean()
            print(f"FINAL sMAPE: {mean_smape:.4f}")
        else:
            market_smap = naive_market_smape(market_path)
            print(f"FALLBACK market sMAPE: {market_smap:.4f}")
            print(f"FINAL sMAPE: {market_smap:.4f}")

# PONTO DE ENTRADA
if __name__ == "__main__":
    main()
