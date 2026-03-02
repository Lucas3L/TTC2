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

from sklearn.preprocessing import MinMaxScaler

# imports utilitários
from src.utils.reproducibility import set_global_seed
from src.utils.helpers import ensure_dir, normalize_columns, add_lag_features
from src.utils.metrics import naive_market_smape
from src.models.evaluate import evaluate
from src.utils.metrics import naive_market_smape

# Caminhos base usando raiz do projeto para evitar dependência de cwd
INPUT_BASE = root / "Dados" / "preprocessed"
OUTPUT_BASE = root / "Resultados" / "baseline_strong"
ensure_dir(OUTPUT_BASE)

TARGET = "quantity"
FEATURES_BASE = ["onpromotion", "unitvalue", "holiday", "month", "day_of_week", "is_weekend"]

WINDOW = 7  # opcional, para gerar lags




# Forecast simples: usa último valor do lag mais recente
def naive_lag_forecast(df, feature="lag_1"):
    return df[feature]

# Processa cada arquivo
def process_file(csv_file):
    df = pd.read_csv(csv_file, parse_dates=["Date"])
    df = normalize_columns(df)
    df = df.dropna(subset=[TARGET])
    df = df.sort_values(["product_id", "date"])
    results = []

    for pid, g in df.groupby("product_id"):
        g = g.copy()
        g = add_lag_features(g, TARGET, lags=[1, 7, 14])
        # baseline only uses lag columns and target, ignore others when dropping
        feat_cols = ['lag_1','lag_7','lag_14', TARGET]
        g = g.dropna(subset=feat_cols)
        features = FEATURES_BASE + ["lag_1", "lag_7", "lag_14"]

        # split treino/val/test
        n = len(g)
        train_end = int(n * 0.7)
        val_end   = int(n * 0.85)
        train_df = g.iloc[:train_end].copy()
        val_df   = g.iloc[train_end:val_end].copy()
        test_df  = g.iloc[val_end:].copy()

        if len(train_df) < WINDOW or len(test_df) < 3:
            continue

        # log-transform target
        for subset in [train_df, val_df, test_df]:
            subset.loc[:, TARGET] = np.log1p(subset[TARGET].clip(lower=0))

        # normalização simples MinMax baseada no treino
        scaler = MinMaxScaler()
        # convert to float before scaling to avoid dtype errors
        train_df[features] = train_df[features].astype(float)
        val_df[features] = val_df[features].astype(float)
        test_df[features] = test_df[features].astype(float)
        
        train_df[features] = scaler.fit_transform(train_df[features])
        val_df[features] = scaler.transform(val_df[features])
        test_df[features] = scaler.transform(test_df[features])

        # previsão usando lag 1 (baseline forte)
        y_pred = naive_lag_forecast(test_df, "lag_1")
        y_true = np.expm1(test_df[TARGET])

        y_pred = np.expm1(y_pred)  # retorna à escala original

        metrics = evaluate(y_true, y_pred)

        results.append({
            "model": "baseline_strong",
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
        print(f"\nRodando baseline forte em: {market_name}")

        all_results = []
        for csv_file in market_path.glob("cat*.csv"):
            df_res = process_file(csv_file)
            if not df_res.empty:
                all_results.append(df_res)

        if all_results:
            final = pd.concat(all_results, ignore_index=True)
            out_file = OUTPUT_BASE / f"{market_name}_baseline_strong.csv"
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
