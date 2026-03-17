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




# Forecast simples: usa último valor conhecido (lag_1)
def naive_lag_forecast(df, feature="lag_1"):
    return df[feature].fillna(0)

# Processa cada arquivo
def process_file(csv_file):
    df = pd.read_csv(csv_file, parse_dates=["date"])
    df = normalize_columns(df)
    df = df.dropna(subset=[TARGET])
    df = df.sort_values(["product_id", "date"])
    results = []

    for pid, g in df.groupby("product_id"):
        g = g.copy().sort_values("date")
        
        # Adicionar lags ANTES da divisão treino/teste
        g = add_lag_features(g, TARGET, lags=[1, 7, 14])
        feat_cols = ['lag_1','lag_7','lag_14', TARGET]
        g = g.dropna(subset=feat_cols)
        
        n = len(g)
        if n < 20:
            continue

        # Dividir em treino/teste (80% teste como nos outros modelos)
        split_idx = int(n * 0.8)
        test_df = g.iloc[split_idx:].copy()
        
        if len(test_df) < 3:
            continue

        # Baseline simples: usar lag_1 como previsão
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
