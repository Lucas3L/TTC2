import sys
from pathlib import Path
from typing import final
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
# --- Realiza a configuração para evitar estouro de memória no Baseline ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import gc
import numpy as np
import math
import traceback
import logging
import argparse

# --- Realiza imports corrigidos para evitar avisos do Pylance ---
from src.utils.reproducibility import set_global_seed
from src.utils.helpers import ensure_dir, normalize_columns, add_lag_features
from src.utils.metrics import naive_market_smape
from src.models.evaluate import evaluate
from src.config.model_params import COMMON_MODEL_PARAMS

try:
    from src.features.scenarios import apply_scenario
except ImportError:
    apply_scenario = None
# -------------------------------------------------------------

# Caminhos base definidos usando a raiz do projeto para evitar dependência de cwd
PROJECT_ROOT = ROOT
INPUT_BASE = PROJECT_ROOT / "Dados" / "split"

OUTPUT_BASE = PROJECT_ROOT / "Resultados" / "baseline_zero_aware"
ensure_dir(OUTPUT_BASE)

# Configuração do log de erros
ERROR_LOG = PROJECT_ROOT / "Resultados" / "errors.log"
logging.basicConfig(
    filename=ERROR_LOG,
    filemode="a",
    format="[%(asctime)s][baseline_zero_aware] %(levelname)s: %(message)s",
    level=logging.WARNING
)

TARGET = "quantity"

WINDOW = COMMON_MODEL_PARAMS["window_size"]
TRAIN_RATIO = COMMON_MODEL_PARAMS["train_ratio"]
VAL_RATIO = COMMON_MODEL_PARAMS.get("val_ratio", 0.0)

LAGS = COMMON_MODEL_PARAMS["lags"]
ROLLING_WINDOWS = COMMON_MODEL_PARAMS["rolling_windows"]
ROLLING_FEATURES = [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]

def naive_lag_forecast(df, feature="lag_1"):
    return df[feature].fillna(0)

def process_file(csv_file, scenario=None, date_from=None, date_to=None):
    df = pd.read_csv(csv_file, parse_dates=["date"])
    df = normalize_columns(df)
    
    if date_from is not None:
        df = df[df["date"] >= pd.to_datetime(date_from)]
    if date_to is not None:
        df = df[df["date"] <= pd.to_datetime(date_to)]
        
    # Realiza a aplicação do cenário para justiça comparativa
    if scenario is not None and apply_scenario is not None:
        try: df = apply_scenario(df, scenario)
        except Exception: pass
    # Fim da aplicação do cenário

    df = df.sort_values(["product_id", "date"])
    results = []
    pred_rows = []

    for pid, g in df.groupby("product_id"):
        g = g.copy().sort_values("date")
        try:
            g = add_lag_features(g, TARGET, lags=LAGS, rolling_windows=ROLLING_WINDOWS)
            lag_cols = [f"lag_{lag}" for lag in LAGS]
            feat_cols = lag_cols + ROLLING_FEATURES + [TARGET]

            # Realiza a mesma limpeza dos outros modelos
            # Garante-se que o Baseline só realize previsões nos mesmos dias que o LSTM/XGBoost
            g = g.dropna(subset=feat_cols)
            g = g[g[TARGET] != -99.0]
            # Fim da limpeza

            n = len(g)
            if n < WINDOW * 2: # Ignora produtos com poucos dados
                continue

            train_end = int(n * TRAIN_RATIO)
            val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
            test_df = g.iloc[val_end:].copy()

            if len(test_df) < 5: # Ignora produtos com poucos dados
                continue

            # Previsão Naive: Amanhã recebe o valor de Hoje
            # Após o dropna acima, não restam mais NaNs ou -99.0 para mascarar
            y_pred = test_df['lag_1']
            y_true = test_df[TARGET]

            if y_true.isna().all() or y_pred.isna().all() or np.isinf(y_true).all() or np.isinf(y_pred).all():
                continue

            metrics = evaluate(y_true, y_pred)

            if any([math.isnan(v) or math.isinf(v) for v in metrics.values()]):
                continue

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
        except Exception as e:
            continue
        finally:
            gc.collect()

    return pd.DataFrame(results), pd.DataFrame(pred_rows)

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
    
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir():
            continue

        market_name = market_path.name
        print(f"\nRodando baseline_zero_aware em: {market_name}")

        # 1. Filtra para ler apenas os splits do cenário atual (Correção de Nomenclatura)
        if args.scenario is not None:
            # Se o cenário for "volume", ele busca por "vol" que é como o split salvou.
            termo_busca = "vol" if args.scenario == "volume" else args.scenario
            arquivos = list(market_path.glob(f"*_{termo_busca}.csv"))
        else:
            arquivos = list(market_path.glob('cat*.csv'))
            
        if not arquivos:
            print(f" [Aviso] Nenhum arquivo encontrado para o cenário {args.scenario} em {market_name}")
            continue

        all_results = []
        all_predictions = []
        
        # 2. Processa os arquivos isolados
        for csv_file in arquivos:
            df_res, df_pred = process_file(
                csv_file, scenario=args.scenario,
                date_from=args.date_from, date_to=args.date_to
            )
            if not df_res.empty:
                all_results.append(df_res)
            if not df_pred.empty:
                all_predictions.append(df_pred)

        # 3. Exporta e Imprime no formato do Orquestrador
        if all_results:
            final = pd.concat(all_results, ignore_index=True)
            suffix = f"_{args.scenario}" if args.scenario else ""
            out_file = OUTPUT_BASE / f"{market_name}_baseline{suffix}.csv"
            ensure_dir(OUTPUT_BASE)
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")
            
            if all_predictions:
                pred_file = OUTPUT_BASE / f"{market_name}_baseline{suffix}_predictions.csv"
                ensure_dir(OUTPUT_BASE)
                pd.concat(all_predictions, ignore_index=True).to_csv(pred_file, index=False)
                print(f"  Curva real vs predito salva em {pred_file}")
            
            # IMPRESSÃO ISOLADA PARA O ORQUESTRADOR
            for cat_file, group in final.groupby('arquivo'):
                print(f"[{cat_file}] FINAL sMAPE: {group['smape'].mean():.4f}")
                print(f"[{cat_file}] MAE: {group['mae'].mean():.4f}")
                print(f"[{cat_file}] RMSE: {group['rmse'].mean():.4f}")
        else:
            market_smap = naive_market_smape(market_path)
            print(f"[fallback_geral.csv] FINAL sMAPE: {market_smap:.4f}")
            print(f"[fallback_geral.csv] MAE: 0.0000")
            print(f"[fallback_geral.csv] RMSE: 0.0000")

if __name__ == "__main__":
    main()