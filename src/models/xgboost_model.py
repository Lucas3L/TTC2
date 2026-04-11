import os
# --- VACINA CONTRA O SEGFAULT DO WINDOWS ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gc
import logging
import traceback
import pandas as pd
import numpy as np
import random
import argparse

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from src.utils.helpers import ensure_dir, normalize_columns, add_lag_features
from src.models.evaluate import evaluate
from src.utils.metrics import naive_market_smape
from src.config.model_params import COMMON_MODEL_PARAMS

try:
    from src.features.scenarios import apply_scenario
except ImportError:
    apply_scenario = None

TARGET = 'quantity'
FEATURES_BASE = COMMON_MODEL_PARAMS["features_base"]
LAGS = COMMON_MODEL_PARAMS["lags"]
ROLLING_WINDOWS = COMMON_MODEL_PARAMS["rolling_windows"]
QTY_FEATURES = [f"lag_{lag}" for lag in LAGS] + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
CYCLIC_FEATURES = ["day_sin", "day_cos", "month_sin", "month_cos"]

WINDOW = COMMON_MODEL_PARAMS["window_size"]
TRAIN_RATIO = COMMON_MODEL_PARAMS["train_ratio"]
VAL_RATIO = COMMON_MODEL_PARAMS["val_ratio"]

INPUT_BASE = ROOT / "Dados" / "preprocessed"
OUTPUT_BASE = ensure_dir(ROOT / "Resultados" / "xgb")

def compute_market_max(market_path):
    max_vals = []
    for csv_file in market_path.glob("cat*.csv"):
        try:
            df = pd.read_csv(csv_file)
            df = normalize_columns(df)
            if TARGET in df.columns:
                max_vals.append(df[TARGET].max())
        except Exception:
            continue
    market_max = np.nanmax(max_vals) if max_vals else np.nan
    if not np.isfinite(market_max) or market_max <= 0:
        market_max = 1.0
    return float(market_max)

def process_file(csv_file, market_max, scenario=None, date_from=None, date_to=None):
    df = pd.read_csv(csv_file, parse_dates=['date'])
    df = normalize_columns(df)
    
    if date_from: df = df[df["date"] >= pd.to_datetime(date_from)]
    if date_to: df = df[df["date"] <= pd.to_datetime(date_to)]

    if scenario is not None and apply_scenario is not None:
        try: df = apply_scenario(df, scenario)
        except Exception: pass

    le = LabelEncoder()
    df['product_id_encoded'] = le.fit_transform(df['product_id'])

    df = add_lag_features(df, TARGET, lags=LAGS, rolling_windows=ROLLING_WINDOWS)
    
    features = [c for c in FEATURES_BASE if c in df.columns] + \
               [c for c in QTY_FEATURES if c in df.columns] + \
               [c for c in CYCLIC_FEATURES if c in df.columns]

    # ==============================================================
    # INJEÇÃO DO CENÁRIO PARA O XGBOOST
    # ==============================================================
    if scenario is not None:
        col_map = {"volume": "volume_cluster", "price": "price_cluster", "kmeans": "kmeans_cluster"}
        cluster_col = col_map.get(scenario)
        if cluster_col and cluster_col in df.columns:
            df[cluster_col] = LabelEncoder().fit_transform(df[cluster_col].astype(str))
            if cluster_col not in features:
                features.append(cluster_col)
    # ==============================================================

    feat_cols = features + [TARGET]
    df = df.dropna(subset=feat_cols)
    df = df[df[TARGET] != -99.0]
    df = df.sort_values(['product_id_encoded', 'date'])

    train_list, val_list, test_list = [], [], []
    
    # 1. Separar o tempo de cada produto (mas ainda não treinar!)
    for pid_encoded, g in df.groupby('product_id_encoded'):
        g = g.copy()
        g[TARGET] = g[TARGET].clip(lower=0) / market_max
        if "unitvalue" in g.columns: g["unitvalue"] = g["unitvalue"].clip(lower=0)
        for c in QTY_FEATURES:
            if c in g.columns: g[c] = g[c] / market_max

        n = len(g)
        if n < WINDOW * 2: continue
            
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_list.append(g.iloc[:train_end].copy())
        val_list.append(g.iloc[train_end:val_end].copy())
        test_list.append(g.iloc[val_end:].copy())

    if not train_list or not test_list: return pd.DataFrame(), pd.DataFrame()

    # 2. Juntar tudo numa matriz Global
    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    X_train = train_df[features].values
    train_y = train_df[TARGET].values
    X_val = val_df[features].values
    val_y = val_df[TARGET].values
    X_test = test_df[features].values
    test_y = test_df[TARGET].values

    if X_train.shape[0] == 0: return pd.DataFrame(), pd.DataFrame()

    # 3. Treinar 1 Único XGBoost para a Loja
    print(f" -> [XGBoost GLOBAL] Iniciando treino | {len(X_train)} registros...")
    sample_weight = np.where(train_y > 0, 3.0, 1.0)
    xgb_cfg = COMMON_MODEL_PARAMS["training_by_model"]["xgboost"]
    
    model = XGBRegressor(
        n_estimators=xgb_cfg["n_estimators"],
        learning_rate=xgb_cfg["learning_rate"],
        max_depth=xgb_cfg["max_depth"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        random_state=42,
        verbosity=0
    )

    try:
        model.fit(
            X_train, train_y, sample_weight=sample_weight,
            eval_set=[(X_val, val_y)], verbose=False
        )
        preds = model.predict(X_test).flatten()
    except Exception as e:
        print(f"[ERRO] Falha no XGBoost: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # 4. Descomprimir Escala e Avaliar
    preds_real = np.clip(preds, 0, None) * market_max
    y_true_real = test_y * market_max

    metrics = evaluate(y_true_real, preds_real)
    
    # Criar DataFrames de Resultados
    metrics_df = pd.DataFrame([{
        "model": "xgboost", "arquivo": csv_file.name,
        "mae": metrics["MAE"], "rmse": metrics["RMSE"], "smape": metrics["sMAPE"]
    }])
    
    pred_df = pd.DataFrame({
        "model": "xgboost", "arquivo": csv_file.name,
        "product_id": le.inverse_transform(test_df['product_id_encoded'].astype(int)),
        "date": test_df['date'],
        "y_true": y_true_real, "y_pred": preds_real,
        "scenario": scenario
    })

    return metrics_df, pred_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir(): continue
        arquivos = list(market_path.glob('cat*.csv'))
        if not arquivos: continue

        market_name = market_path.name
        market_max = compute_market_max(market_path)
        print(f"\nRodando XGBoost em: {market_name} | max_global_quantity={market_max:.4f}")

        all_results, all_predictions = [], []

        for csv_file in arquivos:
            df_res, df_pred = process_file(csv_file, market_max=market_max, scenario=args.scenario, date_from=args.date_from, date_to=args.date_to)
            if not df_res.empty: all_results.append(df_res)
            if not df_pred.empty: all_predictions.append(df_pred)

        if all_results:
            final = pd.concat(all_results, ignore_index=True)
            suffix = f"_{args.scenario}" if args.scenario else ""
            out_file = OUTPUT_BASE / f"{market_name}_xgb{suffix}.csv"
            ensure_dir(out_file.parent)
            final.to_csv(out_file, index=False)
            
            if all_predictions:
                pred_file = OUTPUT_BASE / f"{market_name}_xgb{suffix}_predictions.csv"
                pd.concat(all_predictions, ignore_index=True).to_csv(pred_file, index=False)

            print(f"FINAL sMAPE: {final['smape'].mean():.4f}")
            print(f"MAE: {final['mae'].mean():.4f}")
            print(f"RMSE: {final['rmse'].mean():.4f}")
        else:
            market_smap = naive_market_smape(market_path)
            print(f"FINAL sMAPE: {market_smap:.4f}")

if __name__ == "__main__":
    main()