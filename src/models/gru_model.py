# --- sys.path bootstrap: ensure project root is in sys.path before any src import ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- O TENSORFLOW PRECISA SER O PRIMEIRO A ENTRAR NA MEMÓRIA ---
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, GRU, Dense, Dropout, Embedding, Flatten, Concatenate
from keras.callbacks import EarlyStopping
# ---------------------------------------------------------------

import gc
import logging
import traceback
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Imports locais do projeto
from src.utils.helpers import ensure_dir, normalize_columns, add_lag_features
from src.models.evaluate import evaluate
from src.utils.metrics import naive_market_smape
from src.config.model_params import COMMON_MODEL_PARAMS
try:
    from src.features.scenarios import apply_scenario
except ImportError:
    apply_scenario = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

WINDOW = COMMON_MODEL_PARAMS["window_size"]    
BATCH_SIZE = COMMON_MODEL_PARAMS["training_by_model"]["gru"]["batch_size"]
EPOCHS = COMMON_MODEL_PARAMS["training_by_model"]["gru"]["epochs"]
PATIENCE = COMMON_MODEL_PARAMS["training_by_model"]["gru"]["patience"]
TRAIN_RATIO = COMMON_MODEL_PARAMS["train_ratio"]
VAL_RATIO = COMMON_MODEL_PARAMS["val_ratio"]

FEATURES_BASE = COMMON_MODEL_PARAMS["features_base"]
LAGS = COMMON_MODEL_PARAMS["lags"]
ROLLING_WINDOWS = COMMON_MODEL_PARAMS["rolling_windows"]
QTY_FEATURES = [f"lag_{lag}" for lag in LAGS] + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]

INPUT_BASE = ROOT / "Dados" / "preprocessed"
OUTPUT_BASE = ensure_dir(ROOT / "Resultados" / "gru")
TARGET = 'quantity'


def build_gru_model(n_products, n_features, window):
    input_ts = Input(shape=(window, n_features))
    input_prod = Input(shape=(1,))

    # Sem Masking! Os dados já chegam limpos pelo dropna absoluto.
    x = GRU(64, return_sequences=True)(input_ts)
    x = Dropout(0.2)(x)
    x = GRU(32)(x)

    # Embedding para a rede neural diferenciar produtos/clusters
    emb = Embedding(input_dim=n_products, output_dim=16)(input_prod)
    emb = Flatten()(emb)

    x = Concatenate()([x, emb])
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='softplus')(x)

    model = Model([input_ts, input_prod], output)
    model.compile(optimizer='adam', loss=COMMON_MODEL_PARAMS["loss_by_model"]["gru"])
    return model


def run_gru(X_train, y_train, id_train, X_val, y_val, id_val, X_test, y_test, id_test):
    # n_products precisa ser seguro contra IDs ausentes nos splits
    n_products = int(max(id_train.max(), id_val.max(), id_test.max()) + 1)
    model = build_gru_model(n_products, X_train.shape[-1], WINDOW)

    def has_nan_inf(arr, name):
        if np.any(np.isnan(arr)): return True
        if np.any(np.isinf(arr)): return True
        return False

    if has_nan_inf(X_train, 'X_train') or has_nan_inf(y_train, 'y_train') or has_nan_inf(X_val, 'X_val') or has_nan_inf(y_val, 'y_val') or has_nan_inf(X_test, 'X_test'):
        print(f"[ERRO][GRU] Dados inválidos detectados, abortando treino deste bloco.")
        return np.array([])

    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    try:
        print(f" -> [GRU GLOBAL] Iniciando treino | {len(X_train)} sequências válidas...")
        model.fit(
            [X_train, id_train], y_train,
            validation_data=([X_val, id_val], y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1 # Mostrar progresso no terminal
        )
        preds = model.predict([X_test, id_test], verbose=0).flatten()
    except Exception as e:
        print(f"[ERRO][GRU] Falha no fit/predict: {e}")
        logging.error(f"[ERRO][GRU] Falha no fit/predict: {e}\n{traceback.format_exc()}")
        tf.keras.backend.clear_session()
        gc.collect()
        return np.array([])
    return preds


def create_sequences(data, features, target, window):
    X, y, p, d = [], [], [], []
    for prod_id, g in data.groupby('product_id'):
        values_x = g[features].values
        values_y = g[target].values
        for i in range(len(g) - window):
            X.append(values_x[i:i+window])
            y.append(values_y[i+window])
            p.append(prod_id)
            d.append(g["date"].iloc[i+window])
            
    # Forçar Float32 para salvar memória RAM no Windows
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(p, dtype=np.int32), np.array(d)


def process_file(csv_file, scenario=None, date_from=None, date_to=None):
    df = pd.read_csv(csv_file, parse_dates=['date'])
    df = normalize_columns(df)
    
    if date_from: df = df[df["date"] >= pd.to_datetime(date_from)]
    if date_to: df = df[df["date"] <= pd.to_datetime(date_to)]

    if scenario is not None and apply_scenario is not None:
        try: df = apply_scenario(df, scenario)
        except Exception as e: print(f"Aviso: falha ao aplicar cenário {scenario}")

    le = LabelEncoder()
    df['product_id'] = le.fit_transform(df['product_id'])

    df = add_lag_features(df, TARGET, lags=LAGS, rolling_windows=ROLLING_WINDOWS)

    CYCLIC_FEATURES = ["day_sin", "day_cos", "month_sin", "month_cos"]
    features = [c for c in FEATURES_BASE if c in df.columns] + \
               [c for c in QTY_FEATURES if c in df.columns] + \
               [c for c in CYCLIC_FEATURES if c in df.columns]

    # --- CONEXÃO DOS CENÁRIOS NA REDE NEURAL ---
    if scenario is not None:
        col_map = {"volume": "volume_cluster", "price": "price_cluster", "kmeans": "kmeans_cluster"}
        cluster_col = col_map.get(scenario)
        if cluster_col and cluster_col in df.columns:
            df[cluster_col] = LabelEncoder().fit_transform(df[cluster_col].astype(str))
            if cluster_col not in features:
                features.append(cluster_col)
    # -------------------------------------------

    feat_cols = features + [TARGET]
    df = df.dropna(subset=feat_cols)
    df = df[df[TARGET] != -99.0]
    
    df[features] = df[features].astype(np.float32)
    df[TARGET] = df[TARGET].astype(np.float32)

    df = df.sort_values(['product_id', 'date'])

    train_list, val_list, test_list = [], [], []
    for pid, g in df.groupby('product_id'):
        n = len(g)
        if n < WINDOW * 2: continue # Ignora os picotados

        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_list.append(g.iloc[:train_end].copy())
        val_list.append(g.iloc[train_end:val_end].copy())
        test_list.append(g.iloc[val_end:].copy())

    if not train_list or not test_list: return pd.DataFrame(), pd.DataFrame()

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    if len(train_df) < WINDOW * 2 or len(test_df) < WINDOW:
        return pd.DataFrame(), pd.DataFrame()

    scaler_x = MinMaxScaler()
    train_df[features] = scaler_x.fit_transform(train_df[features])
    val_df[features] = scaler_x.transform(val_df[features])
    test_df[features] = scaler_x.transform(test_df[features])
    
    X_train, y_train, id_train, _ = create_sequences(train_df, features, TARGET, WINDOW)
    X_val, y_val, id_val, _ = create_sequences(val_df, features, TARGET, WINDOW)
    X_test, y_test, id_test, d_test = create_sequences(test_df, features, TARGET, WINDOW)

    if len(X_train) < 10 or len(X_test) < 5:
        return pd.DataFrame(), pd.DataFrame()

    preds = run_gru(X_train, y_train, id_train, X_val, y_val, id_val, X_test, y_test, id_test)
    if len(preds) == 0: return pd.DataFrame(), pd.DataFrame()

    for subset in (train_df, val_df, test_df):
        subset[TARGET] = subset[TARGET].clip(lower=0).astype(float)

    y_test_inv = y_test
    preds_inv = np.clip(preds, 0, None)

    metrics = evaluate(y_test_inv, preds_inv)

    metrics_df = pd.DataFrame([{
        "model": "gru", "arquivo": csv_file.name,
        "mae": metrics["MAE"], "rmse": metrics["RMSE"], "smape": metrics["sMAPE"]
    }])
    pred_df = pd.DataFrame({
        "model": "gru", "arquivo": csv_file.name,
        "product_id": le.inverse_transform(id_test.astype(int)),
        "date": d_test, "y_true": y_test_inv.astype(float),
        "y_pred": preds_inv.astype(float), "scenario": scenario
    })
    return metrics_df, pred_df


def main():
    global EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    EPOCHS = args.epochs

    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir(): continue
        market_name = market_path.name
        print(f"\nRodando GRU em: {market_name}")

        all_results, all_predictions = [], []
        for csv_file in market_path.glob("cat*.csv"):
            df_res, df_pred = process_file(csv_file, scenario=args.scenario, date_from=args.date_from, date_to=args.date_to)
            if not df_res.empty: all_results.append(df_res)
            if not df_pred.empty: all_predictions.append(df_pred)

        if all_results:
            final = pd.concat(all_results, ignore_index=True)
            suffix = f"_{args.scenario}" if args.scenario else ""
            out_file = OUTPUT_BASE / f"{market_name}_gru{suffix}.csv"
            
            try:
                ensure_dir(out_file.parent)
                final.to_csv(out_file, index=False)
                print(f"  Resultados salvos em {out_file}")
            except Exception as e:
                logging.error(f"[ERRO] Falha ao salvar {out_file}: {e}\n{traceback.format_exc()}")

            if all_predictions:
                pred_file = OUTPUT_BASE / f"{market_name}_gru{suffix}_predictions.csv"
                try:
                    ensure_dir(pred_file.parent)
                    pd.concat(all_predictions, ignore_index=True).to_csv(pred_file, index=False)
                except Exception as e:
                    pass

            mean_smap = final['smape'].mean()
            print(f"FINAL sMAPE: {final['smape'].mean():.4f}")
            print(f"MAE: {final['mae'].mean():.4f}")
            print(f"RMSE: {final['rmse'].mean():.4f}")
        else:
            market_smap = naive_market_smape(market_path)
            print(f"FINAL sMAPE: {market_smap:.4f}")

if __name__ == "__main__":
    main()