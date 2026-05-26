


# --- sys.path bootstrap: ensure project root is in sys.path before any src import ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- BLOQUEIO DE THREADS E GPU (antes de qualquer import pesado) ----
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# -----------------------------------------------------------

import gc
import logging
import traceback
import os
import argparse
import sys


# --- (Optional) Debug sys.path ---
print(f"[DEBUG][lstm_model] sys.path: {sys.path}")

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

try:
    from src.features.scenarios import apply_scenario
except ImportError:
    apply_scenario = None

from keras.models import Model
from keras.layers import Input, AdditiveAttention, LSTM, Dense, Dropout, Embedding, Flatten, Concatenate
from keras.callbacks import EarlyStopping

# utilitários
from src.utils.helpers import ensure_dir, normalize_columns, add_lag_features
from src.utils.reproducibility import set_global_seed
from src.models.evaluate import evaluate
# fallback metric across market when individual results missing
from src.utils.metrics import naive_market_smape
from src.config.model_params import COMMON_MODEL_PARAMS


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LSTM_PARAMS = COMMON_MODEL_PARAMS["training_by_model"]["lstm"]  
BATCH_SIZE = LSTM_PARAMS["batch_size"]
EPOCHS = LSTM_PARAMS["epochs"]
PATIENCE = LSTM_PARAMS.get("patience", None)

WINDOW = COMMON_MODEL_PARAMS["window_size"] 
HORIZON = COMMON_MODEL_PARAMS.get("prediction_horizon_days", 1)

TRAIN_RATIO = COMMON_MODEL_PARAMS["train_ratio"]
VAL_RATIO = COMMON_MODEL_PARAMS.get("val_ratio", 0.0)

INPUT_BASE = ROOT / "Dados" / "split"
OUTPUT_BASE = ensure_dir(ROOT / "Resultados" / "lstm")

TARGET = 'quantity'
FEATURES_BASE = COMMON_MODEL_PARAMS["features_base"]
LAGS = COMMON_MODEL_PARAMS["lags"]
ROLLING_WINDOWS = COMMON_MODEL_PARAMS["rolling_windows"]
QTY_FEATURES = [f"lag_{lag}" for lag in LAGS] + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]







def build_lstm_model(n_products, n_features, window):


    from keras.layers import Masking

    input_ts = Input(shape=(window, n_features))
    input_prod = Input(shape=(1,))

    # Adiciona camada Masking para ignorar valores sentinela (-99.0)
    x = Masking(mask_value=-99.0)(input_ts)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    x = AdditiveAttention()([x, x])
    x = LSTM(32)(x)

    emb = Embedding(input_dim=n_products, output_dim=16, name='prod_emb')(input_prod)
    emb = Flatten()(emb)

    x = Concatenate()([x, emb])
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='softplus')(x) # Softplus evita valores negativos
        # Adiciona-se camada Masking para ignorar valores sentinela (-99.0)

    model = Model([input_ts, input_prod], output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=COMMON_MODEL_PARAMS["loss_by_model"]["lstm"],
    )

    return model




def create_sequences(df, window, features, target, positive_weight=3.0):
    X, y, w = [], [], []
    
    # Itera por produto para garantir isolamento de históricos
    values_x = df[features].values
    values_y = df[target].values

    for i in range(len(df) - window):
        target_next = values_y[i + window]
        X.append(values_x[i:i+window])
        y.append(values_y[i+window])
        w.append(positive_weight if target_next > 0 else 1.0)         
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(w, dtype=np.float32)
        # Itera-se por produto para garantir isolamento de históricos

def _normalize_product_frame(g, market_max):
    g = g.copy()
    g[TARGET] = g[TARGET].clip(lower=0) / market_max
    if "unitvalue" in g.columns:
        g["unitvalue"] = g["unitvalue"].clip(lower=0)

    g = add_lag_features(g, target=TARGET, lags=LAGS, rolling_windows=ROLLING_WINDOWS)

    for c in QTY_FEATURES:
        if c in g.columns:
            g[c] = g[c] / market_max
    return g

def process_file_lstm(path, market_max, scenario=None, date_from=None, date_to=None):
    df = pd.read_csv(path, parse_dates=['date'])
    df = normalize_columns(df)
    
    # 1. Filtros Iniciais
    if date_from: df = df[df["date"] >= pd.to_datetime(date_from)]
    if date_to: df = df[df["date"] <= pd.to_datetime(date_to)]
    
    if scenario is not None and apply_scenario is not None:
        try: df = apply_scenario(df, scenario)
        except Exception: pass

    # 2. Preparação de IDs e Features
    le = LabelEncoder()
    df['product_idx'] = le.fit_transform(df['product_id'])
    
    CYCLIC_FEATURES = ["day_sin", "day_cos", "month_sin", "month_cos"]
    features = [c for c in FEATURES_BASE if c in df.columns] + \
               [c for c in QTY_FEATURES if c in df.columns] + \
               [c for c in CYCLIC_FEATURES if c in df.columns]

    # --- CONEXÃO DOS CENÁRIOS NA REDE NEURAL ---
    if scenario is not None:
        col_map = {"volume": "volume_cluster", "price": "price_cluster", "kmeans": "kmeans_cluster"}
        cluster_col = col_map.get(scenario)
        
        if cluster_col and cluster_col in df.columns:
            # 1. Converter labels de texto ('cheap', 'high') para números (0, 1, 2)
            df[cluster_col] = LabelEncoder().fit_transform(df[cluster_col].astype(str))
            
            # 2. INJETAR a coluna na lista de features que a rede neural vai ler
            if cluster_col not in features:
                features.append(cluster_col)
    
    # 3. Limpeza Mandatória (float32 e dropna de lags)
    feat_cols = features + [TARGET]
    df = df.dropna(subset=feat_cols)
    df = df[df[TARGET] != -99.0]
    df[features] = df[features].astype(np.float32)
    df[TARGET] = df[TARGET].astype(np.float32)

    # 4. Agrupamento Global de Sequências
    all_X_train, all_y_train, all_w_train, all_id_train = [], [], [], []
    all_X_val, all_y_val, all_id_val = [], [], []
    all_X_test, all_y_test, all_id_test, all_dates_test = [], [], [], []

    for product_idx, g in df.groupby("product_idx"):
        g = g.sort_values("date")
        # Normalização local (opcional, mas recomendada)
        g[TARGET] = g[TARGET].clip(lower=0) / market_max
        for c in QTY_FEATURES:
            if c in g.columns: g[c] = g[c] / market_max

        n = len(g)
        if n < WINDOW * 2: continue

        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        # Criar sequências
        xt, yt, wt = create_sequences(g.iloc[:train_end], WINDOW, features, TARGET)
        xv, yv, _ = create_sequences(g.iloc[train_end:val_end], WINDOW, features, TARGET)
        xe, ye, _ = create_sequences(g.iloc[val_end:], WINDOW, features, TARGET)

        if len(xt) == 0 or len(xe) == 0: continue

        # Acumular dados globais
        all_X_train.append(xt); all_y_train.append(yt); all_w_train.append(wt)
        all_id_train.append(np.full((len(xt), 1), product_idx))
        
        all_X_val.append(xv); all_y_val.append(yv)
        all_id_val.append(np.full((len(xv), 1), product_idx))
        
        all_X_test.append(xe); all_y_test.append(ye)
        all_id_test.append(np.full((len(xe), 1), product_idx))
        all_dates_test.append(g["date"].iloc[val_end+WINDOW:].values)

    if not all_X_train: return pd.DataFrame(), pd.DataFrame()

    # 5. Concatenação em Tensores Globais
    X_train = np.concatenate(all_X_train); y_train = np.concatenate(all_y_train)
    w_train = np.concatenate(all_w_train); id_train = np.concatenate(all_id_train)
    X_val = np.concatenate(all_X_val); y_val = np.concatenate(all_y_val); id_val = np.concatenate(all_id_val)
    X_test = np.concatenate(all_X_test); id_test = np.concatenate(all_id_test)

    # 6. Treinamento Único do Modelo
    n_products = int(df['product_idx'].max() + 1)
    model = build_lstm_model(n_products, len(features), WINDOW)
    
    print(f" -> [LSTM GLOBAL] Iniciando treino do mercado | {len(X_train)} sequências...")
    
    # --- LÓGICA DINÂMICA DE VALIDAÇÃO E PACIÊNCIA ---
    callbacks_list = []
    val_data = None
    
    # Só aciona o EarlyStopping SE a paciência existir E se houver dados de validação
    if PATIENCE is not None and len(X_val) > 0:
        early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        callbacks_list.append(early_stop)
        val_data = ([X_val, id_val], y_val)
    
    model.fit(
        x=[X_train, id_train], y=y_train, sample_weight=w_train,
        validation_data=val_data, # Fica None se não houver validação (o Keras aceita isso sem erro)
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        callbacks=callbacks_list, # Fica vazio se não houver paciência
        verbose=1
    )

    # 7. Previsão e Descompressão de Resultados
    preds = model.predict([X_test, id_test], verbose=0).flatten()
    preds_real = np.clip(preds, 0, None) * market_max
    y_test_real = np.concatenate(all_y_test) * market_max
    
    metrics = evaluate(y_test_real, preds_real)
    
    # Criar DataFrame de previsões (mapeando IDs de volta)
    pred_df = pd.DataFrame({
        "model": "lstm",
        "arquivo": path.name, # <-- CORREÇÃO AQUI
        "product_id": le.inverse_transform(id_test.flatten().astype(int)),
        "date": np.concatenate(all_dates_test),
        "y_true": y_test_real,
        "y_pred": preds_real,
        "scenario": scenario
    })

    metrics_df = pd.DataFrame([{
        "model": "lstm", 
        "arquivo": path.name,  # <-- CORREÇÃO AQUI
        "mae": metrics["MAE"], 
        "smape": metrics["sMAPE"], 
        "rmse": metrics["RMSE"]
    }])
    
    return metrics_df, pred_df


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


def main():
    global EPOCHS # Apenas LSTM e GRU precisam dessa linha. No XGBoost pode apagar.
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=None,
                        help="Cenário a ser aplicado (volume, price, kmeans)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, # Apenas LSTM e GRU precisam dessa linha. No XGB pode apagar.
                        help="Número de épocas de treinamento")
    parser.add_argument("--date-from", type=str, default=None,
                        help="Data inicial (YYYY-MM-DD) para filtrar histórico")
    parser.add_argument("--date-to", type=str, default=None,
                        help="Data final (YYYY-MM-DD) para filtrar histórico")

    args = parser.parse_args()
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Apenas LSTM e GRU precisam importar o tensorflow aqui na main. O XGBoost não precisa.
    import tensorflow as tf
    tf.random.set_seed(args.seed)

    # Sobrescrever EPOCHS se especificado (Apenas LSTM e GRU)
    EPOCHS = args.epochs

    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir():
            continue

        market_name = market_path.name
        market_max = compute_market_max(market_path)

        # TROQUE "MODELO" pelo nome correto: LSTM, GRU ou XGBoost
        print(f"\nRodando MODELO em: {market_name} | max_global_quantity={market_max:.4f}")

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

        for csv_file in arquivos:
            # TROQUE process_file_lstm para process_file se estiver no GRU ou XGBoost
            df_res, df_pred = process_file_lstm(
                csv_file, market_max=market_max, scenario=args.scenario,
                date_from=args.date_from, date_to=args.date_to
            )

            if not df_res.empty:
                all_results.append(df_res)
            if not df_pred.empty:
                all_predictions.append(df_pred)

        if all_results:
            final = pd.concat(all_results, ignore_index=True)

            # nome do arquivo inclui o cenário para evitar sobrescrita
            suffix = f"_{args.scenario}" if args.scenario else ""
            
            # TROQUE "lstm" para "gru" ou "xgb"
            out_file = OUTPUT_BASE / f"{market_name}_lstm{suffix}.csv"
            ensure_dir(out_file.parent)
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")

            if all_predictions:
                # TROQUE "lstm" para "gru" ou "xgb"
                pred_file = OUTPUT_BASE / f"{market_name}_lstm{suffix}_predictions.csv"
                pd.concat(all_predictions, ignore_index=True).to_csv(pred_file, index=False)
                print(f"  Curva real vs predito salva em {pred_file}")

            # IMPRIME AS MÉTRICAS PARA O ORQUESTRADOR LER (Comunicação limpa)
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