import gc
import os
import argparse

# Configurar path ANTES de importar módulos locais
from src.utils.project_paths import add_project_root_to_sys_path
root = add_project_root_to_sys_path(__file__)

import numpy as np
import pandas as pd
import tensorflow as tf

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

WINDOW = COMMON_MODEL_PARAMS["window_size"]      # Janela de observação
BATCH_SIZE = COMMON_MODEL_PARAMS["training_by_model"]["lstm"]["batch_size"]
EPOCHS = COMMON_MODEL_PARAMS["training_by_model"]["lstm"]["epochs"]
PATIENCE = COMMON_MODEL_PARAMS["training_by_model"]["lstm"]["patience"]
TRAIN_RATIO = COMMON_MODEL_PARAMS["train_ratio"]
VAL_RATIO = COMMON_MODEL_PARAMS["val_ratio"]

INPUT_BASE = root / "Dados" / "preprocessed"
OUTPUT_BASE = ensure_dir(root / "Resultados" / "lstm")

TARGET = 'quantity'
FEATURES_BASE = COMMON_MODEL_PARAMS["features_base"]
LAGS = COMMON_MODEL_PARAMS["lags"]
ROLLING_WINDOWS = COMMON_MODEL_PARAMS["rolling_windows"]
QTY_FEATURES = [f"lag_{lag}" for lag in LAGS] + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]







def build_lstm_model(n_products, n_features, window):

    input_ts = Input(shape=(window, n_features))
    input_prod = Input(shape=(1,))

    x = LSTM(64, return_sequences=True)(input_ts)
    x = Dropout(0.2)(x)

    x = AdditiveAttention()([x,x])
    x = LSTM(32)(x)

    emb = Embedding(input_dim=n_products, output_dim=16, name='prod_emb')(input_prod)
    emb = Flatten()(emb)    

    x = Concatenate()([x, emb])
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='softplus')(x) # Softplus evita valores negativos
    
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
    
    return np.array(X), np.array(y), np.array(w)

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

    # Carregamento e tipagem de data
    df = pd.read_csv(path, parse_dates=['date'])
    df = normalize_columns(df)
    df = df.dropna(subset=[TARGET])
    if date_from is not None:
        df = df[df["date"] >= pd.to_datetime(date_from)]
    if date_to is not None:
        df = df[df["date"] <= pd.to_datetime(date_to)]

    # aplicações de cenários (volume, price, kmeans)
    if scenario is not None:
        try:
            from src.features.scenarios import apply_scenario
            df = apply_scenario(df, scenario)
        except ImportError:
            # caso a importação falhe, ignoramos, mas avisamos
            print(f"Aviso: não foi possível aplicar cenário {scenario}")
    
    product_map = {pid: i for i, pid in enumerate(sorted(df["product_id"].unique()))}
    inv_product_map = {i: pid for pid, i in product_map.items()}
    df["product_idx"] = df["product_id"].map(product_map)

    df = df.sort_values(["product_idx", "date"])

    results = []
    predictions_rows = []

    for product_idx, g in df.groupby("product_idx"):

        g = g.copy().sort_values("date")
        g = _normalize_product_frame(g, market_max)
       
        features = [c for c in FEATURES_BASE if c in g.columns] + \
                [c for c in QTY_FEATURES if c in g.columns]

        keep_cols = features + [TARGET]
        g = g.dropna(subset=keep_cols)

        if len(g) < WINDOW * 6:
            continue

        n = len(g)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_df = g.iloc[:train_end].copy()
        val_df = g.iloc[train_end:val_end].copy()
        test_df = g.iloc[val_end:].copy()

        # Cláusula de guarda para volume mínimo de treino
        if len(train_df) < WINDOW * 5 or len(test_df) < WINDOW:
            continue
        X_train, y_train, w_train = create_sequences(train_df, WINDOW, features, TARGET, positive_weight=3.0)
        X_val, y_val, _ = create_sequences(val_df, WINDOW, features, TARGET, positive_weight=3.0)
        X_test, y_test, _ = create_sequences(test_df, WINDOW, features, TARGET, positive_weight=3.0)    

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            continue

        train_id = np.full((len(X_train), 1), product_idx)
        val_id   = np.full((len(X_val), 1), product_idx)
        test_id  = np.full((len(X_test), 1), product_idx)

        model = build_lstm_model(len(product_map) + 5, len(features),  WINDOW)

        # Configuração de parada antecipada monitorando perda na validação
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True
        )

        model.fit(
            x=[X_train, train_id],
            y=y_train,
            sample_weight=w_train,
            validation_data=([X_val, val_id], y_val), # Avaliação em dados não vistos para evitar overfitting
            epochs=EPOCHS,   # Ciclos totais de treinamento do modelo
            batch_size=BATCH_SIZE,  # Lotes de dados processados por vez 
            callbacks=[early_stop],  # Parada automática se o erro parar de cair
            verbose=0     # Desativa logs repetitivos no terminal
        )
        

        # Predição e retorno para escala original
        preds = model.predict([X_test, test_id], verbose=0).flatten()

        y_real = y_test * market_max
        p_real =  np.clip(preds, 0, None) * market_max
        
        metrics = evaluate(y_real, p_real)
        test_dates = test_df["date"].iloc[WINDOW:].reset_index(drop=True)

        results.append({
            "model": "lstm",
            "arquivo": path.name,
            "product_id": inv_product_map.get(product_idx, product_idx),
            "mae": metrics["MAE"],
            "rmse": metrics["RMSE"],
            "smape": metrics["sMAPE"]
        })

        for dt, y_t, y_p in zip(test_dates, y_real, p_real):
            predictions_rows.append({
                "model": "lstm",
                "arquivo": path.name,
                "product_id": inv_product_map.get(product_idx, product_idx),
                "date": dt,
                "y_true": float(y_t),
                "y_pred": float(y_p),
                "scenario": scenario
            })

        # Limpeza rigorosa de memória para hardware limitado
        tf.keras.backend.clear_session()
        gc.collect()


    return pd.DataFrame(results), pd.DataFrame(predictions_rows)



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
    global EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=None,
                        help="Cenário a ser aplicado (volume, price, kmeans)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Número de épocas de treinamento")
    parser.add_argument("--date-from", type=str, default=None,
                        help="Data inicial (YYYY-MM-DD) para filtrar histórico")
    parser.add_argument("--date-to", type=str, default=None,
                        help="Data final (YYYY-MM-DD) para filtrar histórico")
    args = parser.parse_args()
    set_global_seed(args.seed)

    # Sobrescrever EPOCHS se especificado
    EPOCHS = args.epochs

    # itera apenas sobre o cenário solicitado; o loop de cenários foi movido para main.py
    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir():
            continue

        market_name = market_path.name
        market_max = compute_market_max(market_path)

        print(f"\nRodando LSTM em: {market_name} | max_global_quantity={market_max:.4f}")

        all_results = []
        all_predictions = []

        for csv_file in market_path.glob("cat*.csv"):
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
            out_file = OUTPUT_BASE / f"{market_name}_lstm{suffix}.csv"
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")

            if all_predictions:
                pred_file = OUTPUT_BASE / f"{market_name}_lstm{suffix}_predictions.csv"
                pd.concat(all_predictions, ignore_index=True).to_csv(pred_file, index=False)
                print(f"  Curva real vs predito salva em {pred_file}")

            # imprime métrica final para que o orquestrador capture
            mean_smap = final['smape'].mean()
            print(f"FINAL sMAPE: {final['smape'].mean():.4f}")
        else:
            # calcula métrica de mercado como fallback
            market_smap = naive_market_smape(market_path)
            print(f"FALLBACK market sMAPE: {market_smap:.4f}")
            print(f"FINAL sMAPE: {market_smap:.4f}")

if __name__ == "__main__":
    main()
