from pathlib import Path
import sys
import gc
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Model
from keras.layers import Input, AdditiveAttention, LSTM, Dense, Dropout, Embedding, Flatten, Concatenate
from keras.callbacks import EarlyStopping
import argparse

# utilitários
from src.utils.helpers import ensure_dir, normalize_columns, add_lag_features
from src.utils.reproducibility import set_global_seed
from src.models.evaluate import evaluate


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silencia avisos inúteis do TF
file_path = Path(__file__).resolve()
root = file_path.parents[2]
if str(root) not in sys.path:
    sys.path.append(str(root))


WINDOW = 7      # Janela de observação 14 dias
BATCH_SIZE = 128  # Tamanho do lote para otimização de memória no i3
EPOCHS = 50      # Limite de iterações de treino
PATIENCE = 10     # Tolerância para intPerrupção antecipada 

INPUT_BASE = root / "Dados" / "preprocessed"
OUTPUT_BASE = ensure_dir(root / "Resultados" / "lstm")

TARGET = 'quantity'
FEATURES_BASE = [
     'onpromotion', 'unitvalue','holiday', 
     'month', 'day_of_week', 'is_weekend'
]







def build_lstm_model(n_products, n_features, window):

    input_ts = Input(shape=(window, n_features))
    input_prod = Input(shape=(1,))

    x = LSTM(64, return_sequences=True)(input_ts)
    x = Dropout(0.2)(x)

    att = AdditiveAttention()([x,x])

    x = LSTM(32)(att)
    emb = Embedding(input_dim=n_products, output_dim=16, name='prod_emb')(input_prod)
    emb = Flatten()(emb)    

    x = Concatenate()([x, emb])
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='linear')(x) # Softplus evita valores negativos
    
    model = Model([input_ts, input_prod], output)
    
    
    model.compile(optimizer = tf.keras.optimizers.Adam(0.001), loss='poisson')

    return model




def create_sequences(df, window, features, target):
    X, y = [], []
    
    # Itera por produto para garantir isolamento de históricos
    
    X, y = [], []
    values_x = df[features].values
    values_y = df[target].values

    for i in range(len(df) - window):
        X.append(values_x[i:i+window])
        y.append(values_y[i+window])
            
    return np.array(X), np.array(y)



def process_file_lstm(path, scenario=None):

    # Carregamento e tipagem de data
    df = pd.read_csv(path, parse_dates=['Date'])
    df = normalize_columns(df)
    df = df.dropna(subset=[TARGET])
    df = df[df['Date'].dt.year == 2019]

    # aplicações de cenários (volume, price, kmeans)
    if scenario is not None:
        try:
            from src.features.scenarios import apply_scenario
            df = apply_scenario(df, scenario)
        except ImportError:
            # caso a importação falhe, ignoramos, mas avisamos
            print(f"Aviso: não foi possível aplicar cenário {scenario}")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
    )

    le = LabelEncoder()
    df['product_id'] = le.fit_transform(df['product_id'])

    df = df.sort_values(["product_id", "date"])

    results = []

    n_unique_prods = df['product_id'].nunique()

    for product_id, g in df.groupby("product_id"):

        g = g.copy()

        if len(g) < WINDOW * 6:
            continue
        
        g[TARGET] = np.log1p(g[TARGET].clip(lower=0))
        g['unitvalue'] = np.log1p(g['unitvalue'].clip(lower=0))
        
        g = add_lag_features(g)
        g = g.dropna()
        features = FEATURES_BASE + ['lag_1','lag_7','rolling_mean_3','rolling_mean_7','rolling_mean_14']

        n = len(g)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        train_df = g.iloc[:train_end].copy()
        val_df   = g.iloc[train_end:val_end].copy()
        test_df  = g.iloc[val_end:].copy()

        # Cláusula de guarda para volume mínimo de treino
        if len(train_df) < WINDOW * 5 or len(test_df) < WINDOW:
            continue

        scaler_x = MinMaxScaler()

        train_df.loc[:, features] = scaler_x.fit_transform(train_df[features])

        val_df.loc[:, features] = scaler_x.transform(val_df[features])

        test_df.loc[:, features] = scaler_x.transform(test_df[features])
    
        X_train, y_train = create_sequences(
            train_df, WINDOW, features, TARGET
        )

        X_val, y_val = create_sequences(
            val_df, WINDOW, features, TARGET
        )

        X_test, y_test = create_sequences(
            test_df, WINDOW, features, TARGET
        )

        train_id = np.full((len(X_train), 1), product_id)
        val_id   = np.full((len(X_val), 1), product_id)
        test_id  = np.full((len(X_test), 1), product_id)

        model = build_lstm_model(n_unique_prods + 50, len(features),  WINDOW)

        # Configuração de parada antecipada monitorando perda na validação
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True
        )

        model.fit(
            x=[X_train, train_id],
            y=y_train,
            validation_data=([X_val, val_id], y_val), # Avaliação em dados não vistos para evitar overfitting
            epochs=EPOCHS,   # Ciclos totais de treinamento do modelo
            batch_size=BATCH_SIZE,  # Lotes de dados processados por vez 
            callbacks=[early_stop],  # Parada automática se o erro parar de cair
            verbose=0     # Desativa logs repetitivos no terminal
        )
        

        # Predição e retorno para escala original
        preds = model.predict([X_test, test_id]).flatten()

        y_real = np.expm1(y_test)
        p_real = np.expm1(preds)
        
        metrics = evaluate(y_real, p_real)

        results.append({
            "model": "lstm",
            "arquivo": path.name,
            "product_id": product_id,
            "mae": metrics["MAE"],
            "rmse": metrics["RMSE"],
            "smape": metrics["sMAPE"]
        })

        # Limpeza rigorosa de memória para hardware limitado
        tf.keras.backend.clear_session()
        gc.collect()


    return pd.DataFrame(results)




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=None,
                        help="Cenário a ser aplicado (volume, price, kmeans)")
    args = parser.parse_args()
    set_global_seed(args.seed)

    # itera apenas sobre o cenário solicitado; o loop de cenários foi movido para main.py
    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir():
            continue

        market_name = market_path.name
        print(f"\nRodando LSTM em: {market_name}")

        all_results = []

        for csv_file in market_path.glob("cat*.csv"):
            df_res = process_file_lstm(csv_file, scenario=args.scenario)

            if not df_res.empty:
                all_results.append(df_res)

        if all_results:
            final = pd.concat(all_results, ignore_index=True)

            # nome do arquivo inclui o cenário para evitar sobrescrita
            suffix = f"_{args.scenario}" if args.scenario else ""
            out_file = OUTPUT_BASE / f"{market_name}_lstm{suffix}.csv"
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")

            # imprime métrica final para que o orquestrador capture
            mean_smap = final['smape'].mean()
            print(f"FINAL sMAPE: {mean_smap:.4f}")

if __name__ == "__main__":
    main()

