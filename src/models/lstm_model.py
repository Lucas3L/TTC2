from pathlib import Path
import sys
import gc
import os
import random


import numpy as np
import pandas as pd
import tensorflow as tf


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import argparse



file_path = Path(__file__).resolve()
root = file_path.parents[2]
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.models.evaluate import evaluate




WINDOW = 14      # Janela de observação 14 dias
BATCH_SIZE = 32  # Tamanho do lote para otimização de memória no i3
EPOCHS = 50      # Limite de iterações de treino
PATIENCE = 6     # Tolerância para intPerrupção antecipada 

BASE_DIR = root
INPUT_BASE = BASE_DIR / "Dados" / "preprocessed"
OUTPUT_BASE = BASE_DIR / "Resultados" / "lstm"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

TARGET = 'quantity'
FEATURES = [
     'onpromotion', 'unitvalue',
    'holiday', 'month', 'day_of_week', 'is_weekend'
]




def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)




def build_lstm_model(input_shape):
    # Arquitetura simplificada para evitar overfitting em séries curtas
    model = Sequential([
        LSTM(32, input_shape=input_shape),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam', # Otimizador adaptativo
        loss='mse'        # Erro Quadrático Médio como função de perda
    )
    
    return model




def create_sequences_with_context(df_all, df_target, window, features, target):
    X, y = [], []
    
    # Itera por produto para garantir isolamento de históricos
    for pid in df_target['product_id'].unique():
        
        # Filtra histórico completo e o conjunto alvo (val ou test)
        df_all_p = df_all[df_all['product_id'] == pid]
        df_tar_p = df_target[df_target['product_id'] == pid]
        
        # Valida se há dados suficientes para a primeira janela
        if len(df_all_p) < window + 1:
            continue
        
        X_vals = df_all_p[features].astype(float).values
        y_vals = df_all_p[target].astype(float).values
        
        
        # Calcula o índice de início para não perder dados do conjunto alvo
        idx_start = len(df_all_p) - len(df_tar_p) - window
        
        if idx_start < 0:
            continue
        
        # Desliza a janela preenchendo as sequências 3D
        for i in range(idx_start, idx_start + len(df_tar_p)):
            X.append(X_vals[i:i+window])
            y.append(y_vals[i+window])
            
    return np.array(X), np.array(y)




def create_sequences_simple(df, window, features, target):
    X, y = [], []
    values_x = df[features].astype(float).values
    values_y = df[target].astype(float).values

    for i in range(len(df) - window):
        X.append(values_x[i:i+window])
        y.append(values_y[i+window])

    return np.array(X), np.array(y)




def process_file_lstm(path):

    # Carregamento e tipagem de data
    df = pd.read_csv(path, parse_dates=['Date'])

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
    )

    # Limpeza de registros sem alvo
    df = df.dropna(subset=[TARGET])

    # Separação baseada no split original do dataset
    df = df.sort_values("date")

    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    # Cláusula de guarda para volume mínimo de treino
    if len(train_df) < 200:
        print(f"    Poucos dados treino ({len(train_df)}) em {path.name}")

        return None

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_df[FEATURES] = train_df[FEATURES].astype(float)
    val_df[FEATURES]   = val_df[FEATURES].astype(float)
    test_df[FEATURES]  = test_df[FEATURES].astype(float)
    
    train_df.loc[:, FEATURES] = scaler_x.fit_transform(train_df[FEATURES])
    train_df.loc[:, [TARGET]] = scaler_y.fit_transform(train_df[[TARGET]])

    val_df.loc[:, FEATURES] = scaler_x.transform(val_df[FEATURES])
    val_df.loc[:, [TARGET]] = scaler_y.transform(val_df[[TARGET]])

    test_df.loc[:, FEATURES] = scaler_x.transform(test_df[FEATURES])
    test_df.loc[:, [TARGET]] = scaler_y.transform(test_df[[TARGET]])
   
   
    # Geração de sequências de treino
    X_train, y_train = create_sequences_simple(
        train_df, WINDOW, FEATURES, TARGET
    )
    # Concatenação para prover contexto  aos conjuntos de Validação e Teste
    val_all  = pd.concat([train_df, val_df])
    test_all = pd.concat([val_df, test_df])

    X_val, y_val = create_sequences_with_context(
        pd.concat([train_df, val_df]),
        val_df, WINDOW, FEATURES, TARGET
    )

    X_test, y_test = create_sequences_with_context(
        pd.concat([val_df, test_df]),
        test_df, WINDOW, FEATURES, TARGET
    )

    # Valida se o janelamento gerou dados de teste
    if len(X_test) == 0:
        print(f"    X_test vazio em {path.name}")

        return None

    # Inicialização do modelo Keras
    model = build_lstm_model((WINDOW, len(FEATURES)))

    # Configuração de parada antecipada monitorando perda na validação
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,  # Dados de entrada e resposta real
        validation_data=(X_val, y_val), # Avaliação em dados não vistos para evitar overfitting
        epochs=EPOCHS,   # Ciclos totais de treinamento do modelo
        batch_size=BATCH_SIZE,  # Lotes de dados processados por vez 
        callbacks=[early_stop],  # Parada automática se o erro parar de cair
        verbose=0     # Desativa logs repetitivos no terminal
    )

    # Predição e retorno para escala original
    preds = model.predict(X_test).flatten()

    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    preds_inv = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

    # Limpeza rigorosa de memória para hardware limitado
    tf.keras.backend.clear_session()
    gc.collect()

    metrics = evaluate(y_test_inv, preds_inv)
    print(
        f"    FINAL -> MAE: {metrics['MAE']:.4f} | "
        f"RMSE: {metrics['RMSE']:.4f} | "
        f"sMAPE: {metrics['sMAPE']:.4f}"
    )
    return metrics




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_global_seed(args.seed)

    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir():
            continue

        market_name = market_path.name
        print(f"\nRodando LSTM em: {market_name}")

        all_results = []

        for csv_file in market_path.glob("cat*.csv"):
            metrics = process_file_lstm(csv_file)

            if metrics is None:
                print(f"  Arquivo ignorado: {csv_file.name}")

            else:
                metrics["arquivo"] = csv_file.name
                all_results.append(metrics)

        if all_results:
            final = pd.DataFrame(all_results)
            out_file = OUTPUT_BASE / f"{market_name}_lstm.csv"
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")

if __name__ == "__main__":
    main()

