import numpy as np
import pandas as pd
import tensorflow as tf
import gc
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping


# ========================= CONFIGURAÇÕES GLOBAIS =========================

WINDOW = 14      # Janela de observação 14 dias
BATCH_SIZE = 32  # Tamanho do lote para otimização de memória no i3
EPOCHS = 50      # Limite de iterações de treino
PATIENCE = 6     # Tolerância para intPerrupção antecipada 

TARGET = 'sales' # Variável alvo

# Atributos explicativos selecionados
FEATURES = [
    'price', 'on_promotion', 'dayofweek',
    'weekofyear', 'month', 'lag_1', 'lag_7'
]

# ========================= MODELO LSTM =========================

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

# ========================= JANELAMENTO COM CONTEXTO =========================

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
        
        X_vals = df_all_p[features].values
        y_vals = df_all_p[target].values
        
        # Calcula o índice de início para não perder dados do conjunto alvo
        idx_start = len(df_all_p) - len(df_tar_p) - window
        
        if idx_start < 0:
            continue
        
        # Desliza a janela preenchendo as sequências 3D
        for i in range(idx_start, idx_start + len(df_tar_p)):
            X.append(X_vals[i:i+window])
            y.append(y_vals[i+window])
            
    return np.array(X), np.array(y)

# ========================= PIPELINE PRINCIPAL =========================

def process_file_lstm(path):

    # Carregamento e tipagem de data
    df = pd.read_csv(path, parse_dates=['Date'])
    
    # Limpeza de registros sem alvo
    df = df.dropna(subset=[TARGET])

    # Separação baseada no split original do dataset
    train_df = df[df['split'] == 'train'].copy()
    val_df   = df[df['split'] == 'val'].copy()
    test_df  = df[df['split'] == 'test'].copy()

    # Cláusula de guarda para volume mínimo de treino
    if len(train_df) < 200:
        return None

    scaler = MinMaxScaler()

    # Normalização 
    train_df.loc[:, FEATURES] = scaler.fit_transform(train_df[FEATURES])
    val_df.loc[:, FEATURES]   = scaler.transform(val_df[FEATURES])
    test_df.loc[:, FEATURES]  = scaler.transform(test_df[FEATURES])

    # Geração de sequências de treino
    X_train, y_train = create_sequences_with_context(
        train_df, train_df, WINDOW, FEATURES, TARGET
    )

    # Concatenação para prover contexto  aos conjuntos de Validação e Teste
    val_all  = pd.concat([train_df, val_df])
    test_all = pd.concat([val_df, test_df])

    X_val, y_val = create_sequences_with_context(
        val_all, val_df, WINDOW, FEATURES, TARGET
    )

    X_test, y_test = create_sequences_with_context(
        test_all, test_df, WINDOW, FEATURES, TARGET
    )

    # Valida se o janelamento gerou dados de teste
    if len(X_test) == 0:
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

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    preds_inv = scaler.inverse_transform(preds.reshape(-1,1)).flatten()

    # Limpeza rigorosa de memória para hardware limitado
    tf.keras.backend.clear_session()
    gc.collect()

    return y_test_inv, preds_inv