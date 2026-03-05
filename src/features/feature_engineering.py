import numpy as np

import pandas as pd

def create_sequences_by_product(df, features, target, window , positive_weight=3.0):
    # Identifica dinamicamente a coluna de data para evitar erros de case sensitivity
    date_col = "Date" if "Date" in df.columns else "date"

    X_sequences = []
    y_sequences = []
    sample_weights = []

    # Isolamento por ID de produto: crucial para evitar que séries temporais distintas se misturem
    for _, group in df.groupby("product_id"):
        # Ordenação cronológica obrigatória para garantir que a janela reflita a sequência real
        group = group.sort_values(date_col)

        # Extração dos valores brutos para manipulação vetorial eficiente com Numpy
        X = group[features].values
        y = group[target].values

        # Proteção contra séries históricas insuficientes para o tamanho da janela definido
        if len(X) <= window:
            continue

        # Algoritmo de janela deslizante para gerar amostras sequenciais
        for i in range(len(X) - window):
            target_next = y[i + window]
            X_sequences.append(X[i:i+window])
            # y_seq recebe o valor do dia imediatamente posterior
            y_sequences.append(y[i+window])
            sample_weights.append(positive_weight if target_next > 0 else 1.0)
    # Conversão final para o formato de tensor 3D exigido por modelos LSTM e GRU
    return np.array(X_sequences), np.array(y_sequences), np.array(sample_weights)

def add_price_segments(df):
    # Segmentação por Preço (Quartis)
    df['price_segment'] = pd.qcut(df['unitvalue'], 4, labels=[1, 2, 3, 4])
    return df

def add_volume_segments(df):
    # Segmentação por Volume (Acima/Abaixo da Mediana)
    median_vol = df.groupby('product_id')['quantity'].transform('sum').median()
    df['volume_segment'] = df.groupby('product_id')['quantity'].transform('sum') >= median_vol
    return df

def build_features(
    df,
    target_col="quantity",
    date_col="Date",
    window=7,
    positive_weight=3.0
):
    features = [
        col for col in df.columns
        if col not in [target_col, date_col, "product_id"]
    ]

    X, y, w = create_sequences_by_product(
        df=df,
        features=features,
        target=target_col,
        window=window,
        positive_weight=positive_weight,
    )

    return X, y, w