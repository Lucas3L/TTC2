import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

def create_sequences_by_product(df, features, target, window):
    # Identifica dinamicamente a coluna de data para evitar erros de case sensitivity
    date_col = "Date" if "Date" in df.columns else "date"

    X_sequences = []
    y_sequences = []

    # Isolamento por ID de produto: crucial para evitar que séries temporais distintas se misturem
    for product_id, group in df.groupby("product_id"):
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
            # X_seq recebe o bloco de dias anteriores
            X_sequences.append(X[i:i+window])
            # y_seq recebe o valor do dia imediatamente posterior
            y_sequences.append(y[i+window])

    # Conversão final para o formato de tensor 3D exigido por modelos LSTM e GRU
    return np.array(X_sequences), np.array(y_sequences)

def build_features(
    df,
    target_col="Quantity",
    date_col="Date",
    window=14
):
    features = [
        col for col in df.columns
        if col not in [target_col, date_col, "product_id"]
    ]

    X, y = create_sequences_by_product(
        df=df,
        features=features,
        target=target_col,
        window=window
    )

    return X, y