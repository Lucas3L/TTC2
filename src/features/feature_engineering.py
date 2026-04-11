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



def build_features(
    df,
    target_col="quantity",
    date_col="date",
    window=7,
    positive_weight=3.0
):
    # 1. Bloqueia colunas de metadados e observation
    forbidden = [target_col, date_col, "product_id", "observation", "category", "market"]
    features = [col for col in df.columns if col not in forbidden]

    # 2. Garante que as features são numéricas
    df_numeric = df.copy()
    for col in features:
        if df_numeric[col].dtype == 'object':
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(0)

    # 3. (Sugestão) Normalização: Aplique MinMaxScaler nas features numéricas antes do build_features para melhor performance do LSTM.

    # 4. Geração das sequências
    X, y, w = create_sequences_by_product(
        df=df_numeric,
        features=features,
        target=target_col,
        window=window,
        positive_weight=positive_weight,
    )
    return X, y, w