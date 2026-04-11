import numpy as np

def create_sequences(df, feature_cols, target_col, window):
    X, y = [], []

    # Usa groupby para eficiência e ordena por data
    for _, group in df.groupby("product_id"):
        group = group.sort_values("date")
        X_values = group[feature_cols].values
        y_values = group[target_col].values

        # Proteção contra séries menores que a janela
        if len(group) <= window:
            continue

        # Sliding window
        for i in range(len(group) - window):
            X.append(X_values[i : i + window])
            y.append(y_values[i + window])

    # Retorna arrays float32 para compatibilidade com Keras/TensorFlow
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')