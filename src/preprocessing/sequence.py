import numpy as np

def create_sequences(df, feature_cols, target_col, window):
    # Inicializa listas para armazenar os conjuntos de entrada x e alvo y
    X, y = [], []

    # Itera sobre cada produto individualmente para evitar vazamento de dados entre IDs
    for product_id in df["product_id"].unique():
        # Filtra o subconjunto de dados pertencente a um único produto
        sub = df[df["product_id"] == product_id]

        # Converte as colunas de atributos e alvo em matrizes Numpy para processamento rápido
        X_values = sub[feature_cols].values
        y_values = sub[target_col].values

        # Desliza a janela temporal sobre o historico do produto atual
        for i in range(len(sub) - window):
            # Captura o bloco de dias anteriores  como entrada
            X.append(X_values[i:i+window])
            # Captura o valor do dia seguinte como o objetivo da previsão
            y.append(y_values[i+window])

    # Converte as listas finais em arrays Numpy 3D para o Keras
    return np.array(X), np.array(y)