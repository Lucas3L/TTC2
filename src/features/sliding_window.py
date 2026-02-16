import numpy as np

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

    # Conversão final para o formato de tensor 3D exigido por modelos LSTM e GRU
    return np.array(X_sequences), np.array(y_sequences)