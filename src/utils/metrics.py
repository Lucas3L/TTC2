import numpy as np

def mae(y_true, y_pred):
    # Converte para array e achata para garantir cálculo vetorial linear
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    # Retorna a média das diferenças absolutas
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    # Garante o formato de vetor para operações elemento a elemento
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    # Retorna a raiz da média dos erros quadrados
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def smape(y_true, y_pred):
    # Padronização de formato para evitar erros de dimensão
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    # Cálculo do denominador simétrico
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Cálculo da diferença relativa com proteção contra divisão por zero
    diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)

    # Retorna o erro percentual simétrico médio em escala 0-100
    return 100 * np.mean(diff)


def evaluate_all(y_true, y_pred):
    # Agrupa todas as métricas em um dicionário para exportação em tabelas ou CSV
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred)
    }