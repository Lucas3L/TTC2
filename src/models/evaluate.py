import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def smape(y_true, y_pred):
    # Converte entradas para arrays e garante formato unidimensional para o cálculo
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # Cálculo da média das magnitudes para normalização simétrica do erro
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    # Calcula a diferença absoluta ponderada pelo denominador simétrico
    diff = np.abs(y_true - y_pred) / denominator

    # Cláusula de guarda: define erro como zero onde real e previsto são zero
    diff[denominator == 0] = 0.0

    # Retorna a média do erro percentual simétrico em escala de 0 a 100
    return np.mean(diff) * 100


def evaluate_model(y_true, y_pred):
    # Padronização de formato para garantir compatibilidade com saídas do GRU/LSTM/XGBoost
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # Erro Médio Absoluto: indica a magnitude média do erro em unidades reais 
    mae = mean_absolute_error(y_true, y_pred)
    
    # Raiz do Erro Quadrático Médio: evidencia se o modelo está cometendo erros muito grandes
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Execução da métrica sMAPE para avaliação da precisão percentual estável
    smape_val = smape(y_true, y_pred)

    # Consolidação das métricas em dicionário para geração automática de tabelas de resultados
    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape_val
    }