import numpy as np
from scipy.stats import t

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=2000, ci=95, seed=42):
    """
    Calcula intervalo de confiança via bootstrap para validar a estabilidade do modelo.
    """
    # Inicializa o gerador de números aleatórios com semente fixa para reprodutibilidade
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    stats = []
    n = len(y_true)

    # Reamostragem com reposição para gerar a distribuição da métrica
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        stat = metric_fn(y_true[idx], y_pred[idx])
        stats.append(stat)

    stats = np.array(stats)

    # Cálculo dos percentis para definir os limites do intervalo de confiança
    lower = np.percentile(stats, (100-ci)/2)
    upper = np.percentile(stats, 100 - (100-ci)/2)

    return np.mean(stats), lower, upper


def diebold_mariano_test(e1, e2, h=1):
    """
    Teste Diebold-Mariano para comparar se a diferença de erro entre dois modelos é significante.
    """
    # d representa a diferença de erro entre o Modelo 1 e o Modelo 2
    d = e1 - e2
    T = len(d)

    mean_d = np.mean(d)

    # Cálculo da variância e autocovariância para lidar com a dependência temporal dos erros
    gamma0 = np.var(d, ddof=1)
    # Ajuste para o horizonte de previsão para considerar autocorrelação
    dm_stat = mean_d / np.sqrt((gamma0 + 2 * np.sum([np.cov(d[:-k], d[k:])[0,1] for k in range(1, h)])) / T)

    # Cálculo do p-value através da distribuição t de Student
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=T-1))

    return dm_stat, p_value