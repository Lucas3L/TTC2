import os
import random
import numpy as np
import tensorflow as tf

def set_global_seed(seed: int = 42):
    """
    Garante reprodutibilidade total do pipeline.
    """
    # Fixa o hash do Python para garantir ordem consistente em dicionários e sets   
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Força o TensorFlow a usar algoritmos determinísticos 
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    # Fixa as sementes para Python nativo, Numpy e TensorFlow
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

from pathlib import Path
import pandas as pd
from datetime import datetime
import uuid

# Define o caminho central para o rastreamento histórico de experimentos
LOG_PATH = Path("Resultados/experiments_log.csv")

def log_experiment(
    description: str,
    models: list,
    features: list,
    window: int,
    metrics_df: pd.DataFrame,
    random_seed: int,
    replica_of: str = None
):

    # Gera um ID único e captura o momento da execução para rastreabilidade
    experiment_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Estrutura os dados integrando parâmetros de entrada e resultados de saída
    row = {
        "experiment_id": experiment_id,
        "datetime": timestamp,
        "description": description,
        "models": ",".join(models),
        "features": ",".join(features),
        "window": window,
        "random_seed": random_seed, # Registra a semente usada na rodada
        "metrics_set": "MAE,RMSE,sMAPE",
        "avg_mae": metrics_df["mae"].mean(),
        "avg_rmse": metrics_df["rmse"].mean(),
        "avg_smape": metrics_df["smape"].mean(),
        "replica_of": replica_of
    }

    df_row = pd.DataFrame([row])

    # Persistência incremental, anexa ao log existente ou cria um novo arquivo
    if LOG_PATH.exists():
        df_row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_PATH, index=False)

    return experiment_id
