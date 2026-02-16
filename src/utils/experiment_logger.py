import pandas as pd
from datetime import datetime

class ExperimentLogger:

    def __init__(self, out_path):
        # Define o caminho do arquivo de saída e inicializa a lista de registros
        self.out_path = out_path
        self.records = []

    def log(self, model, market, product, metrics, seed):
        # Cria um dicionário unificando identificadores, parâmetros e métricas
        record = {
            "timestamp": datetime.now(), # Registro exato do momento da execução
            "model": model,              # Nome do algoritmo
            "market": market,             # Identificação da unidade de negócio/mercado
            "product_id": product,        # ID do produto para análise granular
            "random_seed": seed,          # Semente para garantir reprodutibilidade futura
            **metrics                     # Desempacota as métricas
        }
        self.records.append(record) # Adiciona o experimento à memória temporária

    def save(self):
        # Converte a lista de dicionários acumulada em um DataFrame do Pandas
        df = pd.DataFrame(self.records)
        # Persiste todos os resultados em um único CSV para análise estatística
        df.to_csv(self.out_path, index=False)