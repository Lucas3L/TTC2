from pathlib import Path
import pandas as pd
from validators import (corrigir_datas_temporais,
                        corrigir_valores_temporais,
                        tratar_outliers_iqr_por_produto,)
import numpy as np

INPUT_BASE = Path("Dados/processed")
OUTPUT_BASE = Path("Dados/preprocessed")

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

for market_path in INPUT_BASE.iterdir():

    if not market_path.is_dir():
        continue

    market_name = market_path.name
    print(f"\nProcessando mercado: {market_name}")

    # Organiza os caminhos de saida e cria um diretoria caso não exista
    market_output = OUTPUT_BASE / market_name
    market_output.mkdir(parents=True, exist_ok=True)

    # Lê os arquivos csv que iniciam com cat e imprime o nome do arquivo
    for csv_file in market_path.glob("cat*.csv"):
        print(f"  Lendo {csv_file.name}")

        # Leitura do arquivo e conversão da coluna data
        df = pd.read_csv(csv_file, parse_dates=["date"])  # espera snake_case
        # Detalha e salva as anomalias encontradas
        df["observation"] = "ok"
        anomalias = []

        # 1) Correção de datas (vetorizada por produto)
        df, anomalias = corrigir_datas_temporais(
            df,
            max_faltantes=2,
            anomalias=anomalias
        )
        for coluna in ["quantity", "unitvalue", "productcost"]:
            df, anomalias = corrigir_valores_temporais(
            df,
            coluna=coluna,
            window=7,
            anomalias=anomalias
        )
        # 2) Correção de valores (vetorizada) - usa nomes em snake_case
        for coluna in ["quantity", "unitvalue", "productcost"]:
            df, anomalias = tratar_outliers_iqr_por_produto(
                df,
                coluna=coluna,
                iqr_factor=1.5,
                anomalias=anomalias
            )

        # 3) Criação de novas features baseadas na data
        dow = df["date"].dt.dayofweek        
        df["month"] = df["date"].dt.month
        df["is_weekend"] = dow.isin([5, 6]).astype(int)

        # 4) Features cíclicas (seno/cosseno)
        df["day_sin"] = np.sin(2 * np.pi * dow / 7)
        df["day_cos"] = np.cos(2 * np.pi * dow / 7)
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

        # Define o arquivo de saida e salva tudo no formato .csv
        output_file = market_output / csv_file.name

        # Salvamento do arquivo csv e sinalização o salvamento
        df.to_csv(output_file, index=False)
        print(f"    [OK] Salvo em {output_file}")

        # Salva as anomalias encontradas em um arquivo separado
        if anomalias:
            pd.DataFrame(anomalias).to_csv(
                market_output / f"anomalies_{csv_file.stem}.csv",
                index=False
            )
