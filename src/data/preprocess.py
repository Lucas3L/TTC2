from pathlib import Path
import pandas as pd
from validators import (corrigir_datas_temporais,
                        corrigir_valores_temporais)


# caminho da entrada e saida de dados 
INPUT_BASE = Path("Dados/processed")
OUTPUT_BASE = Path("Dados/preprocessed")

# Cria um diretorio no caminho das pastas se não existirem
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Deve precorrer tudo o que tem dentro de raw
for market_path in INPUT_BASE.iterdir():
    # ignora o que não é pasta
    if not market_path.is_dir():
        continue

    # Atribui o nome da pasta varianel e imprime sinalizando o mercado
    market_name = market_path.name
    print(f"\nProcessando mercado: {market_name}")

    # Organiza os caminhos de saida e cria um diretoria caso não exista
    market_output = OUTPUT_BASE / market_name
    market_output.mkdir(parents=True, exist_ok=True)

    # Lê os arquivos csv que iniciam com cat e imprime o nome do arquivo
    for csv_file in market_path.glob("cat*.csv"):
        print(f"  Lendo {csv_file.name}")

        # Leitura do arquivo e conversão da coluna data para datatime(tipo de amostragem)
        df = pd.read_csv(csv_file, parse_dates=["Date"])
        # Detalha e salva as anomalias encontradas
        df["observation"] = "ok"
        anomalias = []

        # Chama a Função de correção de datas temporais
        df, anomalias = corrigir_datas_temporais(
            df,
            max_faltantes=2,
            anomalias=anomalias
        )

        # Chama a Função de correção de valores temporais para cada coluna
        for coluna in ["Quantity", "UnitValue", "ProductCost"]:
            df, anomalias = corrigir_valores_temporais(
                df,
                coluna=coluna,
                window=7,
                anomalias=anomalias
        )

        # Criação de novas features baseadas na data semanal
        df["day_of_week"] = df["Date"].dt.dayofweek
        # Criação de novas features baseadas no mês
        df["month"] = df["Date"].dt.month
        #validação boleana para fim de semana
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Define o arquivo de saida e salva tudo no formato .csv
        output_file = market_output / csv_file.name

        # Salvamento do arquivo csv e sinalização o salvamento
        df.to_csv(output_file, index=False)
        print(f"    ✔ Salvo em {output_file}")

        # Salva as anomalias encontradas em um arquivo separado
        if anomalias:
            pd.DataFrame(anomalias).to_csv(
                market_output / f"anomalies_{csv_file.stem}.csv",
                index=False
            )
