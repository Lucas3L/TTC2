from pathlib import Path
import pandas as pd

# Definição de localização dos arquivos
INPUT_BASE = Path("Dados/preprocessed")
OUTPUT_BASE = Path("Dados/split")

# Realiza a criação das pastas necessarias para execução
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Uso dos dados para treinamento 
train_ratio = 0.7
# Uso dos dados para validar
val_ratio = 0.15

# Valida a pasta mercado dentro da entrada de dados, se não tiver continua
for market_path in INPUT_BASE.iterdir():
    if not market_path.is_dir():
        continue

    # Mostra o nome da pasta 
    market_name = market_path.name
    print(f"\nSeparando dados do mercado: {market_name}")

    # Cria pasta de caida para cada mercado e garante que será criada mesmo que nao exista
    market_output = OUTPUT_BASE / market_name
    market_output.mkdir(parents=True, exist_ok=True)

    # Valida todos os dados, que iniciam com cat e termninam com csv
    for csv_file in market_path.glob("cat*.csv"):
        print(f"  Processando {csv_file.name}")

        # Realiza a leitura do arquivo e salva a data com novo formato
        df = pd.read_csv(csv_file, parse_dates=["Date"])
        
        # Cria um array para guardar dados 
        splits = []

        # Divide os produtos por id e junta eles por cada cronologica
        for product_id, g in df.groupby("product_id"):
            g = g.sort_values("Date")

            # Armazena quantidade de registros( total de valores de g)
            n = len(g)

            # Converte os valores percentuais em inteiro
            train_end = int(n * train_ratio)
                        
            # Converte os valores percentuais em inteiro
            val_end = int(n * (train_ratio + val_ratio))

            # Define a quantidade de linha para treino 0 69-
            g_train = g.iloc[:train_end].copy()

            # Define a quantidade de linha para validação 70 84-
            g_val = g.iloc[train_end:val_end].copy()

            # Define a quantidade de linha para teste 84 99-
            g_test = g.iloc[val_end:].copy()

            # Cria ou subscreve linhas do conjuneto de treino 
            g_train["split"] = "train"

            # Cria ou subscreve linhas do conjuneto de validação 
            g_val["split"] = "val"

            # Cria ou subscreve linhas do conjuneto de teste 
            g_test["split"] = "test"

            # Guarda os 3 valores a cima em uma lista só
            splits.append(pd.concat([g_train, g_val, g_test]))

        # Empilha todas as datas de cada produto em uma, com indices 0,1...
        df_split = pd.concat(splits, ignore_index=True)

        # Define e monta a saida dos arquivos deixando o  nome do csv original
        output_file = market_output / csv_file.name
        df_split.to_csv(output_file, index=False)

        print(f"    O Split foi salvo em {output_file}")
