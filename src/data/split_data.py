from pathlib import Path
import pandas as pd
import gc
import numpy as np

# Definição de localização dos arquivos
INPUT_BASE = Path("Dados/preprocessed")
OUTPUT_BASE = Path("Dados/split")

# Realiza a criação das pastas necessarias para execução
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Uso dos dados para treinamento 
train_ratio = 0.7
# Uso dos dados para validar
val_ratio = 0.15

# Janela mínima para considerar um produto (evita splits inválidos)
WINDOW = 7

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

    # mantem anomalias do mercado para um arquivo consolidado
    market_anomalias = []

    # Valida todos os dados, que iniciam com cat e termninam com csv
    for csv_file in market_path.glob("cat*.csv"):
        print(f"  Processando {csv_file.name}")

        # Realiza a leitura do arquivo
        df = pd.read_csv(csv_file)

        # Normaliza nomes de colunas para snake_case
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
        )
        
        # Converte coluna date para datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='ISO8601', errors='coerce')

        if 'date' not in df.columns:
            print(f"    Ignorando {csv_file.name}: coluna 'date' ausente")
            continue

        # arquivo de saída
        output_file = market_output / csv_file.name

        # escreve por produto (stream), evitando juntar tudo em memória
        first_write = True

        for product_id, g in df.groupby('product_id'):
            g = g.sort_values('date')
            n = len(g)

            if n <= WINDOW:
                market_anomalias.append({
                    'file': csv_file.name,
                    'product_id': product_id,
                    'issue': 'too_short',
                    'n': int(n)
                })
                continue

            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            g_train = g.iloc[:train_end].copy()
            g_val = g.iloc[train_end:val_end].copy()
            g_test = g.iloc[val_end:].copy()

            g_train['split'] = 'train'
            g_val['split'] = 'val'
            g_test['split'] = 'test'

            out_chunk = pd.concat([g_train, g_val, g_test])

            out_chunk.to_csv(output_file, mode='w' if first_write else 'a', header=first_write, index=False, encoding='utf-8')
            first_write = False

        # salva anomalias consolidadas do mercado
        if market_anomalias:
            anom_out = market_output / 'anomalies_all.csv'
            anom_df = pd.DataFrame(market_anomalias)
            if anom_out.exists():
                anom_df.to_csv(anom_out, mode='a', header=False, index=False, encoding='utf-8')
            else:
                anom_df.to_csv(anom_out, index=False, encoding='utf-8')

        print(f"    O Split foi salvo em {output_file}")

        del df
        gc.collect()
