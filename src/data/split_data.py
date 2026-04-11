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


# --- NOVA LÓGICA: processa todos os produtos do mercado juntos para cenários globais ---
for market_path in INPUT_BASE.iterdir():
    if not market_path.is_dir():
        continue

    market_name = market_path.name
    print(f"\nSeparando dados do mercado: {market_name}")
    market_output = OUTPUT_BASE / market_name
    market_output.mkdir(parents=True, exist_ok=True)
    market_anomalias = []

    # 1. Carrega todos os produtos do mercado
    market_data = []
    for csv_file in market_path.glob("cat*.csv"):
        print(f"  Lendo {csv_file.name}")
        df = pd.read_csv(csv_file)
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
        )
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'date' not in df.columns:
            print(f"    Ignorando {csv_file.name}: coluna 'date' ausente")
            continue
        df['source_file'] = csv_file.name
        market_data.append(df)
    if not market_data:
        continue
    full_market_df = pd.concat(market_data, ignore_index=True)


    # 2. Calcula estatísticas APENAS no treino de cada produto
    if 'quantity' not in full_market_df.columns:
        print(f"    Ignorando mercado {market_name}: coluna 'quantity' ausente")
        continue
    product_stats = []
    for pid, g in full_market_df.groupby('product_id'):
        g = g.sort_values('date')
        n = len(g)
        if n <= WINDOW:
            continue
        train_end_idx = int(n * train_ratio)
        train_data = g.iloc[:train_end_idx]
        avg_qty_train = train_data['quantity'].mean()
        avg_price_train = train_data['unitvalue'].mean() if 'unitvalue' in train_data.columns else 0
        product_stats.append({
            'product_id': pid,
            'avg_qty_train': avg_qty_train,
            'avg_price_train': avg_price_train
        })
    stats_df = pd.DataFrame(product_stats)
    if stats_df.empty:
        continue
    stats_df['group'] = pd.qcut(stats_df['avg_qty_train'], 3, labels=['low_vol', 'med_vol', 'high_vol'])

    # 3. Split e salva por grupo, usando apenas o grupo definido pelo treino
    for group_name in ['low_vol', 'med_vol', 'high_vol']:
        group_pids = stats_df[stats_df['group'] == group_name]['product_id']
        df_group = full_market_df[full_market_df['product_id'].isin(group_pids)].copy()
        if df_group.empty:
            continue
        out_rows = []
        for pid, g in df_group.groupby('product_id'):
            g = g.sort_values('date')
            n = len(g)
            if n <= WINDOW:
                market_anomalias.append({
                    'file': g['source_file'].iloc[0],
                    'product_id': pid,
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
            out_rows.append(pd.concat([g_train, g_val, g_test]))
        if out_rows:
            out_df = pd.concat(out_rows, ignore_index=True)
            output_file = market_output / f"{group_name}.csv"
            out_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"    Split do grupo {group_name} salvo em {output_file}")

    # Salva anomalias consolidadas do mercado
    if market_anomalias:
        anom_out = market_output / 'anomalies_all.csv'
        anom_df = pd.DataFrame(market_anomalias)
        if anom_out.exists():
            anom_df.to_csv(anom_out, mode='a', header=False, index=False, encoding='utf-8')
        else:
            anom_df.to_csv(anom_out, index=False, encoding='utf-8')

    gc.collect()
