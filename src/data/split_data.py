from pathlib import Path
import pandas as pd
import gc
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

# --- NOVA LÓGICA: processa cada categoria isoladamente e gera os 3 cenários ---
for market_path in INPUT_BASE.iterdir():
    if not market_path.is_dir():
        continue

    market_name = market_path.name
    print(f"\nSeparando dados do mercado: {market_name}")
    market_output = OUTPUT_BASE / market_name
    market_output.mkdir(parents=True, exist_ok=True)
    market_anomalias = []

    # 1. Carrega e isola cada categoria
    for csv_file in market_path.glob("cat*.csv"):
        category_name = csv_file.stem
        print(f"  Lendo e isolando categoria: {category_name}")
        
        df_cat = pd.read_csv(csv_file)
        df_cat.columns = (
            df_cat.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
        )
        if 'date' in df_cat.columns:
            df_cat['date'] = pd.to_datetime(df_cat['date'], errors='coerce')
        else:
            print(f"    Ignorando {csv_file.name}: coluna 'date' ausente")
            continue
     
        df_cat['source_file'] = csv_file.name

        if 'quantity' not in df_cat.columns or 'unitvalue' not in df_cat.columns:
            print(f"    Ignorando {csv_file.name}: colunas obrigatórias ausentes")
            continue

        # 2. Calcula estatísticas de Treino (Volume e Preço)
        product_stats = []
        for pid, g in df_cat.groupby('product_id'):
            g = g.sort_values('date')
            n = len(g)
            if n <= WINDOW:
                continue

            train_end_idx = int(n * train_ratio)
            train_data = g.iloc[:train_end_idx]

            avg_qty_train = train_data['quantity'].mean()
            avg_price_train = train_data['unitvalue'].mean()
        
            product_stats.append({
                'product_id': pid,
                'avg_qty_train': avg_qty_train,
                'avg_price_train': avg_price_train
            })

        stats_df = pd.DataFrame(product_stats)
        if stats_df.empty:
            continue

        # 3. Processa os 3 Agrupamentos (Volume, Price, K-Means)
        # 3.1 Volume
        try:
            stats_df['vol'] = pd.qcut(stats_df['avg_qty_train'], 3, labels=['low_vol', 'med_vol', 'high_vol'], duplicates='drop')
        except ValueError:
            stats_df['vol'] = 'med_vol'

        # 3.2 Price
        try:
            stats_df['price'] = pd.qcut(stats_df['avg_price_train'], 3, labels=['cheap_price', 'mid_price', 'expensive_price'], duplicates='drop')
        except ValueError:
            stats_df['price'] = 'mid_price'

        # 3.3 k-Means (Bidimensional: Volume + Preço)
        if len(stats_df) >= 3:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(stats_df[['avg_qty_train', 'avg_price_train']])
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            stats_df['kmeans'] = kmeans.fit_predict(scaled_features)
            stats_df['kmeans'] = stats_df['kmeans'].map({0: 'c0_kmeans', 1: 'c1_kmeans', 2: 'c2_kmeans'})
        else:
            stats_df['kmeans'] = 'c0_kmeans'

        # 4. Exporta os Arquivos Físicos para cada cenário
        cenarios = ['vol', 'price', 'kmeans']
        
        for cenario_key in cenarios:
            grupos = stats_df[cenario_key].unique()
            
            for group_name in grupos:
                group_pids = stats_df[stats_df[cenario_key] == group_name]['product_id']
                df_group = df_cat[df_cat['product_id'].isin(group_pids)].copy()
                
                if df_group.empty:
                    continue

                out_rows = []
                for pid, g in df_group.groupby('product_id'):
                    g = g.sort_values('date')
                    n = len(g)
                    if n <= WINDOW:
                        if cenario_key == 'vol': # Loga anomalia apenas 1 vez por produto
                            market_anomalias.append({
                                'file': g['source_file'].iloc[0],
                                'product_id': pid,
                                'issue': 'too_short',
                                'n': int(n)
                            })
                        continue

                    train_end = int(n * train_ratio)
                    val_end = int(n * (train_ratio + val_ratio))
                   
                    g_train = g.iloc[:train_end].copy(); g_train['split'] = 'train'
                    g_val = g.iloc[train_end:val_end].copy(); g_val['split'] = 'val'
                    g_test = g.iloc[val_end:].copy(); g_test['split'] = 'test'
                 
                    out_rows.append(pd.concat([g_train, g_val, g_test]))
             
                if out_rows:
                    out_df = pd.concat(out_rows, ignore_index=True)
                    output_file = market_output / f"{category_name}_{group_name}.csv"
                    out_df.to_csv(output_file, index=False, encoding='utf-8')
                    print(f"    Split salvo: {output_file.name}")

        # Limpeza de memória correta e alinhada
        del df_cat
        gc.collect()

    # Salva anomalias consolidadas do mercado
    if market_anomalias:
        anom_out = market_output / 'anomalies_all.csv'
        anom_df = pd.DataFrame(market_anomalias)
        if anom_out.exists():
            anom_df.to_csv(anom_out, mode='a', header=False, index=False, encoding='utf-8')
        else:
            anom_df.to_csv(anom_out, index=False, encoding='utf-8')

    gc.collect()