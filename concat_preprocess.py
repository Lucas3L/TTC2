import pandas as pd
from pathlib import Path

# Caminhos base
processed_base = Path("Dados/processed")
preprocessed_base = Path("Dados/preprocessed")

# Para cada mercado
for market_dir in processed_base.iterdir():
    if not market_dir.is_dir():
        continue
    market_name = market_dir.name
    print(f"Processando {market_name}")
    out_dir = preprocessed_base / market_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Para cada categoria (ajuste os padrões conforme necessário)
    for cat in ["cat01_non_alcoholic", "cat02_alcoholic"]:
        arquivos = list(market_dir.glob(f"{cat}*.csv"))
        if not arquivos:
            print(f"  Nenhum arquivo encontrado para {cat}")
            continue
        dfs = []
        for arq in arquivos:
            df = pd.read_csv(arq)
            dfs.append(df)
        df_cat = pd.concat(dfs, ignore_index=True)
        out_path = out_dir / f"{cat}.csv"
        df_cat.to_csv(out_path, index=False)
        print(f"  [OK] Salvo {out_path} com {df_cat['product_id'].nunique()} produtos e {len(df_cat)} linhas.")
