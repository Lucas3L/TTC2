from pathlib import Path
import pandas as pd
import gc

# Caminho dos dados processados
PROCESSED_BASE = Path("Dados/processed")
# Caminho do relatório de buracos
REPORT_FILE = Path("Dados/processed/validation_missing_dates.csv")

results = []

for market_path in PROCESSED_BASE.iterdir():
    if not market_path.is_dir():
        continue
    market_name = market_path.name
    for csv_file in market_path.glob("cat*.csv"):
        try:
            # Lê apenas as colunas necessárias
            df = pd.read_csv(csv_file, usecols=["product_id", "date"], parse_dates=["date"])
            if df.empty:
                continue
            # Normaliza datas para 00:00:00
            df["date"] = df["date"].dt.normalize()
            for pid, g in df.groupby("product_id"):
                if g.empty:
                    continue
                start = g["date"].min()
                end = g["date"].max()
                n_expected = (end - start).days + 1
                n_actual = g["date"].nunique()
                if n_actual != n_expected:
                    expected = pd.date_range(start, end, freq="D")
                    missing = expected.difference(g["date"])
                    results.append({
                        "market": market_name,
                        "file": csv_file.name,
                        "product_id": pid,
                        "start_date": start,
                        "end_date": end,
                        "n_expected": n_expected,
                        "n_actual": n_actual,
                        "n_missing": len(missing),
                        # Limita a lista de datas faltantes para não explodir o CSV
                        "missing_dates": ",".join(str(d.date()) for d in missing[:10])
                    })
            del df
            gc.collect()
        except Exception as e:
            print(f"Erro lendo {csv_file}: {e}")
            continue

if results:
    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT_FILE, index=False)
    print(f"Relatório de buracos salvo em {REPORT_FILE}")
else:
    print("Nenhum buraco de data encontrado. Todas as séries estão completas!")
