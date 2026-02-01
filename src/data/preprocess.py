from pathlib import Path
import pandas as pd

# caminho da entrada e saida de dados 
INPUT_BASE = Path("Dados/processed")
OUTPUT_BASE = Path("Dados/preprocessed")

# Cria um diretorio no caminho das pastas se não existirem
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# 
for market_path in INPUT_BASE.iterdir():
    if not market_path.is_dir():
        continue

    market_name = market_path.name
    print(f"\nProcessando mercado: {market_name}")

    market_output = OUTPUT_BASE / market_name
    market_output.mkdir(parents=True, exist_ok=True)

    for csv_file in market_path.glob("cat*.csv"):
        print(f"  Lendo {csv_file.name}")

        df = pd.read_csv(csv_file, parse_dates=["Date"])

        if (df["Quantity"] < 0).any():
            raise ValueError("Quantidade negativa encontrada")

        if (df["UnitValue"] <= 0).any():
            raise ValueError("UnitValue inválido")

        if (df["ProductCost"] <= 0).any():
            raise ValueError("ProductCost inválido")

        if not df["OnPromotion"].isin([0, 1]).all():
            raise ValueError("OnPromotion inválido")

        if not df["Holiday"].isin([0, 1]).all():
            raise ValueError("Holiday inválido")

        for product_id, g in df.groupby("product_id"):
            datas = g["Date"].sort_values()

            esperado = pd.date_range(
                start=datas.min(),
                end=datas.max(),
                freq="D"
            )

            if len(datas) != len(esperado):
                raise ValueError(
                    f"Datas faltantes no produto {product_id}"
                )

        df["day_of_week"] = df["Date"].dt.dayofweek
        df["month"] = df["Date"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        output_file = market_output / csv_file.name
        df.to_csv(output_file, index=False)

        print(f"    ✔ Salvo em {output_file}")