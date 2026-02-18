import pandas as pd
from pathlib import Path
import sys
import numpy as np


file_path = Path(__file__).resolve()
root = file_path.parents[2]
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.features.feature_engineering import build_features

BASE_DIR = Path("Dados")
INPUT_DIR = BASE_DIR / "split"
OUTPUT_DIR = BASE_DIR / "features"

OUTPUT_DIR.mkdir(exist_ok=True)

# Conversão da data em dataframe para depois ser tratada
def process_market(market_dir):
    market_name = market_dir.name
    print(f"\nProcessando mercado: {market_name}")

    out_market_dir = OUTPUT_DIR / market_name
    out_market_dir.mkdir(exist_ok=True)

    for file in market_dir.glob("*.csv"):
        print(f"  Gerando features para {file.name}")

        df = pd.read_csv(file)
        print(df.columns.tolist())

        X, y = build_features(
            df,
            target_col="Quantity",
            date_col="date",
            window=14
        )

        np.save(out_market_dir / f"X_{file.stem}.npy", X)
        np.save(out_market_dir / f"y_{file.stem}.npy", y)
        
        output_path = out_market_dir / file.name

        print(f"    Salvo em {output_path}")

def main():
    for market_dir in INPUT_DIR.iterdir():
        if market_dir.is_dir():
            process_market(market_dir)


if __name__ == "__main__":
    main()