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
        all_X, all_y, all_w = [], [], []
        # Isolamento por produto: cada janela pertence a um único product_id
        for pid, group in df.groupby("product_id"):
            group = group.sort_values("date")
            X_p, y_p, w_p = build_features(
                group,
                target_col="quantity",
                date_col="date",
                window=7,
                positive_weight=3.0
            )
            if len(X_p) > 0:
                all_X.append(X_p)
                all_y.append(y_p)
                all_w.append(w_p)
        if all_X:
            final_X = np.concatenate(all_X).astype('float32')
            final_y = np.concatenate(all_y).astype('float32')
            final_w = np.concatenate(all_w).astype('float32')
            np.save(out_market_dir / f"X_{file.stem}.npy", final_X)
            np.save(out_market_dir / f"y_{file.stem}.npy", final_y)
            np.save(out_market_dir / f"w_{file.stem}.npy", final_w)
            print(f"    Salvo em {out_market_dir}")
        else:
            print(f"    Nenhuma sequência válida para {file.name}")

def main():
    for market_dir in INPUT_DIR.iterdir():
        if market_dir.is_dir():
            process_market(market_dir)


if __name__ == "__main__":
    main()