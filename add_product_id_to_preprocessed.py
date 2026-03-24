import pandas as pd
import sys
from pathlib import Path
import glob

# Uso: python add_product_id_to_preprocessed.py <pasta_preprocessed>
def add_product_id_to_file(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'product_id' not in df.columns:
            # Tenta inferir o product_id do nome do arquivo
            product_id = Path(csv_path).stem.split('_')[-2] + '_' + Path(csv_path).stem.split('_')[-1]
            df['product_id'] = product_id
            df.to_csv(csv_path, index=False)
            print(f"Corrigido: {csv_path} (product_id={product_id})")
        else:
            print(f"OK: {csv_path} (já possui product_id)")
    except Exception as e:
        print(f"Erro ao processar {csv_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python add_product_id_to_preprocessed.py <pasta_preprocessed>")
        sys.exit(1)
    pasta = sys.argv[1]
    arquivos = glob.glob(str(Path(pasta) / '*.csv'))
    for arq in arquivos:
        add_product_id_to_file(arq)
