import pandas as pd
import sys
from pathlib import Path

# Uso: python add_product_id_column.py <arquivo.csv> <product_id>
if len(sys.argv) != 3:
    print("Uso: python add_product_id_column.py <arquivo.csv> <product_id>")
    sys.exit(1)

csv_path = Path(sys.argv[1])
product_id = sys.argv[2]

# Lê o CSV
df = pd.read_csv(csv_path)

# Adiciona a coluna product_id
df['product_id'] = product_id

# Salva sobrescrevendo o arquivo original (ou mude para outro nome se preferir)
df.to_csv(csv_path, index=False)

print(f"Coluna 'product_id' adicionada ao arquivo {csv_path} com valor '{product_id}' para todas as linhas.")
