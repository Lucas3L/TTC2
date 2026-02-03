from pathlib import Path
import pandas as pd

# Caminho de entrada (dados brutos)
BASE_PATH = Path("Dados/raw")

# Caminho de saída (dados processados)
OUTPUT_BASE = Path("Dados/processed")

# Percorre todos os mercados
for market_path in BASE_PATH.iterdir():

    # Verifica somente pastas
    if not market_path.is_dir():
        continue
    
    # Retorna o nome do Mercado apos atribuir
    market_name = market_path.name
    print(f"Processando {market_name}")

    # Envia a rota até a pasta de destino
    market_output = OUTPUT_BASE / market_name
    market_output.mkdir(parents=True, exist_ok=True)

    # Percorre categorias do mercado
    for category_path in market_path.iterdir():

        # Verifica se tem pastas
        if not category_path.is_dir():
            continue

        # Identifica a categoria 
        category_code = category_path.name.split("-")[0]
        print(f"  Categoria {category_code}")

        # Salva os dados em um array
        dados_categoria = []

        # Percorre arquivos CSV da categoria
        for csv_file in category_path.glob("*.csv"):

            # Atribui um id ao produto
            product_id = csv_file.stem.split("_")[0]

            # Le e converte a colona data 
            df = pd.read_csv(csv_file, parse_dates=["Date"])

            # Garante Date válida
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])

            # Se a coluna ProductCost não existir, cria com NaN
            if "ProductCost" not in df.columns:
                df["ProductCost"] = pd.NA


            # Metadados
            df["product_id"] = product_id
            df["category"] = category_code
            df["market"] = market_name

            dados_categoria.append(df)

        # Se houver dados na categoria
        if dados_categoria:
            df_categoria = pd.concat(dados_categoria, ignore_index=True)

            df_categoria = df_categoria.sort_values(
                ["product_id", "Date"]
            )

            # Validação estrutural
            colunas_esperadas = {
                "Date",
                "Quantity",
                "UnitValue",
                "ProductCost",
                "product_id",
                "category",
                "market"
            }

            # Salva as colunas faltantes
            faltando = colunas_esperadas - set(df_categoria.columns)

            # Verifica se há dados faltantes
            if faltando:
                raise ValueError(
                    f"Colunas ausentes na categoria {category_code}: {faltando}"
                )

            # Envia os dados para a pasta de saida
            output_file = market_output / f"cat{category_code}.csv"
            df_categoria.to_csv(output_file, index=False)

            print(f"    ✔ Arquivo salvo: {output_file}")

        else:
            print(f"    ⚠️ Categoria {category_code} sem produtos")
