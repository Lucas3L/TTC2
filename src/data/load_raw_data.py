from pathlib import Path
import pandas as pd
import numpy as np
import gc
import re

# Caminho de entrada (dados brutos)
BASE_PATH = Path("Dados/raw")

# Caminho de saída (dados processados)
OUTPUT_BASE = Path("Dados/processed")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converte nomes de colunas para snake_case estável.

    - lower case
    - espaços -> _
    - remove caracteres não alfanuméricos exceto _
    """
    cols = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^0-9a-z_]", "", regex=True)
    )
    df.columns = cols
    return df


# Percorre todos os mercados
for market_path in BASE_PATH.iterdir():

    # Verifica somente pastas
    if not market_path.is_dir():
        continue

    # Retorna o nome do Mercado
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
            try:
                # Atribui um id ao produto
                product_id = csv_file.stem.split("_")[0]

                # Le e converte a coluna data (sem elevar memória)
                df = pd.read_csv(csv_file, parse_dates=["Date"])  # original header may vary
            except Exception as e:
                print(f"    Erro lendo {csv_file}: {e}")
                continue

            # Normaliza nomes de colunas para evitar KeyError
            df = normalize_columns(df)

            # Garante Date válida
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
            else:
                print(f"    Arquivo ignorado (sem coluna date): {csv_file}")
                continue

            # Se a coluna productcost não existir, cria com NaN
            if "productcost" not in df.columns:
                df["productcost"] = pd.NA

            # Metadados
            df["product_id"] = product_id
            df["category"] = category_code
            df["market"] = market_name

            dados_categoria.append(df)

        # Se houver dados na categoria
        if dados_categoria:
            # concat com copy=False para reduzir cópias de RAM quando possível
            df_categoria = pd.concat(dados_categoria, ignore_index=True, copy=False)

            # Normaliza colunas novamente após concat (segurança)
            df_categoria = normalize_columns(df_categoria)

            df_categoria = df_categoria.sort_values(["product_id", "date"])

            # Validação estrutural (em snake_case)
            colunas_esperadas = {
                "date",
                "quantity",
                "unitvalue",
                "productcost",
                "product_id",
                "category",
                "market",
            }

            faltando = colunas_esperadas - set(df_categoria.columns)

            if faltando:
                raise ValueError(f"Colunas ausentes na categoria {category_code}: {faltando}")

            # --- Vetorização de correções temporais (evita loops lentos) ---
            # Substitui valores <= 0 por NaN e preenche com média móvel por produto
            for coluna in ("quantity", "unitvalue"):
                if coluna in df_categoria.columns:
                    df_categoria[coluna] = df_categoria.groupby("product_id")[coluna].transform(
                        lambda x: x.mask(x <= 0).fillna(x.rolling(window=7, min_periods=1, center=True).mean())
                    )

            # Engenharia de features temporais (inclui cíclicas)
            df_categoria["month"] = df_categoria["date"].dt.month
            df_categoria["day_of_week"] = df_categoria["date"].dt.weekday
            df_categoria["month_sin"] = np.sin(2 * np.pi * df_categoria["month"] / 12)
            df_categoria["month_cos"] = np.cos(2 * np.pi * df_categoria["month"] / 12)
            df_categoria["dow_sin"] = np.sin(2 * np.pi * df_categoria["day_of_week"] / 7)
            df_categoria["dow_cos"] = np.cos(2 * np.pi * df_categoria["day_of_week"] / 7)

            # Envia os dados para a pasta de saida
            output_file = market_output / f"cat{category_code}.csv"
            df_categoria.to_csv(output_file, index=False)

            print(f"    ✔ Arquivo salvo: {output_file}")

            # Limpeza de memória imediata após salvar arquivo grande
            del df_categoria
            del dados_categoria
            gc.collect()

        else:
            print(f"    ⚠️ Categoria {category_code} sem produtos")

    # limpeza por mercado
    gc.collect()
