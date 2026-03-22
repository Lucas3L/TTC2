from pathlib import Path
import pandas as pd
import numpy as np
import gc
import re
from datetime import datetime

# Caminho de entrada (dados brutos)
BASE_PATH = Path("Dados/raw")

# Caminho de saída (dados processados)
OUTPUT_BASE = Path("Dados/processed")

# Heurística para identificar zero potencialmente anômalo de quantidade
ZERO_CONTEXT_WINDOW = 5          # usa 2 dias antes e 2 depois (janela centrada)
ZERO_CONTEXT_MIN_PERIODS = 3     # mínimo de observações válidas no contexto
ZERO_CONTEXT_THRESHOLD = 5.0     # média local acima disso torna zero suspeito
IGNORE_SUNDAY_ZERO = True         # não imputar zeros de domingo (loja possivelmente fechada)
HOLIDAY_COLUMN = "holiday"        # se existir e for 1/True, zero é considerado plausível

# Trilhas de auditoria de descarte
discard_records = []


def register_discard(
    stage: str,
    reason: str,
    severity: str = "warning",
    market: str | None = None,
    category: str | None = None,
    file_path: str | None = None,
    rows_affected: int | None = None,
):
    discard_records.append(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "stage": stage,
            "severity": severity,
            "reason": reason,
            "market": market,
            "category": category,
            "file_path": file_path,
            "rows_affected": rows_affected,
        }
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

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
                register_discard(
                    stage="raw_ingestion",
                    reason=f"read_error: {e}",
                    severity="critical",
                    market=market_name,
                    category=category_code,
                    file_path=str(csv_file),
                )
                continue

            # Normaliza nomes de colunas para evitar KeyError
            df = normalize_columns(df)

            # Garante Date válida
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
            else:
                print(f"    Arquivo ignorado (sem coluna date): {csv_file}")
                register_discard(
                    stage="raw_ingestion",
                    reason="missing_date_column",
                    severity="critical",
                    market=market_name,
                    category=category_code,
                    file_path=str(csv_file),
                )
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
                msg = f"Colunas ausentes na categoria {category_code}: {faltando}"
                print(f"    ⚠️ {msg}")
                register_discard(
                    stage="raw_ingestion",
                    reason=f"missing_required_columns: {sorted(list(faltando))}",
                    severity="critical",
                    market=market_name,
                    category=category_code,
                    file_path=str(category_path),
                )
                continue

            # --- Vetorização de correções temporais (evita loops lentos) ---
            # IMPORTANTE:
            # - quantity pode ser zero legítimo (dias sem venda), então preservamos 0.
            # - unitvalue zero/negativo é inválido e deve ser imputado.
            # - para quantity, corrigimos apenas valores negativos/NaN.
            # - para unitvalue, corrigimos valores <= 0/NaN.
            for coluna in ("quantity", "unitvalue"):
                if coluna not in df_categoria.columns:
                    continue

                if coluna == "quantity":
                    qty = df_categoria[coluna].astype(float)

                    # contexto local sem considerar zeros para checar se zero atual é plausível
                    local_mean_non_zero = (
                        df_categoria.groupby("product_id")[coluna]
                        .transform(
                            lambda x: x.where(x != 0).rolling(
                                window=ZERO_CONTEXT_WINDOW,
                                min_periods=ZERO_CONTEXT_MIN_PERIODS,
                                center=True,
                            ).mean()
                        )
                    )

                    # em dias potencialmente sem operação (domingo/feriado), zero é plausível
                    sunday_closed_mask = (
                        (df_categoria["date"].dt.dayofweek == 6) if IGNORE_SUNDAY_ZERO else pd.Series(False, index=df_categoria.index)
                    )
                    holiday_closed_mask = (
                        df_categoria[HOLIDAY_COLUMN].fillna(0).astype(float).gt(0)
                        if HOLIDAY_COLUMN in df_categoria.columns
                        else pd.Series(False, index=df_categoria.index)
                    )
                    closure_mask = sunday_closed_mask | holiday_closed_mask

                    suspicious_zero_mask = (
                        qty.eq(0)
                        & local_mean_non_zero.notna()
                        & (local_mean_non_zero >= ZERO_CONTEXT_THRESHOLD)
                        & (~closure_mask)
                    )

                    invalid_mask = qty.isna() | (qty < 0) | suspicious_zero_mask
                else:  # unitvalue
                    invalid_mask = df_categoria[coluna].isna() | (df_categoria[coluna] <= 0)

                rolling_mean = (
                    df_categoria.groupby("product_id")[coluna]
                    .transform(lambda x: x.where(~invalid_mask.loc[x.index]).rolling(window=7, min_periods=1, center=True).mean())
                )
                df_categoria.loc[invalid_mask, coluna] = rolling_mean.loc[invalid_mask]

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

            print(f"    [OK] Arquivo salvo: {output_file}")

            # Limpeza de memória imediata após salvar arquivo grande
            del df_categoria
            del dados_categoria
            gc.collect()

        else:
            print(f"    ⚠️ Categoria {category_code} sem produtos")
            register_discard(
                stage="raw_ingestion",
                reason="empty_category_no_products",
                severity="warning",
                market=market_name,
                category=category_code,
                file_path=str(category_path),
            )

    # limpeza por mercado
    gc.collect()

# Persistência da trilha de descarte para auditoria
if discard_records:
    discard_df = pd.DataFrame(discard_records)
    discard_file = OUTPUT_BASE / "discarded_records_raw_ingestion.csv"
    discard_df.to_csv(discard_file, index=False)
    print(f"\n[INFO] Log de descartes salvo em: {discard_file}")
