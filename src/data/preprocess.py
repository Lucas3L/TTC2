from pathlib import Path
import pandas as pd
from datetime import datetime
from validators import (corrigir_datas_temporais,
                        corrigir_valores_temporais,
                        tratar_outliers_iqr_por_produto,)
import numpy as np

INPUT_BASE = Path("Dados/processed")
OUTPUT_BASE = Path("Dados/preprocessed")

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
discard_records = []


def register_discard(
    market: str,
    csv_name: str,
    reason: str,
    severity: str = "warning",
    rows_affected: int | None = None,
):
    discard_records.append(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "stage": "preprocess",
            "severity": severity,
            "market": market,
            "csv_file": csv_name,
            "reason": reason,
            "rows_affected": rows_affected,
        }
    )

for market_path in INPUT_BASE.iterdir():

    if not market_path.is_dir():
        continue

    market_name = market_path.name
    print(f"\nProcessando mercado: {market_name}")

    # Organiza os caminhos de saida e cria um diretoria caso não exista
    market_output = OUTPUT_BASE / market_name
    market_output.mkdir(parents=True, exist_ok=True)

    # Lê os arquivos csv que iniciam com cat e imprime o nome do arquivo
    for csv_file in market_path.glob("cat*.csv"):
        print(f"  Lendo {csv_file.name}")

        # Leitura do arquivo e conversão da coluna data
        try:
            df = pd.read_csv(csv_file, parse_dates=["date"])  # espera snake_case
        except Exception as e:
            print(f"    [ERRO] Falha de leitura em {csv_file.name}: {e}")
            register_discard(
                market_name,
                csv_file.name,
                f"read_error: {e}",
                severity="critical",
            )
            continue

        required_cols = {"date", "product_id", "quantity", "unitvalue", "productcost"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            print(f"    [AVISO] Arquivo descartado por colunas ausentes: {sorted(missing_cols)}")
            register_discard(
                market_name,
                csv_file.name,
                f"missing_required_columns: {sorted(missing_cols)}",
                severity="critical",
            )
            continue

        original_len = len(df)
        # Detalha e salva as anomalias encontradas
        df["observation"] = "ok"
        anomalias = []

        # 1) Correção de datas (vetorizada por produto)
        df, anomalias = corrigir_datas_temporais(
            df,
            max_faltantes=2,
            anomalias=anomalias
        )
        for coluna in ["quantity", "unitvalue", "productcost"]:
            df, anomalias = corrigir_valores_temporais(
            df,
            coluna=coluna,
            window=7,
            anomalias=anomalias
        )
        # 2) Correção de valores (vetorizada) - usa nomes em snake_case
        for coluna in ["quantity", "unitvalue", "productcost"]:
            df, anomalias = tratar_outliers_iqr_por_produto(
                df,
                coluna=coluna,
                iqr_factor=1.5,
                anomalias=anomalias
            )

        if df.empty:
            print(f"    [AVISO] Arquivo {csv_file.name} ficou vazio após validações.")
            register_discard(
                market_name,
                csv_file.name,
                "empty_after_validation",
                severity="warning",
                rows_affected=original_len,
            )
            continue

        # 3) Criação de novas features baseadas na data
        dow = df["date"].dt.dayofweek        
        df["month"] = df["date"].dt.month
        df["is_weekend"] = dow.isin([5, 6]).astype(int)

        # 4) Features cíclicas (seno/cosseno)
        df["day_sin"] = np.sin(2 * np.pi * dow / 7)
        df["day_cos"] = np.cos(2 * np.pi * dow / 7)
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

        # Define o arquivo de saida e salva tudo no formato .csv
        output_file = market_output / csv_file.name

        # Salvamento do arquivo csv e sinalização o salvamento
        df.to_csv(output_file, index=False)
        print(f"    [OK] Salvo em {output_file}")

        # Salva as anomalias encontradas em um arquivo separado
        if anomalias:
            pd.DataFrame(anomalias).to_csv(
                market_output / f"anomalies_{csv_file.stem}.csv",
                index=False
            )

        dropped_rows = max(original_len - len(df), 0)
        if dropped_rows > 0:
            register_discard(
                market_name,
                csv_file.name,
                f"rows_dropped_during_processing: {dropped_rows}",
                severity="info",
                rows_affected=dropped_rows,
            )

if discard_records:
    discard_df = pd.DataFrame(discard_records)
    discard_file = OUTPUT_BASE / "discarded_records_preprocess.csv"
    discard_df.to_csv(discard_file, index=False)
    print(f"\n[INFO] Log de descartes do preprocess salvo em: {discard_file}")
