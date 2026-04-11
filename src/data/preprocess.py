import gc
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Importação dos validadores customizados
from validators import (
    corrigir_datas_temporais,
    corrigir_valores_temporais,
    tratar_outliers_iqr_por_produto,
)

# --- Configurações de Ambiente ---
INPUT_BASE = Path("Dados/raw")
OUTPUT_BASE = Path("Dados/preprocessed")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

discard_records = []

def register_discard(stage, reason, market, category, file_path):
    """Registra falhas de integridade para auditoria do TCC."""
    discard_records.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "reason": reason,
        "market": market,
        "category": category,
        "file_path": file_path
    })

# --- Pipeline Principal ---
for market_path in INPUT_BASE.iterdir():
    if not market_path.is_dir():
        continue
    
    market_name = market_path.name
    print(f"\n>>> Processando Mercado: {market_name}")
    market_output = OUTPUT_BASE / market_name
    market_output.mkdir(parents=True, exist_ok=True)

    for category_path in market_path.iterdir():
        if not category_path.is_dir():
            continue
        
        category_code = category_path.name.split("-")[0]
        print(f"  Categoria: {category_code}")
        dados_categoria = []

        # 1. Fase de Ingestão e Padronização de Colunas
        for csv_file in category_path.glob("*.csv"):
            try:
                # Leitura robusta com detecção de separador e encoding sig (remove BOM do Excel)
                df = pd.read_csv(csv_file, sep=None, engine='python', encoding='utf-8-sig')
                
                # Normalização de nomes (remove espaços, pontos e underscores temporariamente para mapeamento)
                df.columns = [c.lower().strip().replace(' ', '').replace('.', '') for c in df.columns]
                
                # Injeção Crítica: Prioridade ao ID do arquivo (metadado) para evitar perda de dados
                product_id_from_file = csv_file.stem.split("_")[0]
                df['product_id'] = product_id_from_file

                # Mapeamento para o padrão do Pipeline
                mapping = {
                    'productcost': 'productcost',
                    'product_cost': 'productcost',
                    'quantity': 'quantity',
                    'date': 'date',
                    'unitvalue': 'unitvalue',
                    'onpromotion': 'onpromotion',
                    'holiday': 'holiday'
                }
                df = df.rename(columns=mapping)
                
                # Garantia de Tipagem
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                
                # SÓ dropamos se a data for inválida. Quantity NaN será tratado nos validadores.
                df = df.dropna(subset=["date"])
                df["category"] = category_code
                df["market"] = market_name
                
                dados_categoria.append(df)
                
            except Exception as e:
                register_discard("raw_ingestion", str(e), market_name, category_code, str(csv_file))

        # 2. Fase de Limpeza Estatística e Consolidação
        if dados_categoria:
            df_cat = pd.concat(dados_categoria, ignore_index=True)
            
            # GARANTIA DE INTEGRIDADE: Remove duplicatas (evita bias no modelo)
            df_cat = df_cat.drop_duplicates(subset=["product_id", "date"], keep="last")
            
            anomalias = []

            # Sequência lógica de tratamento: Datas -> Valores -> Outliers
            df_cat, anomalias = corrigir_datas_temporais(df_cat, anomalias=anomalias)

            for col in ["quantity", "unitvalue", "productcost"]:
                if col in df_cat.columns:
                    # Resolve falhas (ffill/bfill/rolling mean)
                    df_cat, anomalias = corrigir_valores_temporais(df_cat, col, anomalias=anomalias)
                    # Resolve extremos com Clipping (Fator 3.0 para preservar picos saudáveis)
                    df_cat, anomalias = tratar_outliers_iqr_por_produto(df_cat, col, iqr_factor=3.0, anomalias=anomalias)

            # 3. Engenharia de Atributos Cíclicos (Sazonalidade para Redes Neurais)
            # Transforma tempo linear em coordenadas circulares
            dow = df_cat["date"].dt.dayofweek
            df_cat["day_sin"] = np.sin(2 * np.pi * dow / 7)
            df_cat["day_cos"] = np.cos(2 * np.pi * dow / 7)
            
            df_cat["month"] = df_cat["date"].dt.month
            df_cat["month_sin"] = np.sin(2 * np.pi * (df_cat["month"]-1) / 12)
            df_cat["month_cos"] = np.cos(2 * np.pi * (df_cat["month"]-1) / 12)

            # Limpeza final de colunas de apoio
            df_cat = df_cat.drop(columns=['productid', 'unnamed:0'], errors='ignore')

            # 4. Persistência dos Artefatos
            output_file = market_output / f"cat{category_code}.csv"
            df_cat.to_csv(output_file, index=False)
            print(f"    [OK] {output_file.name} salvo com {len(df_cat)} registros.")

            if anomalias:
                anomalias_file = market_output / f"anomalias_{category_code}.csv"
                pd.DataFrame(anomalias).to_csv(anomalias_file, index=False)

            # Gestão de memória para datasets grandes
            del df_cat
            dados_categoria.clear()
            gc.collect()

# Salva trilha de descarte para auditoria do TCC
if discard_records:
    pd.DataFrame(discard_records).to_csv(OUTPUT_BASE / "discarded_pipeline.csv", index=False)
    print(f"\n[INFO] Relatório de descartes gerado em: {OUTPUT_BASE / 'discarded_pipeline.csv'}")

print("\n================ PREPROCESSAMENTO CONCLUÍDO ================")