import pandas as pd
import numpy as np


# --- Correção de datas temporais (usa 'date') ---
def corrigir_datas_temporais(df, max_faltantes=2, anomalias=None):
    if anomalias is None:
        anomalias = []

    df = df.copy()
    # garante datetime em 'date'
    df["date"] = pd.to_datetime(df["date"])

    novos = []
    for product_id, g in df.groupby("product_id"):
        g = g.sort_values("date")
        datas = g["date"]
        esperado = pd.date_range(start=datas.min(), end=datas.max(), freq="D")
        faltantes = esperado.difference(datas)

        if 0 < len(faltantes) <= max_faltantes:
            for data in faltantes:
                linha = g.iloc[-1].copy()
                linha["date"] = data
                linha["observation"] = "date_interpolated"
                for c in ["quantity", "unitvalue", "productcost"]:
                    if c in linha:
                        linha[c] = np.nan
                novos.append(linha)
        elif len(faltantes) > max_faltantes:
            g = g.copy()
            g["observation"] = "date_gap_severe"
            anomalias.extend(g.to_dict("records"))

    if novos:
        df = pd.concat([df, pd.DataFrame(novos)], ignore_index=True)

    return df, anomalias



# --- Correção vetorizada de valores temporais ---
def corrigir_valores_temporais(df, coluna, window=7, anomalias=None):
    """Versão vetorizada que corrige valores <=0 ou NaN usando média móvel por produto.

    Retorna (df, anomalias) para compatibilidade com o pipeline.
    """
    if anomalias is None:
        anomalias = []

    df = df.copy()

    # normaliza nome da coluna para lidar com inputs como 'Quantity' ou 'quantity'
    col = coluna.lower()

    # cria coluna observation se não existir
    if "observation" not in df.columns:
        df["observation"] = "ok"

    # máscara de inválidos
    invalid_mask = df[col].isna() | (df[col] <= 0)

    # média móvel por produto ignorando valores <= 0
    rolling_mean = (
        df.groupby("product_id")[col]
        .transform(lambda x: x.where(x > 0).rolling(window=window, center=True, min_periods=1).mean())
    )

    # aplica correção vetorizada
    df.loc[invalid_mask, "observation"] = f"{col}_corrected_vectorized"
    df[col] = df[col].where(~invalid_mask, rolling_mean)

    # backup com média global por produto
    global_mean = df.groupby("product_id")[col].transform("mean")
    df[col] = df[col].fillna(global_mean)

    # casos extremos: se ainda houver NaN, marca anomalia severa e preenche com 0
    if df[col].isna().any():
        anom_sev = df.loc[df[col].isna()].copy()
        anom_sev["observation"] = f"{col}_invalid_severe"
        anomalias.extend(anom_sev.to_dict("records"))
        df[col] = df[col].fillna(0)

    return df, anomalias
