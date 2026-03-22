import pandas as pd
import numpy as np

# Heurística contextual para zeros de quantidade
ZERO_CONTEXT_WINDOW = 5
ZERO_CONTEXT_MIN_PERIODS = 3
ZERO_CONTEXT_THRESHOLD = 5.0
IGNORE_SUNDAY_ZERO = True
HOLIDAY_COLUMN = "holiday"


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


# --- Tratamento de Outliers usando IQR por produto ---
def tratar_outliers_iqr_por_produto(df, coluna, iqr_factor=1.5, anomalias=None):
    """Remove outliers usando IQR por produto.
    
    Retorna (df, anomalias) para compatibilidade com o pipeline.
    """
    if anomalias is None:
        anomalias = []
    
    df = df.copy()
    col = coluna.lower()
    
    # cria coluna observation se não existir
    if "observation" not in df.columns:
        df["observation"] = "ok"
    
    for product_id, g in df.groupby("product_id"):
        Q1 = g[col].quantile(0.25)
        Q3 = g[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        
        # identifica outliers
        outlier_mask = (g[col] < lower_bound) | (g[col] > upper_bound)
        
        if outlier_mask.any():
            outliers = g[outlier_mask].copy()
            outliers["observation"] = f"{col}_outlier_removed"
            anomalias.extend(outliers.to_dict("records"))
            
            # substitui outliers pela mediana do produto
            median = g[col].median()
            df.loc[g[outlier_mask].index, col] = median
            df.loc[g[outlier_mask].index, "observation"] = f"{col}_median_imputed"
    
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
    if col == "quantity":
        qty = df[col].astype(float)

        local_mean_non_zero = (
            df.groupby("product_id")[col]
            .transform(
                lambda x: x.where(x != 0).rolling(
                    window=ZERO_CONTEXT_WINDOW,
                    min_periods=ZERO_CONTEXT_MIN_PERIODS,
                    center=True,
                ).mean()
            )
        )

        if "date" in df.columns and IGNORE_SUNDAY_ZERO:
            sunday_closed_mask = df["date"].dt.dayofweek == 6
        else:
            sunday_closed_mask = pd.Series(False, index=df.index)

        holiday_closed_mask = (
            df[HOLIDAY_COLUMN].fillna(0).astype(float).gt(0)
            if HOLIDAY_COLUMN in df.columns
            else pd.Series(False, index=df.index)
        )
        closure_mask = sunday_closed_mask | holiday_closed_mask

        suspicious_zero_mask = (
            qty.eq(0)
            & local_mean_non_zero.notna()
            & (local_mean_non_zero >= ZERO_CONTEXT_THRESHOLD)
            & (~closure_mask)
        )

        invalid_mask = qty.isna() | (qty < 0) | suspicious_zero_mask
        valid_series = qty.where(~invalid_mask)
    else:
        invalid_mask = df[col].isna() | (df[col] <= 0)
        valid_series = df[col].where(df[col] > 0)

    # média móvel por produto para imputação de pontos inválidos
    rolling_mean = (
        valid_series.groupby(df["product_id"])
        .transform(lambda x: x.rolling(window=window, center=True, min_periods=1).mean())
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
