import pandas as pd
import numpy as np

# Heurística contextual para zeros de quantidade
ZERO_CONTEXT_WINDOW = 5
ZERO_CONTEXT_MIN_PERIODS = 3
ZERO_CONTEXT_THRESHOLD = 5.0
IGNORE_SUNDAY_ZERO = True
HOLIDAY_COLUMN = "holiday"

def corrigir_datas_temporais(df, max_faltantes=2, anomalias=None):
    """
    Cria linhas para datas faltantes. 
    Preço e Custo são herdados (ffill), mas Quantidade fica como NaN para 
    ser tratada pela interpolação estatística, evitando o 'Time Shift'.
    """
    if anomalias is None: anomalias = []
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    novos = []

    for product_id, g in df.groupby("product_id"):
        g = g.sort_values("date")
        datas = g["date"]
        esperado = pd.date_range(start=datas.min(), end=datas.max(), freq="D")
        faltantes = esperado.difference(datas)

        if 0 < len(faltantes) <= max_faltantes:
            for data in faltantes:
                # Busca o registro anterior para herdar metadados (categoria, preço, etc)
                anterior = g[g["date"] < data]
                if not anterior.empty:
                    linha = anterior.iloc[-1].copy()
                else:
                    linha = g.iloc[0].copy()
                
                linha["date"] = data
                linha["observation"] = "date_interpolated"
                # CRÍTICO: Deixamos quantity como NaN para a função seguinte interpolar,
                # impedindo que a Terça receba o valor exato da Segunda (ffill).
                if "quantity" in linha: linha["quantity"] = np.nan
                novos.append(linha)
        elif len(faltantes) > max_faltantes:
            anom_g = g.copy()
            anom_g["observation"] = "date_gap_severe"
            anomalias.extend(anom_g.to_dict("records"))

    if novos:
        df = pd.concat([df, pd.DataFrame(novos)], ignore_index=True)
    return df, anomalias

def tratar_outliers_iqr_por_produto(df, coluna, iqr_factor=None, anomalias=None):
    if anomalias is None: anomalias = []
    df = df.copy()
    col = coluna.lower()
    if iqr_factor is None:
        iqr_factor = 3.0 if col == "quantity" else 1.5
    
    for pid, g in df.groupby("product_id"):
        q1 = g[col].quantile(0.25)
        q3 = g[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + (iqr_factor * iqr)
        lower_bound = q1 - (iqr_factor * iqr)

        # Proteção: Promoção e Feriado NÃO sofrem clipping (são picos reais)
        mask_outlier = (
            ((g[col] < lower_bound) | (g[col] > upper_bound)) & 
            (g.get('onpromotion', 0).fillna(0) == 0) & 
            (g.get('holiday', 0).fillna(0) == 0)
        )

        if mask_outlier.any():
            idx = g[mask_outlier].index
            outliers = g[mask_outlier].copy()
            outliers["observation"] = f"{col}_outlier_clipped"
            anomalias.extend(outliers.to_dict("records"))
            
            # Clipping: Trava no limite estatístico para não "cegar" o modelo
            df.loc[idx, col] = np.clip(df.loc[idx, col], lower_bound, upper_bound)
            df.loc[idx, "observation"] = f"{col}_clipped"
            
    return df, anomalias

def corrigir_valores_temporais(df, coluna, window=7, anomalias=None):
    """
    Corrige falhas usando interpolação linear (tendência) em vez de ffill (repetição).
    Isso garante que terça seja um degrau entre segunda e quarta, e não uma cópia.
    """
    if anomalias is None: anomalias = []
    df = df.copy()
    col = coluna.lower()
    
    # Identificação de falhas
    if col == "quantity":
        qty = df[col].astype(float)
        local_mean = df.groupby("product_id")[col].transform(
            lambda x: x.where(x != 0).rolling(window=ZERO_CONTEXT_WINDOW, min_periods=1, center=True).mean()
        )
        
        sunday_mask = (df["date"].dt.dayofweek == 6) if IGNORE_SUNDAY_ZERO else False
        holiday_mask = df[HOLIDAY_COLUMN].fillna(0).gt(0) if HOLIDAY_COLUMN in df.columns else False
        
        suspicious_zero = (qty == 0) & (~sunday_mask) & (~holiday_mask) & (local_mean >= ZERO_CONTEXT_THRESHOLD)
        invalid_mask = qty.isna() | (qty < 0) | suspicious_zero
    else:
        invalid_mask = df[col].isna() | (df[col] <= 0)

    # TRATAMENTO: Substitui inválidos por NaN para interpolar
    df.loc[invalid_mask, col] = np.nan
    
    # INTERPOLAÇÃO LINEAR: O segredo para evitar o "terça vira segunda"
    # Ele traça uma linha reta entre os pontos válidos.
    df[col] = df.groupby("product_id")[col].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    
    df.loc[invalid_mask, "observation"] = f"{col}_interpolated"
    
    # Fallbacks de segurança para UnitValue e Cost (Preço não pode ser zero)
    if col in ["unitvalue", "productcost"]:
        df[col] = df[col].fillna(df.groupby("product_id")[col].transform("mean"))
        df[col] = df[col].fillna(df.groupby("category")[col].transform("mean"))
        # Caso extremo (produto novo sem histórico nenhum)
        df[col] = df[col].fillna(1.0) 
        
    return df, anomalias