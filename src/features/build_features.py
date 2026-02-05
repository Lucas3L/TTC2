import pandas as pd

# Conversão da data em dataframe para depois ser tratada
def build_time_features(df, date_col="date"):
    df[date_col] = pd.to_datetime(df[date_col])

    # separação dos dados em categorias de tempo diferentes
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["day"] = df[date_col].dt.day
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    return df

# Criação de features de atraso a serem utilizadas posteriormente separante em 1, 7 e 14 dias
def build_lag_features(
    df,
    target_col,
    group_cols,
    lags=(1, 7, 14)
):
    # Para cada lag cria uma nova coluna com os dados a respectiva lag
    for lag in lags:
        # Criação de coluna conforme a categoria, separa os produtos em colunas e agrupa os valores anteriores
        df[f"{target_col}_lag_{lag}"] = (
            df
            .groupby(group_cols)[target_col]
            .shift(lag)
        )
    return df

# Features para construção de media moveis de acordo com a tendencia
def build_rolling_features(
    df,
    target_col,
    group_cols,
    windows=(7, 14)
):
    # Testa diverentes janelas de memoria 
    for window in windows:
        # Cria e nomeia a nova coluna de acordo com a categoria
        df[f"{target_col}_rolling_mean_{window}"] = (
            df
            # Agrupa os itens e fundamenta baseando-se em dados anteriores e calcula a media dos ultimos 7 dias
            .groupby(group_cols)[target_col]
            .shift(1)
            .rolling(window)
            .mean()
        )
    return df

# Cria uma feature que preve as vendas, define a data como coluna temporal e aprogupa o produto do mercado
def build_features(
    df,
    target_col,
    date_col="date",
    group_cols=("product_id", "market")
):
    # Ordem de agrupamento crescente
    df = df.sort_values([*group_cols, date_col]).reset_index(drop=True)
    # Transforma os dados para que sejam mes, dia , dia da semana.
    df = build_time_features(df, date_col)
    # Cria e monitora a venda do ultimo dia, ultimos 7 e 14.
    df = build_lag_features(df, target_col, group_cols)
    # Calcula a tendencia com media movel
    df = build_rolling_features(df, target_col, group_cols)

    return df
