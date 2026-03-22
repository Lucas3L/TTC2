import pandas as pd
from sklearn.cluster import KMeans
from src.config.scenario_params import SCENARIO_PARAMS


def apply_scenario(df, scenario):

    if scenario == "volume":
        return group_by_volume(df)

    elif scenario == "price":
        return group_by_price(df)

    elif scenario == "kmeans":
        return group_by_kmeans(df)

    else:
        raise ValueError(f"Cenário inválido: {scenario}")


def group_by_volume(df):
    """
    Agrupamento baseado no volume médio de vendas.
    Divide os produtos em três grupos: baixo, médio e alto volume.
    """
    cfg = SCENARIO_PARAMS["volume"]
    df["volume_cluster"] = pd.qcut(
        df[cfg["source_column"]], q=cfg["q"], labels=cfg["labels"]
    )
    return df


def group_by_price(df):
    """
    Agrupamento baseado no valor unitário do produto.
    Divide os produtos em baratos, médios e caros.
    """
    cfg = SCENARIO_PARAMS["price"]
    df["price_cluster"] = pd.qcut(
        df[cfg["source_column"]], q=cfg["q"], labels=cfg["labels"]
    )
    return df


def group_by_kmeans(df):
    """
    Agrupamento não supervisionado via K-Means usando
    média de volume e preço por produto.
    """
    cfg = SCENARIO_PARAMS["kmeans"]
    X = df.groupby(cfg["groupby_key"])[cfg["feature_columns"]].mean()

    kmeans = KMeans(
        n_clusters=cfg["n_clusters"],
        random_state=cfg["random_state"],
        n_init=cfg["n_init"],
    )
    clusters = kmeans.fit_predict(X)

    cluster_map = dict(zip(X.index, clusters))
    df["kmeans_cluster"] = df["product_id"].map(cluster_map)

    return df
