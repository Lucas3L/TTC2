import pandas as pd
from sklearn.cluster import KMeans


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
    df["volume_cluster"] = pd.qcut(
        df["quantity"], q=3, labels=["low", "medium", "high"]
    )
    return df


def group_by_price(df):
    """
    Agrupamento baseado no valor unitário do produto.
    Divide os produtos em baratos, médios e caros.
    """
    df["price_cluster"] = pd.qcut(
        df["unitvalue"], q=3, labels=["cheap", "mid", "expensive"]
    )
    return df


def group_by_kmeans(df):
    """
    Agrupamento não supervisionado via K-Means usando
    média de volume e preço por produto.
    """
    X = df.groupby("product_id")[["quantity", "unitvalue"]].mean()

    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X)

    cluster_map = dict(zip(X.index, clusters))
    df["kmeans_cluster"] = df["product_id"].map(cluster_map)

    return df