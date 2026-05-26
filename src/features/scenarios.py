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


def group_by_volume(df, train_mask=None):

    cfg = SCENARIO_PARAMS["volume"]
    if train_mask is not None:
        train_df = df[train_mask]
        _, bins = pd.qcut(
            train_df[cfg["source_column"]], q=cfg["q"], retbins=True, duplicates="drop"
        )
        n_bins = len(bins) - 1
        labels = cfg["labels"][:n_bins]
        df["volume_cluster"] = pd.cut(
            df[cfg["source_column"]], bins=bins, labels=labels, include_lowest=True
        )
    else:
        # Lógica original (com risco de leakage)
        try:
            _, bin_edges = pd.qcut(
                df[cfg["source_column"]], q=cfg["q"], retbins=True, duplicates="drop"
            )
            n_bins = len(bin_edges) - 1
            labels = cfg["labels"][:n_bins]
            df["volume_cluster"] = pd.cut(
                df[cfg["source_column"]], bins=bin_edges, labels=labels, include_lowest=True
            )
        except ValueError:
            df["volume_cluster"] = pd.qcut(
                df[cfg["source_column"]], q=cfg["q"], labels=False, duplicates="drop"
            )
    return df


def group_by_price(df, train_mask=None):

    cfg = SCENARIO_PARAMS["price"]
    if train_mask is not None:
        train_df = df[train_mask]
        _, bins = pd.qcut(
            train_df[cfg["source_column"]], q=cfg["q"], retbins=True, duplicates="drop"
        )
        n_bins = len(bins) - 1
        labels = cfg["labels"][:n_bins]
        df["price_cluster"] = pd.cut(
            df[cfg["source_column"]], bins=bins, labels=labels, include_lowest=True
        )
    else:
        df["price_cluster"] = pd.qcut(
            df[cfg["source_column"]], q=cfg["q"], labels=cfg["labels"], duplicates="drop"
        )
    return df


from sklearn.preprocessing import StandardScaler

def group_by_kmeans(df, train_mask=None):

    cfg = SCENARIO_PARAMS["kmeans"]
    # Fit apenas no treino
    df_calc = df[train_mask] if train_mask is not None else df
    X_calc = df_calc.groupby(cfg["groupby_key"])[cfg["feature_columns"]].mean()
    scaler = StandardScaler()
    kmeans = KMeans(
        n_clusters=cfg["n_clusters"],
        random_state=cfg["random_state"],
        n_init=cfg["n_init"],
    )
    X_scaled = scaler.fit_transform(X_calc)
    kmeans.fit(X_scaled)
    # Predict para todos os produtos
    X_all = df.groupby(cfg["groupby_key"])[cfg["feature_columns"]].mean()
    X_all_scaled = scaler.transform(X_all)
    clusters = kmeans.predict(X_all_scaled)
    cluster_map = dict(zip(X_all.index, clusters))
    df["kmeans_cluster"] = df["product_id"].map(cluster_map)
    return df
