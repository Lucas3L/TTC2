from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def generate_product_clusters(df, n_clusters=6, train_mask=None):
    """
    Gera clusters de produtos usando apenas dados de treino (se train_mask fornecido),
    com média diária, desvio padrão e preço médio. Evita leakage e viés temporal.
    """
    # Filtra para treino se necessário
    df_calc = df[train_mask] if train_mask is not None else df

    # Agrega: média diária, desvio padrão e preço médio
    stats = df_calc.groupby('product_id').agg({
        'quantity': ['mean', 'std'],
        'unitvalue': 'mean'
    })
    stats.columns = ['avg_qty', 'std_qty', 'avg_price']
    stats = stats.fillna(0).reset_index()

    # Normalização
    scaler = StandardScaler()
    scaled = scaler.fit_transform(stats[['avg_qty', 'std_qty', 'avg_price']])

    # K-Means
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    stats['cluster_kmeans'] = km.fit_predict(scaled)

    return stats[['product_id', 'cluster_kmeans']]