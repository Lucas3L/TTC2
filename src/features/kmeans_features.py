from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def generate_product_clusters(df, n_clusters=6):
    # Consolida métricas por produto (Total Vendido e Valor Médio)
    stats = df.groupby('product_id').agg({
        'quantity': 'sum',
        'unitvalue': 'mean'
    }).reset_index()
    
    # Normalização para que o volume não domine o preço no K-Means
    scaler = StandardScaler()
    scaled = scaler.fit_transform(stats[['quantity', 'unitvalue']])
    
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    stats['cluster_kmeans'] = km.fit_predict(scaled)
    
    return stats[['product_id', 'cluster_kmeans']]