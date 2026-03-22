SCENARIO_PARAMS = {
    "enabled_scenarios": ["volume", "price", "kmeans"],
    "volume": {
        "q": 3,
        "labels": ["low", "medium", "high"],
        "source_column": "quantity",
    },
    "price": {
        "q": 3,
        "labels": ["cheap", "mid", "expensive"],
        "source_column": "unitvalue",
    },
    "kmeans": {
        "n_clusters": 3,
        "random_state": 42,
        "n_init": "auto",
        "groupby_key": "product_id",
        "feature_columns": ["quantity", "unitvalue"],
    },
}
