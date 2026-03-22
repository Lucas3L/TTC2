COMMON_MODEL_PARAMS = {
    "window_size": 7,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "prediction_horizon_days": 1,
    "lags": [1, 7, 14],
    "rolling_windows": [3, 7, 14],
    "features_base": [
        "onpromotion",
        "unitvalue",
        "holiday",
        "month",
        "day_of_week",
        "day_sin",
        "day_cos",
        "is_weekend",
    ],
    "loss_by_model": {
        "lstm": "mae",
        "gru": "poisson",
    },
    "training_by_model": {
        "lstm": {"batch_size": 128, "epochs": 50, "patience": 10},
        "gru": {"batch_size": 32, "epochs": 50, "patience": 6},
        "xgboost": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 20,
        },
    },
}
