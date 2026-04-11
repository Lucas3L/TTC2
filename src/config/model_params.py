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
    ],
    "loss_by_model": {
        "lstm": "mse",
        "gru": "mse",
    },
    "training_by_model": {
        "lstm": {"batch_size": 32, "epochs": 50, "patience": 15},
        "gru": {"batch_size": 32, "epochs": 50, "patience": 15},
        "xgboost": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
}