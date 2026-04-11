from pathlib import Path
from src.config.scenario_params import SCENARIO_PARAMS


BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"


DEFAULT_EXPERIMENT_CONFIG = {
    "random_seed": 42,
    "n_replicas": 1,
    "scenarios": SCENARIO_PARAMS["enabled_scenarios"],
    "models": {
        "LSTM": SRC_DIR / "models" / "lstm_model.py",
        "GRU": SRC_DIR / "models" / "gru_model.py",
        "XGBoost": SRC_DIR / "models" / "xgboost_model.py",
        "Baseline": SRC_DIR / "models" / "baseline_zero_aware.py",
    },
    # Filtro temporal global opcional para todos os modelos.
    # formato YYYY-MM-DD ou None para sem filtro.
    "date_from": None,
    "date_to": None,
}
