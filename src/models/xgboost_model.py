import os
import gc
import pandas as pd
import numpy as np
import argparse

from src.utils.project_paths import add_project_root_to_sys_path
root = add_project_root_to_sys_path(__file__)

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# utilitários
from src.utils.helpers import ensure_dir, normalize_columns, add_lag_features
from src.utils.reproducibility import set_global_seed
from src.models.evaluate import evaluate
from src.utils.metrics import naive_market_smape
from src.config.model_params import COMMON_MODEL_PARAMS

# imports de cenário opcionais
try:
    from src.features.scenarios import apply_scenario
except ImportError:
    apply_scenario = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TARGET = 'quantity'
# atributos fixos;  ramificações para adicionar lags dinamicamente
FEATURES_BASE = COMMON_MODEL_PARAMS["features_base"]
FEATURES = FEATURES_BASE.copy()
LAGS = COMMON_MODEL_PARAMS["lags"]
ROLLING_WINDOWS = COMMON_MODEL_PARAMS["rolling_windows"]
QTY_FEATURES = [f"lag_{lag}" for lag in LAGS] + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]

WINDOW = COMMON_MODEL_PARAMS["window_size"]
TRAIN_RATIO = COMMON_MODEL_PARAMS["train_ratio"]
VAL_RATIO = COMMON_MODEL_PARAMS["val_ratio"]

INPUT_BASE = root / "Dados" / "preprocessed"
OUTPUT_BASE = ensure_dir(root / "Resultados" / "xgb")




def compute_market_max(market_path):
    max_vals = []
    for csv_file in market_path.glob("cat*.csv"):
        try:
            df = pd.read_csv(csv_file)
            df = normalize_columns(df)
            if TARGET in df.columns:
                max_vals.append(df[TARGET].max())
        except Exception:
            continue

    market_max = np.nanmax(max_vals) if max_vals else np.nan
    if not np.isfinite(market_max) or market_max <= 0:
        market_max = 1.0
    return float(market_max)



def run_xgb(train_df, val_df, test_df, features):

    # alvo já preparado nas funções chamadoras
    train_y = train_df[TARGET].values
    val_y = val_df[TARGET].values
    test_y = test_df[TARGET].values

    # Features
    X_train = train_df[features].values
    X_val = val_df[features].values
    X_test = test_df[features].values

    # mais peso para alvos positivos em séries esparsas
    sample_weight = np.where(train_y > 0, 3.0, 1.0)

    xgb_cfg = COMMON_MODEL_PARAMS["training_by_model"]["xgboost"]
    model = XGBRegressor(
        n_estimators=xgb_cfg["n_estimators"],
        learning_rate=xgb_cfg["learning_rate"],
        max_depth=xgb_cfg["max_depth"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train,
        train_y,
        sample_weight=sample_weight,
        eval_set=[(X_val, val_y)],
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        verbose=False
    )

    preds = model.predict(X_test).flatten()


    return preds, test_y


def process_file(csv_file,  market_max, scenario=None, date_from=None, date_to=None):
    df = pd.read_csv(csv_file, parse_dates=['date'])
    df = normalize_columns(df)
    df = df.dropna(subset=[TARGET])
    if date_from is not None:
        df = df[df["date"] >= pd.to_datetime(date_from)]
    if date_to is not None:
        df = df[df["date"] <= pd.to_datetime(date_to)]

    if scenario is not None and apply_scenario is not None:
        try:
            df = apply_scenario(df, scenario)
        except Exception as e:
            print(f"Aviso: falha ao aplicar cenário {scenario} -> {e}")

    le = LabelEncoder()
    df['product_id_encoded'] = le.fit_transform(df['product_id'])

    df = df.sort_values(["product_id_encoded", "date"])

    results = []
    pred_rows = []

    for pid_encoded, g in df.groupby('product_id_encoded'):
        g = g.copy()
        # adiciona lags antes de definir a lista de features usada pelo modelo
        # create lags then only drop entries missing target or predictor features
        g[TARGET] = g[TARGET].clip(lower=0) / market_max
        if "unitvalue" in g.columns:
            g["unitvalue"] = g["unitvalue"].clip(lower=0)
        g = add_lag_features(g, TARGET, lags=LAGS, rolling_windows=ROLLING_WINDOWS)
       
        for c in QTY_FEATURES:
            if c in g.columns:
                g[c] = g[c] / market_max

        features = [c for c in FEATURES_BASE if c in g.columns] + [c for c in QTY_FEATURES if c in g.columns]
        g = g.dropna(subset=features + [TARGET])

        n = len(g)
        if n <= WINDOW:
            continue
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_df = g.iloc[:train_end].copy()
        val_df = g.iloc[train_end:val_end].copy()
        test_df = g.iloc[val_end:].copy()

        if len(train_df) < 20 or len(val_df) < 5 or len(test_df) < 5:
            continue
        try:
            preds, y_true = run_xgb(train_df, val_df, test_df, features)

            y_true_real = y_true * market_max
            preds_real = np.clip(preds, 0, None) * market_max
            metrics = evaluate(y_true_real, preds_real)
            original_id = le.inverse_transform([pid_encoded])[0]

            results.append(
                {
                    "model": "xgboost",
                    "arquivo": csv_file.name,
                    "product_id": original_id,
                    "mae": metrics["MAE"],
                    "rmse": metrics["RMSE"],
                    "smape": metrics["sMAPE"],
                }
            )
            for dt, y_t, y_p in zip(test_df["date"], y_true_real, preds_real):
                pred_rows.append({
                    "model": "xgboost",
                    "arquivo": csv_file.name,
                    "product_id": original_id,
                    "date": dt,
                    "y_true": float(y_t),
                    "y_pred": float(y_p),
                    "scenario": scenario
                })
        except Exception as e:
            print(f"Erro no produto {pid_encoded}: {e}")
            continue

        gc.collect()

    return pd.DataFrame(results), pd.DataFrame(pred_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=None,
                        help="Cenário a ser aplicado (volume, price, kmeans)")
    parser.add_argument("--date-from", type=str, default=None,
                        help="Data inicial (YYYY-MM-DD) para filtrar histórico")
    parser.add_argument("--date-to", type=str, default=None,
                        help="Data final (YYYY-MM-DD) para filtrar histórico")
    args = parser.parse_args()

    set_global_seed(args.seed)

    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir():
            continue

        market_name = market_path.name
        market_max = compute_market_max(market_path)
        print(f"\nRodando XGBoost em: {market_name} | max_global_quantity={market_max:.4f}")

        all_results = []
        all_predictions = []

        for csv_file in market_path.glob("cat*.csv"):
            df_res, df_pred = process_file(
                csv_file, market_max=market_max, scenario=args.scenario,
                date_from=args.date_from, date_to=args.date_to
            )
            if not df_res.empty:
                all_results.append(df_res)
            if not df_pred.empty:
                all_predictions.append(df_pred)

        if all_results:
            final = pd.concat(all_results, ignore_index=True)
            suffix = f"_{args.scenario}" if args.scenario else ""
            out_file = OUTPUT_BASE / f"{market_name}_xgb{suffix}.csv"
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")
            if all_predictions:
                pred_file = OUTPUT_BASE / f"{market_name}_xgb{suffix}_predictions.csv"
                pd.concat(all_predictions, ignore_index=True).to_csv(pred_file, index=False)
                print(f"  Curva real vs predito salva em {pred_file}")

            print(f"FINAL sMAPE: {final['smape'].mean():.4f}")
        else:
            market_smap = naive_market_smape(market_path)
            print(f"FALLBACK market sMAPE: {market_smap:.4f}")
            print(f"FINAL sMAPE: {market_smap:.4f}")


if __name__ == "__main__":
    main()
