from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import os
import sys
import gc

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBRegressor

# utilitários
from src.utils.helpers import ensure_dir, normalize_columns, add_lag_features
from src.utils.reproducibility import set_global_seed
from src.models.evaluate import evaluate

# imports de cenário opcionais
try:
    from src.features.scenarios import apply_scenario
except ImportError:
    apply_scenario = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file_path = Path(__file__).resolve()
root = file_path.parents[2]
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.models.evaluate import evaluate

TARGET = 'quantity'
# atributos fixos; usaremos ramificações para adicionar lags dinamicamente
FEATURES_BASE = [
    'onpromotion', 'unitvalue', 'holiday', 'month', 'day_of_week', 'is_weekend'
]
FEATURES = FEATURES_BASE.copy()

WINDOW = 14  # opcional, pode usar lags se desejar

INPUT_BASE = root / "Dados" / "preprocessed"
OUTPUT_BASE = ensure_dir(root / "Resultados" / "xgb")







def run_xgb(train_df, val_df, test_df, features):
    """
    Treina XGBoost e retorna previsões e métricas.

    Assume que o alvo já foi transformado (log1p) externamente.
    """
    # alvo já preparado nas funções chamadoras
    train_y = train_df[TARGET].values
    val_y = val_df[TARGET].values
    test_y = test_df[TARGET].values

    # Features
    X_train = train_df[features].values
    X_val = val_df[features].values
    X_test = test_df[features].values

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train, train_y,
        eval_set=[(X_val, val_y)],
        early_stopping_rounds=20,
        verbose=False
    )

    preds = model.predict(X_test)
    # garantir flatten
    if preds.ndim == 1:
        preds = preds.flatten()
    y_test_inv = np.expm1(test_y)
    preds_inv = np.expm1(preds)

    metrics = evaluate(y_test_inv, preds_inv)

    return preds_inv, metrics


def process_file(csv_file, scenario=None):
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df = normalize_columns(df)
    df = df.dropna(subset=[TARGET])

    if scenario is not None and apply_scenario is not None:
        try:
            df = apply_scenario(df, scenario)
        except Exception as e:
            print(f"Aviso: falha ao aplicar cenário {scenario} -> {e}")

    le = LabelEncoder()
    df['product_id_encoded'] = le.fit_transform(df['product_id'])

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df.sort_values(['product_id', 'date'])

    results = []

    for pid_encoded, g in df.groupby('product_id_encoded'):
        g = g.copy()
        # adiciona lags antes de definir a lista de features usada pelo modelo
        g = add_lag_features(g, TARGET, lags=[1, 7, 14]).dropna()
        features = FEATURES_BASE + ['lag_1', 'lag_7', 'lag_14']

        n = len(g)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_df = g.iloc[:train_end].copy()
        val_df = g.iloc[train_end:val_end].copy()
        test_df = g.iloc[val_end:].copy()

        if len(train_df) < 20 or len(val_df) < 5 or len(test_df) < 5:
            continue

        # transformação segura usando .loc para evitar SettingWithCopyWarning
        for subset in (train_df, val_df, test_df):
            subset.loc[:, TARGET] = np.log1p(subset[TARGET].clip(lower=0))

        # Escala os features com MinMaxScaler ajustado apenas no conjunto de treino
        scaler = MinMaxScaler()
        # garantir tipo numérico antes do fit/transform
        train_df.loc[:, features] = train_df.loc[:, features].astype(float)
        val_df.loc[:, features] = val_df.loc[:, features].astype(float)
        test_df.loc[:, features] = test_df.loc[:, features].astype(float)
        scaler.fit(train_df.loc[:, features])
        train_df.loc[:, features] = scaler.transform(train_df.loc[:, features])
        val_df.loc[:, features] = scaler.transform(val_df.loc[:, features])
        test_df.loc[:, features] = scaler.transform(test_df.loc[:, features])

        try:
            preds, metrics = run_xgb(train_df, val_df, test_df, features)
            # garantia contra array 2D/1D retornado pelo XGBoost
            if preds.ndim == 1:
                preds = preds.flatten()

            original_id = le.inverse_transform([pid_encoded])[0]

            results.append({
                "model": "xgboost",
                "arquivo": csv_file.name,
                "product_id": original_id,
                "mae": metrics["MAE"],
                "rmse": metrics["RMSE"],
                "smape": metrics["sMAPE"]
            })

        except Exception as e:
            print(f"Erro no produto {pid_encoded}: {e}")
            continue

        gc.collect()

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=None,
                        help="Cenário a ser aplicado (volume, price, kmeans)")
    args = parser.parse_args()

    set_global_seed(args.seed)

    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir():
            continue

        market_name = market_path.name
        print(f"\nRodando XGBoost em: {market_name}")

        all_results = []

        for csv_file in market_path.glob("cat*.csv"):
            df_res = process_file(csv_file, scenario=args.scenario)
            if not df_res.empty:
                all_results.append(df_res)

        if all_results:
            final = pd.concat(all_results, ignore_index=True)
            suffix = f"_{args.scenario}" if args.scenario else ""
            out_file = OUTPUT_BASE / f"{market_name}_xgb{suffix}.csv"
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")

            mean_smap = final['smape'].mean()
            print(f"FINAL sMAPE: {mean_smap:.4f}")


if __name__ == "__main__":
    main()