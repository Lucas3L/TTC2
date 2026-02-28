from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Model
from keras.layers import Input, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
import gc
import os
import sys

# imports utilitários
from src.utils.helpers import ensure_dir, normalize_columns, add_lag_features
from src.utils.reproducibility import set_global_seed
from src.models.evaluate import evaluate

# imports de cenário são opcionais
from src.features.scenarios import apply_scenario

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# adiciona raiz do repositório ao sys.path para imports absolutos
file_path = Path(__file__).resolve()
root = file_path.parents[2]
if str(root) not in sys.path:
    sys.path.append(str(root))

WINDOW = 14      # Janela de dias anteriores para predição
BATCH_SIZE = 32  # Amostras por lote para controle de memória no Samsung Book 2
EPOCHS = 50      # Máximo de iterações de treino
PATIENCE = 6     # Tolerância para parada antecipada caso o erro não diminua

# características básicas que valem para todos os modelos
FEATURES_BASE = ['onpromotion', 'unitvalue', 'holiday', 'month', 'day_of_week', 'is_weekend']

# caminhos base usando a raiz do projeto
file_path = Path(__file__).resolve()
root = file_path.parents[2]
INPUT_BASE = root / "Dados" / "preprocessed"
OUTPUT_BASE = ensure_dir(root / "Resultados" / "gru")

TARGET = 'quantity'



def build_gru_model(n_products, n_features, window):
    input_ts = Input(shape=(window, n_features))
    input_prod = Input(shape=(1,))

    x = GRU(64, return_sequences=True)(input_ts)
    x = Dropout(0.2)(x)
    x = GRU(32)(x)

    emb = tf.keras.layers.Embedding(input_dim=n_products, output_dim=16)(input_prod)
    emb = tf.keras.layers.Flatten()(emb)

    x = tf.keras.layers.Concatenate()([x, emb])
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='softplus')(x)

    model = Model([input_ts, input_prod], output)
    model.compile(optimizer='adam', loss='poisson')
    return model


def run_gru(X_train, y_train, id_train, X_val, y_val, id_val, X_test, y_test, id_test):
    """
    Ajusta o modelo único usando embeddings de produto.
    As entradas X_* já devem vir escalonadas e os ids preparados.
    """
    n_products = int(max(id_train.max(), id_val.max(), id_test.max()) + 1)
    model = build_gru_model(n_products, X_train.shape[-1], WINDOW)

    early_stop = EarlyStopping(
        monitor='val_loss', patience=PATIENCE, restore_best_weights=True
    )

    model.fit(
        [X_train, id_train], y_train,
        validation_data=([X_val, id_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    preds = model.predict([X_test, id_test]).flatten()
    return preds

def create_sequences(data, features, target, window ):
    X, y, p = [], [], []

    for prod_id, g in data.groupby('product_id'):
        values_x = g[features].values
        values_y = g[target].values

        for i in range(len(g) - window):
            X.append(values_x[i:i+window])
            y.append(values_y[i+window])
            p.append(prod_id)

    return np.array(X), np.array(y), np.array(p)

def process_file(csv_file, scenario=None):
    # carrega e aplica cenário caso exista
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df = normalize_columns(df)
    # elimina linhas com alvo ausente antes de qualquer transformação
    df = df.dropna(subset=[TARGET])

    if scenario is not None:
        try:
            df = apply_scenario(df, scenario)
        except Exception as e:
            print(f"Aviso: falha ao aplicar cenário {scenario} -> {e}")

    le = LabelEncoder()
    df['product_id'] = le.fit_transform(df['product_id'])

    # geração de features adicionais (lags e médias móveis)
    df = add_lag_features(df, TARGET)
    df = df.dropna()

    # lista final de características utilizadas
    features = FEATURES_BASE + ['lag_1','lag_7','rolling_mean_3','rolling_mean_7','rolling_mean_14']

    df = df.sort_values(['product_id', 'date'])

    # separa conjuntos por produto mas concatena para treinamento global
    train_list, val_list, test_list = [], [], []
    for pid, g in df.groupby('product_id'):
        n = len(g)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_list.append(g.iloc[:train_end].copy())
        val_list.append(g.iloc[train_end:val_end].copy())
        test_list.append(g.iloc[val_end:].copy())

    if not train_list or not test_list:
        return pd.DataFrame()

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    # garantias de tamanho mínimo
    if len(train_df) < 1000 or len(test_df) < 300:
        return pd.DataFrame()

    # transformação dos alvos e features
    for subset in (train_df, val_df, test_df):
        subset[TARGET] = np.log1p(subset[TARGET].clip(lower=0))
        subset['unitvalue'] = np.log1p(subset['unitvalue'].clip(lower=0))

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_df.loc[:, features] = scaler_x.fit_transform(train_df[features])
    train_df[[TARGET]] = scaler_y.fit_transform(train_df[[TARGET]])

    val_df.loc[:, features] = scaler_x.transform(val_df[features])
    val_df[[TARGET]] = scaler_y.transform(val_df[[TARGET]])

    test_df.loc[:, features] = scaler_x.transform(test_df[features])
    test_df[[TARGET]] = scaler_y.transform(test_df[[TARGET]])

    # sequências vetorizadas para todos os produtos
    X_train, y_train, id_train = create_sequences(train_df, features, TARGET, WINDOW)
    X_val, y_val, id_val = create_sequences(val_df, features, TARGET, WINDOW)
    X_test, y_test, id_test = create_sequences(test_df, features, TARGET, WINDOW)

    # parâmetros mínimos de amostra
    if len(X_train) < 1000 or len(X_test) < 300:
        return pd.DataFrame()

    preds = run_gru(X_train, y_train, id_train, X_val, y_val, id_val, X_test, y_test, id_test)

    # decodifica métricas
    y_test_inv = np.expm1(scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten())
    preds_inv = np.expm1(scaler_y.inverse_transform(preds.reshape(-1,1)).flatten())
    metrics = evaluate(y_test_inv, preds_inv)

    # apenas retorna as métricas globais para o arquivo
    return pd.DataFrame([{
        "model": "gru",
        "arquivo": csv_file.name,
        "mae": metrics["MAE"],
        "rmse": metrics["RMSE"],
        "smape": metrics["sMAPE"]
    }])

# variável de conveniência agora fornecida por reproducibility
# função set_global_seed já importa de src.utils.reproducibility


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=None,
                        help="Cenário a ser aplicado (volume, price, kmeans)")
    args = parser.parse_args()

    set_global_seed(args.seed)

    # o loop de cenários é gerenciado por main.py; aqui tratamos apenas o solicitado
    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir():
            continue

        market_name = market_path.name
        print(f"\nRodando GRU em: {market_name}")

        all_results = []

        for csv_file in market_path.glob("cat*.csv"):
            df_res = process_file(csv_file, scenario=args.scenario)

            if not df_res.empty:
                all_results.append(df_res)

        if all_results:
            final = pd.concat(all_results, ignore_index=True)

            suffix = f"_{args.scenario}" if args.scenario else ""
            out_file = OUTPUT_BASE / f"{market_name}_gru{suffix}.csv"
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")

            mean_smap = final['smape'].mean()
            print(f"FINAL sMAPE: {mean_smap:.4f}")

if __name__ == "__main__":
    main()
