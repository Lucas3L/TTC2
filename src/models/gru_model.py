from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf
import gc
import sys

# Garante import correto
file_path = Path(__file__).resolve()
root = file_path.parents[2]
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.models.evaluate import evaluate


WINDOW = 14      # Janela de dias anteriores para predição
BATCH_SIZE = 32  # Amostras por lote para controle de memória no Samsung Book 2
EPOCHS = 50      # Máximo de iterações de treino
PATIENCE = 6     # Tolerância para parada antecipada caso o erro não diminua

INPUT_BASE = Path("Dados/preprocessed")
OUTPUT_BASE = Path("Resultados/gru")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


TARGET = 'quantity' # Variável dependente

# Variáveis independentes
FEATURES = [
     'onpromotion', 'unitvalue',
     'holiday', 'month', 'day_of_week', 'is_weekend'
]


def create_sequences_by_product(df, features, target, window):

    Xs, ys = [], []

    # Agrupamento para processar cada produto de forma isolada
    for pid, group in df.groupby('product_id'):
        group = group.sort_values('date') # Garante a ordem cronológica do histórico

        X = group[features].values
        y = group[target].values

        # Desliza a janela criando amostras 3D 
        for i in range(len(X) - window):
            Xs.append(X[i:i+window])   # Histórico da janela
            ys.append(y[i+window])     # Alvo a ser previsto 

    return np.array(Xs), np.array(ys)


def run_gru(train_df, val_df, test_df):

    scaler_x = MinMaxScaler() # Normalizador para escala entre 0 e 1
    scaler_y = MinMaxScaler()

    # Ajuste do escalonador no treino e aplicação no teste para evitar vazamento
    train_df[FEATURES] = scaler_x.fit_transform(train_df[FEATURES])
    train_df[[TARGET]] = scaler_y.fit_transform(train_df[[TARGET]])
    
    val_df[FEATURES] = scaler_x.transform(val_df[FEATURES])
    val_df[[TARGET]] = scaler_y.transform(val_df[[TARGET]])

    test_df[FEATURES] = scaler_x.transform(test_df[FEATURES])
    test_df[[TARGET]] = scaler_y.transform(test_df[[TARGET]])

    # Geração das sequências temporais respeitando o isolamento por ID
    X_train, y_train = create_sequences_by_product(
        train_df, FEATURES, TARGET, WINDOW
    )

    val_all = pd.concat([train_df, val_df])
    test_all = pd.concat([val_df, test_df])

    X_val, y_val = create_sequences_with_context(
        val_all, WINDOW, FEATURES, TARGET
    )

    X_test, y_test = create_sequences_with_context(
        test_all, WINDOW , FEATURES, TARGET
    )

    # Definição da arquitetura da rede neural recorrente
    model = Sequential([
        GRU(32, input_shape=(WINDOW, len(FEATURES))), # Camada GRU com 32 neurônios
        Dense(1) # Camada de saída para regressão 
    ])

    # Compilação com otimizador Adam e função de perda MSE
    model.compile(
        optimizer='adam',
        loss='mse'
    )

    # Critério de parada antecipada monitorando a perda na validação
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True # Recupera o melhor estado do modelo
    )

    # Processo de ajuste de pesos 
    model.fit(
        X_train, y_train,      # Sequências de treino e alvos reais
        validation_data=(X_val, y_val),# Reserva treino para validação interna
        epochs=EPOCHS,                 # Número de passagens completas pelos dados
        batch_size=BATCH_SIZE,         # Quantidade de dados por atualização de pesos
        callbacks=[early_stop],        # Aciona o EarlyStopping se necessário
        verbose=1                      # Exibe o progresso do erro no terminal
    )

    # Execução das predições sobre os dados de teste janelados
    preds = model.predict(X_test).flatten()

    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    preds_inv  = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

    # Cálculo das métricas comparativas através do módulo evaluate
    metrics = evaluate(y_test_inv, preds_inv)
    print("\nResultados do Modelo:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print(f"FINAL sMAPE: {metrics['sMAPE']:.4f}")

    return preds, metrics

def create_sequences_with_context(df_all, window, features, target):

    X, y = [], []

    for pid, group in df_all.groupby('product_id'):
        group = group.sort_values('date')

        X_vals = group[features].astype(float).values
        y_vals = group[target].astype(float).values

        for i in range(len(X_vals) - window):
            X.append(X_vals[i:i+window])
            y.append(y_vals[i+window])

    return np.array(X), np.array(y)


def process_file(csv_file):

    df = pd.read_csv(csv_file, parse_dates=['Date'])

    df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(' ', '_')
    )   

    df = df.sort_values([ 'date'])

    n = len(df)

    train_end = int(n * 0.70)
    val_end  = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    print(f"Tamanho total: {n} | Treino: {len(train_df)} | Val: {len(val_df)} | Teste: {len(test_df)}")


    if len(train_df) < WINDOW * 5:
        return None

    return run_gru(train_df, val_df, test_df)

def main():

    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir():
            continue

        market_name = market_path.name
        print(f"\nRodando GRU em: {market_name}")

        all_results = []

        for csv_file in market_path.glob("cat*.csv"):
            result = process_file(csv_file)

            if result:
                preds, metrics = result
                metrics["arquivo"] = csv_file.name
                all_results.append(metrics)

        if all_results:
            final = pd.DataFrame(all_results)
            out_file = OUTPUT_BASE / f"{market_name}_gru.csv"
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")


if __name__ == "__main__":
    main()
