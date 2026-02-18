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

INPUT_BASE = Path("Dados/features")
OUTPUT_BASE = Path("Resultados/gru")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


TARGET = 'sales' # Variável dependente

# Variáveis independentes
FEATURES = [
    'price', 'on_promotion', 'dayofweek',
    'weekofyear', 'month', 'lag_1', 'lag_7'
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


def run_gru(train_df, test_df):

    scaler_X = MinMaxScaler() # Normalizador para escala entre 0 e 1

    # Ajuste do escalonador no treino e aplicação no teste para evitar vazamento
    X_train = scaler_X.fit_transform(train_df[FEATURES])
    X_test = scaler_X.transform(test_df[FEATURES])

    # Cópias para preservar os dados originais e aplicar os valores escalados
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()

    train_df_scaled[FEATURES] = X_train
    test_df_scaled[FEATURES] = X_test

    # Geração das sequências temporais respeitando o isolamento por ID
    X_train_seq, y_train_seq = create_sequences_by_product(
        train_df_scaled, FEATURES, TARGET, WINDOW
    )

    X_test_seq, y_test_seq = create_sequences_by_product(
        test_df_scaled, FEATURES, TARGET, WINDOW
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
        X_train_seq, y_train_seq,      # Sequências de treino e alvos reais
        validation_split=0.1,          # Reserva 10% do treino para validação interna
        epochs=EPOCHS,                 # Número de passagens completas pelos dados
        batch_size=BATCH_SIZE,         # Quantidade de dados por atualização de pesos
        callbacks=[early_stop],        # Aciona o EarlyStopping se necessário
        verbose=1                      # Exibe o progresso do erro no terminal
    )

    # Execução das predições sobre os dados de teste janelados
    preds = model.predict(X_test_seq).ravel()

    # Cálculo das métricas comparativas através do módulo evaluate
    metrics = evaluate(y_test_seq, preds, model_name="GRU")
    print("\nResultados do Modelo:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print(f"FINAL sMAPE: {metrics['smape']:.4f}")

    return preds, metrics["smape"]

def process_file(csv_file):

    df = pd.read_csv(csv_file, parse_dates=['Date'])

    df = df.sort_values(['product_id', 'Date'])

    train = df[df['split'] == 'train']
    test  = df[df['split'] == 'test']

    if len(train) < 200 or len(test) < 30:
        return None

    return run_gru(train, test)

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
