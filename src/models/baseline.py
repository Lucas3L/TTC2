from pathlib import Path
import pandas as pd
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Definição dos caminhos
INPUT_BASE = Path("Dados/split")
OUTPUT_BASE = Path("Resultados/baseline")

# Verificação e Criação de Pastas
if not INPUT_BASE.exists():
    print(f"Erro: A pasta de entrada {INPUT_BASE} não existe!")
    sys.exit()
else:
    print(f"Pasta de entrada encontrada: {INPUT_BASE}")

# Cria a pasta de saída (caso não exista)
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
print(f"Pasta de saída criada {OUTPUT_BASE}")

# Guarda os valores de quantidade para serem usados
TARGET = "Quantity"

# função que compara os valores reais e preditos 
def evaluate(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Media dos erros entre reais e predição
    mae = mean_absolute_error(y_true, y_pred)
    # Média dos erros ao quadrado entre reais e predição
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Mostra a porcentagem de erros por magnitude
    smape = (
        np.mean(
            2 * np.abs(y_true - y_pred) /
            (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        ) * 100
    )
    return mae, rmse, smape

# Função que utiliza o modelo naive, que ultiliza o ultimo valor registrado para comparar
def naive_forecast(series):
    return series.shift(1)

# Função que utiliza a media movel simples para comparar reais e predição dos ultimos 7 dias
def moving_average_forecast(series, window=7):
    return series.shift(1).rolling(window).mean()

# Copia os dados e extrai o dia da semana e adiciona outra coluna dia da semana dow
def seasonal_forecast(train, test, target):
    mean_by_dow = (
        train
        .groupby(train["Date"].dt.dayofweek)[target]
        .mean()
    )
    # para cada linha do conjunto identifica o dia da semana
    preds = test["Date"].dt.dayofweek.map(mean_by_dow)

    return preds

# Função de processamento do arquivo csv
def process_file(csv_file):
    # carrega os arquivos para a memeoria
    df = pd.read_csv(csv_file, parse_dates=["Date"])
    # Ordena pela data de forma hierarquica de acordo com a categoria
    df = df.sort_values(["market", "category", "product_id", "Date"])
    # limpa os dados nulos
    df = df.dropna(subset=[TARGET])

# inicia um array vazio para guardar os resultados
    results = []

    # Laõ que divide os dados por id de produto, 
    # separa os dados de treino, teste e validação
    for product_id, g in df.groupby("product_id"):
        train = g[g["split"] == "train"]
        val = g[g["split"] == "val"]
        test = g[g["split"] == "test"]

        # Ignora produtos sem dados de 7 a 14 dias
        if len(train) < 14 or len(test) < 7:
            continue

        # Concatena Validação e Teste para garantir que
        #  o primeiro dia de teste tenha data de ontem
        full_series = pd.concat([val[TARGET], test[TARGET]])
        naive_pred_full = naive_forecast(full_series)

        # Validação e definição apenas dos dias de teste
        naive_pred = naive_pred_full.iloc[-len(test):]

        # União de teste, treino e validação
        ma_pred = moving_average_forecast(
            pd.concat([train[TARGET], val[TARGET], test[TARGET]])
        ).iloc[-len(test):]

        # Mapeoa de acordo com os dias da semana
        seasonal_pred = seasonal_forecast(train, test, TARGET)

        # Isola valores de quantidade do periodo de testes
        y_true = test[TARGET]


        # Calcule uma unica vez por modelo
        n_mae, n_rmse, n_smape = evaluate(y_true, naive_pred)
        ma_mae, ma_rmse, ma_smape = evaluate(y_true, ma_pred)
        s_mae, s_rmse, s_smape = evaluate(y_true, seasonal_pred)

        results.append({
            # identificadores dos resultados
            "naive_mae": n_mae,
            "naive_rmse": n_rmse,
            "naive_smape": n_smape,
            
            "ma7_mae": ma_mae,
            "ma7_rmse": ma_rmse,
            "ma7_smape": ma_smape,  
            
            "seasonal_mae": s_mae,
            "seasonal_rmse": s_rmse,
            "seasonal_smape": s_smape
        })
    # Retorna os resultados dos modelos
    return pd.DataFrame(results)


def main():
    # Percore dentro das pastas Dados/split cada um dos mercados
    for market_path in INPUT_BASE.iterdir():
        # Valida se é um diretoria e não arquivos
        if not market_path.is_dir():
            continue
        # Salva o nome do mercado
        market_name = market_path.name
        print(f"\nRodando baseline em: {market_name}")

        # array para salvar os resultados
        all_results = []

        # Laço que percorre tos os arquivos com cat do csv dentro de meracado
        for csv_file in market_path.glob("cat*.csv"):
            df_res = process_file(csv_file)
            all_results.append(df_res)

        # Verifica se contem dados, se tem devem ser empilhados 
        # e zera o index para iniciar do 0 ....
        if all_results:
            final = pd.concat(all_results, ignore_index=True)

            # Construi o caminho até o local e salva os dados no diretorio  
            out_file = OUTPUT_BASE / f"{market_name}_baseline.csv"
            final.to_csv(out_file, index=False)

            print(f"  Resultados salvos em {out_file}")

# PONTO DE ENTRADA
# Proteção caso o scrip seja utilizado em outra chamada para reutilização
if __name__ == "__main__":
    main()
