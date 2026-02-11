from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import sys

file_path = Path(__file__).resolve()
root = file_path.parents[2]
if str(root) not in sys.path:
    sys.path.append(str(root))

try:
    from src.models.evaluate import evaluate
except ImportError:
    from evaluate import evaluate

# Caminhos de entrada e saida dos dados 
INPUT_BASE = Path("Dados/features")
OUTPUT_BASE = Path("Resultados/xgboost")

# Cria caso não existir
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Array para guardar a quantidade dos itens
TARGET = "Quantity"

# historico de lags de aconrdo com os parametros
FEATURES = [
    "year", "month", "week", "day", "dayofweek", "is_weekend",
    "Quantity_lag_1", "Quantity_lag_7", "Quantity_lag_14"
]

# Função que garante a integridade dos dados
def process_file(csv_file):

    # conversão para o tipo datime
    df = pd.read_csv(csv_file, parse_dates=["Date"])

    # Armazena antes de qualquer filtro 
    initial_products = df["product_id"].nunique()

    # Ordenação dos dados 
    df = df.sort_values(["product_id", "Date"])

    # Remove as linha que possuem valores nulos
    df = df.dropna(subset=FEATURES + [TARGET])

    # Treino validação e Testes
    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]

    # Define o numero maximo de treino e testes minimos
    if len(train) < 100 or len(test) < 20:
        return pd.DataFrame()
    
    # Valida se de fato o produto possui historico suficiente para ser avalidado
    valid_products_df = df.groupby("product_id").filter(
        lambda x: len(x[x["split"] == "train"]) >= 100 and len(x[x["split"] == "test"]) >= 20
    )
    
    # Classe de produtos aceitos para o trabalho e tambem os dados não aceitos
    final_products = valid_products_df["product_id"].nunique()
    dropped = initial_products - final_products

    # Mostra os descartes
    if dropped > 0:
        print(f"  {csv_file.name}: {dropped} produtos descartados por dados insuficientes (Restaram: {final_products})")

    # Garante que não havera dados nulos
    if final_products == 0:
        return pd.DataFrame()
    
    # Faz o mapeamento para minimizar funcao de perda
    X_train = train[FEATURES]
    y_train = train[TARGET]

    # Permite monitoramenteo de dados não vistos durante interação de treino 
    X_val = val[FEATURES]
    y_val = val[TARGET]

    # Geração das metricas de avaliação
    X_test = test[FEATURES]
    y_test = test[TARGET]

    model = xgb.XGBRegressor(
        n_estimators=1000, # n de arvore maximos
        learning_rate=0.03, # valor corrigido por cada arvore
        max_depth=8, # Profundidade de cada arvore
        subsample=0.8, # Aleatoriedade para prever vicio do modelo
        colsample_bytree=0.8, # fração para construção de cada coluna
        random_state=42, # Persistencia e reprodutividade dos dados
        n_jobs=-1 # nucleos usados
     )

    model.fit(
        X_train, y_train, # Busca por padroes matematicos
        eval_set=[(X_val, y_val)], # Defini validação para o algoritimo testar
        early_stopping_rounds=50, # parametro de parada caso não diminuir validação
        verbose=False # Exibição de erro desativada
    )

    # Cria uma copia e uma nova coluna com as previsoes ao lado dos reais
    test = test.copy()
    test["pred"] = model.predict(X_test)

    results = (
        test
        # Separa o conjunto de testes para que cada 
        # erro de cada produto seja calculado independente 
        .groupby("product_id") 
        # Executa função para receber a serie de valores reais dos produtos
        # Alem de organizar o retorno da função para facilitar posteriormente
        .apply(lambda x: pd.Series(
            evaluate(x[TARGET], x["pred"]),
            index=["mae", "rmse", "smape"]
        ))
        # Transforma o id de volta em coluna para ser salvo em csv
        .reset_index()
    )

    return results


def main():
    # Percorre a lista de todos os arquivos da pasta, 
    # Verificação de arquivos com outros formatos
    for market_path in INPUT_BASE.iterdir():
        if not market_path.is_dir():
            continue

        # retira o nome da pasta para ser utilizada e imprime
        market_name = market_path.name
        print(f"\nRodando XGBoost em: {market_name}")

        # Array para salvar todos os resultados
        all_results = []

        # Procura por arquivos que sejam csv e iniciem com cat,
        # Para cada arquivo será carregado, processado e retornado
        for csv_file in market_path.glob("cat*.csv"):
            df_res = process_file(csv_file)
            all_results.append(df_res)

        # Verificação se não há listas vazias, se nenhum arquivo cat.csv passou pelos filtros
        # Junta os dados e ignora indices
        # Criação de caminho
        # Salva esse Dataframe em arquivos .csv 
        if all_results:
            final = pd.concat(all_results, ignore_index=True)
            out_file = OUTPUT_BASE / f"{market_name}_xgboost.csv"
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")


if __name__ == "__main__":
    main()
