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
INPUT_BASE = Path("Dados/preprocessed")
OUTPUT_BASE = Path("Resultados/xgboost")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Array para guardar
TARGET = "quantity"
WINDOW = 14
FEATURES = [
     'onpromotion', 'unitvalue',
     'holiday', 'month', 'day_of_week', 'is_weekend'
]




def create_lag_features_by_product(df, features, target, window):

    Xs, ys = [], []

    for _, group in df.groupby('product_id'):
        group = group.sort_values('date')

        X = group[features].values
        y = group[target].values

        for i in range(len(X) - window):
            Xs.append(X[i:i+window].flatten())  # transforma em vetor 1D
            ys.append(y[i+window])

    return np.array(Xs), np.array(ys)




# Função que garante a integridade dos dados
def process_file(csv_file):

    # conversão para o tipo datime
    df = pd.read_csv(csv_file, parse_dates=["Date"])

    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(' ', '_')
    )

    # Ordenação dos dados 
    df = df.sort_values(["date"])
    # Remove as linha que possuem valores nulos
    df = df.dropna(subset=FEATURES + [TARGET])

    n = len(df)
    train_end = int(n * 0.70)
    val_end  = int(n * 0.85)

    # Treino validação e Testes
    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    # Define o numero maximo de treino e testes minimos
    if len(train_df) < 100 or len(test_df) < 20:
        return pd.DataFrame()
    
    # Faz o mapeamento para minimizar funcao de perda
    X_train, y_train = create_lag_features_by_product(
        train_df, FEATURES, TARGET, WINDOW                                        
    )

    # Permite monitoramenteo de dados não vistos durante interação de treino 
    val_all = pd.concat([train_df, val_df])

    # Geração das metricas de avaliação
    test_all = pd.concat([val_df, test_df])

    X_val, y_val =create_lag_features_by_product(
        val_all, FEATURES, TARGET, WINDOW
    )

    X_test, y_test = create_lag_features_by_product(
    test_all, FEATURES, TARGET, WINDOW
)

    if len(X_train) < 1000 or len(X_val) < 300 or len(X_test) < 300:
        return pd.DataFrame()

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
        verbose=False # Exibição de erro desativada
    )

    # Cria uma copia e uma nova coluna com as previsoes ao lado dos reais
    preds = model.predict(X_test)

    metrics = evaluate(y_test, preds)

    print(
    f"FINAL -> MAE: {metrics['MAE']:.4f} | "
    f"RMSE: {metrics['RMSE']:.4f} | "
    f"FINAL sMAPE: {metrics['sMAPE']:.4f}"
    )

    return pd.DataFrame([metrics])





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
            if not df_res.empty:
                df_res["arquivo"] = csv_file.name
                all_results.append(df_res)

        # Verificação se não há listas vazias, se nenhum arquivo cat.csv passou pelos filtros
        # Junta os dados e ignora indices
        # Criação de caminho
        # Salva esse Dataframe em arquivos .csv 
        if all_results:
            final = pd.concat(all_results, ignore_index=True)
            final = final[["arquivo", "MAE", "RMSE", "sMAPE"]]
            out_file = OUTPUT_BASE / f"{market_name}_xgboost.csv"
            final.to_csv(out_file, index=False)
            print(f"  Resultados salvos em {out_file}")

if __name__ == "__main__":
    main()
