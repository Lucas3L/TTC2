from pathlib import Path
import pandas as pd




# Define o diretório raiz onde os CSVs de cada modelo foram salvos
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_PATH = BASE_DIR / "Resultados"




def load_model_results(model_name):
    # Localiza arquivos CSV recursivamente que terminam com o sufixo do modelo
    files = RESULTS_PATH.glob(f"**/*_{model_name}.csv")
    dfs = []

    for f in files:
        # Carrega os dados de métricas (mae, rmse, smape) salvos anteriormente
        df = pd.read_csv(f)

        #Padroniza nomes das colunas
        df.columns = df.columns.str.lower()

        # Garante que todas as colunas existam
        if "arquivo" not in df.columns:
            df["arquivo"] = None

        # Extrai o nome do mercado do arquivo para identificação na tabela final
        df["market"] = f.stem.replace(f"_{model_name}", "")
        df["model"] = model_name

        dfs.append(df[["model", "market", "arquivo", "mae", "rmse", "smape"]])

    # Cláusula de guarda para evitar erro de concatenação caso não encontre arquivos
    if not dfs:
        return pd.DataFrame(columns=["model", "market", "arquivo", "mae", "rmse", "smape"])

    # Une todos os mercados de um mesmo modelo em um único DataFrame
    return pd.concat(dfs, ignore_index=True)




def aggregate_all_models():
    # Lista dos modelos que compõem o escopo do trabalho na Dez Telecom
    models = ["baseline", "xgboost", "lstm", "gru"]
    all_data = []

    for model in models:
        # Carrega os resultados consolidados por mercado para cada algoritmo
        df = load_model_results(model)
        if not df.empty:
            # Identifica a origem do dado antes da junção global
            all_data.append(df)

    # Consolida todos os modelos em uma base única para análise estatística
    final = pd.concat(all_data, ignore_index=True)

    # Persistência do resultado final em disco para uso no Excel ou Power BI do TCC
    out = RESULTS_PATH / "consolidated_results.csv"
    final.to_csv(out, index=False)

    print(f"\n Resultados consolidados salvos em: {out}")
    return final




if __name__ == "__main__":
    # Ponto de entrada para execução do script de integração
    aggregate_all_models()