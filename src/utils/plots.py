import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path



plt.style.use("seaborn-v0_8-whitegrid")

# Definição dos caminhos para leitura dos dados consolidados e salvamento das imagens
RESULTS_FILE = Path("Resultados/consolidated_results.csv")
OUTPUT_DIR = Path("Resultados/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Cria a pasta de gráficos caso não exista
COLUMN_MAP = {
    "mae": ["mae", "mae_test", "MAE"],
    "rmse": ["rmse", "rmse_test", "RMSE"],
    "smape": ["smape", "smape_test", "SMAPE"]
}



def find_column(df, metric):
    for col in COLUMN_MAP[metric]:
        if col in df.columns:
            return col
    raise ValueError(f" Nenhuma coluna válida encontrada para {metric}")





def plot_metric(metric):
    # Carrega a base consolidada contendo resultados de todos os algoritmos
    df = pd.read_csv(RESULTS_FILE)

    # Agrupa por modelo e calcula a média da métrica 
    summary = (
        df.groupby("model")[metric]
        .mean()
        .sort_values() # Ordena do melhor desempenho  para o pior
    )

    plt.figure(figsize=(8,5))
    ax = summary.plot(kind="bar") # Gera o gráfico de barras comparativo

    # Configuração da estética do gráfico para publicação acadêmica
    plt.figure(figsize=(8,5))
    plt.title(f"Comparação dos Modelos - {metric.upper()}")
    plt.ylabel(metric.upper())
    plt.xlabel("Modelo")
    plt.xticks(rotation=0) # Mantém os nomes dos modelos na horizontal para facilitar a leitura
    plt.tight_layout() # Ajusta margens automaticamente

    for i, v in enumerate(summary.values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # Salvamento do arquivo em alta resolução para impressão em monografias
    out_png = OUTPUT_DIR / f"{metric}.png"
    out_pdf = OUTPUT_DIR / f"{metric}.pdf"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close() # Fecha a figura para liberar memória do Samsung Book 2

    print(f" Gráfico salvo em: {out_png}")



def generate_all_plots():
    # Itera sobre as três métricas principais de avaliação do projeto
    for metric in ["mae", "rmse", "smape"]:
        plot_metric(metric)


if __name__ == "__main__":
    # Ponto de entrada para a geração automática da galeria de resultados
    generate_all_plots()