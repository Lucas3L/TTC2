import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Definição dos caminhos para leitura dos dados consolidados e salvamento das imagens
RESULTS_FILE = Path("Resultados/consolidated_results.csv")
OUTPUT_DIR = Path("Resultados/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Cria a pasta de gráficos caso não exista

def plot_metric(metric):
    # Carrega a base consolidada contendo resultados de todos os algoritmos
    df = pd.read_csv(RESULTS_FILE)

    # Agrupa por modelo e calcula a média da métrica 
    summary = (
        df.groupby("model")[metric]
        .mean()
        .sort_values() # Ordena do melhor desempenho  para o pior
    )

    # Configuração da estética do gráfico para publicação acadêmica
    plt.figure(figsize=(8,5))
    summary.plot(kind="bar") # Gera o gráfico de barras comparativo
    plt.title(f"Comparação dos Modelos - {metric.upper()}")
    plt.ylabel(metric.upper())
    plt.xlabel("Modelo")
    plt.xticks(rotation=0) # Mantém os nomes dos modelos na horizontal para facilitar a leitura
    plt.tight_layout() # Ajusta margens automaticamente

    # Salvamento do arquivo em alta resolução para impressão em monografias
    out = OUTPUT_DIR / f"{metric}.png"
    plt.savefig(out, dpi=300)
    plt.close() # Fecha a figura para liberar memória do Samsung Book 2

    print(f"📊 Gráfico salvo em: {out}")


def generate_all_plots():
    # Itera sobre as três métricas principais de avaliação do projeto
    for metric in ["mae", "rmse", "smape"]:
        plot_metric(metric)


if __name__ == "__main__":
    # Ponto de entrada para a geração automática da galeria de resultados
    generate_all_plots()