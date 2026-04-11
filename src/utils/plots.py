import os
import glob
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# --- Configurações de Diretório ---
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_PATH = BASE_DIR / "Resultados"
OUTPUT_DIR = RESULTS_PATH / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _get_plt():
    """Tenta importar matplotlib de forma segura para ambientes sem interface gráfica."""
    try:
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        return plt
    except ModuleNotFoundError:
        print("[AVISO] Matplotlib não encontrado. Gráficos não serão gerados.")
        return None

def plot_quantidade_por_tempo(predictions, output_path: str = None, title_suffix: str = ""):
    """
    Plota série temporal Real vs Predito com métrica de erro no título.
    Agrupa os dados por data (soma de todos os produtos do arquivo).
    """
    plt = _get_plt()
    if plt is None:
        return

    # Carregamento flexível dos dados
    if isinstance(predictions, (str, Path)):
        df = pd.read_csv(predictions)
    else:
        df = predictions.copy()

    if 'date' not in df.columns or 'y_true' not in df.columns or 'y_pred' not in df.columns:
        print(f"[ERRO] Colunas necessárias ausentes em {predictions}")
        return

    df['date'] = pd.to_datetime(df['date'])

    # Cálculo do sMAPE global do arquivo para o título
    y_t = df['y_true'].values
    y_p = df['y_pred'].values
    denom = (np.abs(y_t) + np.abs(y_p)) / 2.0
    smape_val = 100 * np.mean(np.abs(y_t - y_p) / np.where(denom == 0, 1, denom))

    # Agrupamento por data para visualização da tendência
    agrupado = df.groupby('date').agg({'y_true': 'sum', 'y_pred': 'sum'}).reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(agrupado['date'], agrupado['y_true'], label='Real', color='#2c3e50', linewidth=2)
    plt.plot(agrupado['date'], agrupado['y_pred'], label='Predito', color='#e67e22', linestyle='--', linewidth=2)
    
    plt.title(f'Performance Global: {title_suffix}\n(sMAPE Médio: {smape_val:.2f}%)', fontsize=14)
    plt.xlabel('Data')
    plt.ylabel('Quantidade Total')
    plt.legend()
    plt.tight_layout()
    
    final_path = output_path or "quantidade_por_tempo.png"
    plt.savefig(final_path, dpi=300)
    plt.close()
    print(f"[+] Gráfico salvo em: {final_path}")

def gerar_graficos_todos_predictions():
    """Varre a pasta Resultados e gera gráficos para todos os arquivos de predição encontrados."""
    padrao = str(RESULTS_PATH / "**" / "*_predictions.csv")
    arquivos = glob.glob(padrao, recursive=True)

    if not arquivos:
        print(f"[-] Nenhum arquivo *_predictions.csv encontrado em {RESULTS_PATH}")
        return

    for arq in arquivos:
        nome_base = Path(arq).stem
        out_path = OUTPUT_DIR / f"{nome_base}.png"
        
        # Extrai o nome do modelo/cenário do nome do arquivo para o título
        label = nome_base.replace("_predictions", "").replace("_", " ").upper()
        
        try:
            plot_quantidade_por_tempo(arq, output_path=str(out_path), title_suffix=label)
        except Exception as e:
            print(f"[ERRO] Falha ao processar {nome_base}: {e}")

if __name__ == "__main__":
    print(f"[DEBUG] Iniciando geração de gráficos em: {OUTPUT_DIR}")
    gerar_graficos_todos_predictions()