import subprocess
import time
import csv
import re
from pathlib import Path
from datetime import datetime


# Semente base para garantir que os experimentos sejam replicáveis
RANDOM_SEED = 42
# Número de execuções por modelo para permitir análise de desvio padrão
N_REPLICAS = 3

# Gestão de caminhos robusta utilizando Pathlib para evitar erros de diretório
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

# Define e garante a existência da pasta de resultados
LOG_DIR = BASE_DIR / "results"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "experimental_results.csv"

# Dicionário de mapeamento dos scripts dos modelos 
MODELS = {
    "LSTM": SRC_DIR / "models" / "lstm_model.py",
    "GRU": SRC_DIR / "models" / "gru_model.py",
    "XGBoost": SRC_DIR / "models" / "xgboost_model.py"
}


def extract_smape(output: str):
    """
    Utiliza Regex para localizar o valor do sMAPE no console do script filho.
    Exemplo esperado no print do modelo: 'FINAL sMAPE: 12.34'
    """
    match = re.search(r"FINAL sMAPE:\s*([\d\.]+)", output)
    if match:
        return float(match.group(1))
    return None


def run_model(model_name, script_path, seed, replica_id):
    """
    Dispara a execução do modelo em um processo isolado via subprocess.
    """
    print(f"\n Executando {model_name} | Réplica {replica_id} | Seed {seed}")

    start_time = time.time()

    # Executa o comando 'python script.py --seed X' e captura a saída
    process = subprocess.run(
        ["python", str(script_path), "--seed", str(seed)],
        capture_output=True,
        text=True
    )

    # Calcula o tempo total de processamento do modelo
    runtime = time.time() - start_time
    # Une a saída padrão e de erro para facilitar o debug caso falhe
    output = process.stdout + process.stderr

    # Verifica se o processo terminou sem erros 
    status = "OK" if process.returncode == 0 else "ERRO"

    smape = extract_smape(output)

    if status == "ERRO":
        print(f" Erro ao executar {model_name}")
        print(output)
    else:
        print(f" {model_name} finalizado em {runtime:.2f}s | sMAPE: {smape}")

    return smape, runtime, status, output


def init_csv():
    """
    Inicializa o arquivo de log com o cabeçalho científico.
    """
    if not LOG_FILE.exists():
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "model",
                "replica",
                "seed",
                "smape",
                "runtime_sec",
                "status"
            ])


def log_result(model_name, replica_id, seed, smape, runtime, status):
    """
    Persiste os dados da rodada no CSV para posterior análise estatística.
    """
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            replica_id,
            seed,
            smape,
            round(runtime, 3),
            status
        ])


def main():
    print("\n================ INICIANDO EXPERIMENTOS =================\n")

    init_csv()

    # Loop de réplicas para garantir significância estatística dos resultados
    for replica_id in range(1, N_REPLICAS + 1):
        # Altera a semente a cada réplica para explorar diferentes inicializações
        current_seed = RANDOM_SEED + replica_id

        print(f"\n========== RÉPLICA {replica_id} | SEED {current_seed} ==========\n")

        # Itera sobre os modelos definidos para comparação
        for model_name, script_path in MODELS.items():
            smape, runtime, status, output = run_model(
                model_name,
                script_path,
                current_seed,
                replica_id
            )

            # Grava o resultado no repositório central de métricas
            log_result(
                model_name,
                replica_id,
                current_seed,
                smape,
                runtime,
                status
            )

    print("\n================ EXPERIMENTOS FINALIZADOS ================\n")
    print(f" Resultados salvos em: {LOG_FILE}")


if __name__ == "__main__":
    main()