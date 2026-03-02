import os
import random
import numpy as np
import tensorflow as tf
import subprocess
import time
import csv
import re
from pathlib import Path
from datetime import datetime
import argparse
import math

# --- CONFIGURAÇÕES GLOBAIS ---
RANDOM_SEED = 42
N_REPLICAS = 3
SCENARIOS = ["volume", "price", "kmeans"]

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

# Import utilitário para garantir diretórios
try:
    from src.utils.helpers import ensure_dir
except ImportError:
    # Fallback caso o helper não esteja acessível
    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

OUTPUT_DIR = ensure_dir(BASE_DIR / "Resultados")
ERROR_LOG = OUTPUT_DIR / "errors.log"

# Dicionário de mapeamento dos scripts
MODELS = {
    "LSTM": SRC_DIR / "models" / "lstm_model.py",
    "GRU": SRC_DIR / "models" / "gru_model.py",
    "XGBoost": SRC_DIR / "models" / "xgboost_model.py",
    "Baseline": SRC_DIR / "models" / "baseline.py"
}

def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def extract_metrics(output: str):
    """
    Extração universal via Regex. 
    Busca padrões como 'FINAL sMAPE: 10.5' ou 'MAE: 2.3'
    """
    metrics = {}
    # aceita formatos como 12.34, 1e-03, -2.5E+02 ou nan
    float_re = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|NaN)"
    patterns = {
        "smape": rf"FINAL sMAPE:\s*{float_re}",
        "mae": rf"MAE:\s*{float_re}",
        "rmse": rf"RMSE:\s*{float_re}"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if not match:
            metrics[key] = None
            continue

        val_str = match.group(1)
        try:
            val = float(val_str)
            # normalize NaN to None for clearer downstream handling
            metrics[key] = None if math.isnan(val) else val
        except Exception:
            metrics[key] = None
    return metrics

def run_model(model_name, script_path, seed, replica_id, scenario):
    # 1. Verificação de Paths e Arquivos
    if not script_path.exists():
        msg = f"[ERRO] Script não encontrado: {script_path}"
        print(msg)
        with open(ERROR_LOG, "a") as f: 
            f.write(f"{datetime.now()} | {msg}\n")
        return None, 0, "MISSING"

    print(f"\n>>> {model_name} | Cenário: {scenario} | Réplica: {replica_id} | Seed: {seed}")

    start_time = time.time()

    # 2. Execução via Subprocess
    # Nota: Passamos as flags que todos os seus scripts agora aceitam
    process = subprocess.run(
        ["python", str(script_path), "--seed", str(seed), "--scenario", str(scenario)],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR)
    )

    runtime = time.time() - start_time
    output = process.stdout + process.stderr
    status = "OK" if process.returncode == 0 else "ERRO"

    # 3. Extração de Métricas Universal
    results = extract_metrics(output)

    if status == "ERRO":
        print(f" [!] Falha na execução do {model_name}. Verifique o error.log")
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now()}] {model_name} | {scenario} | replica {replica_id} | SEED {seed}\n")
            f.write(output)
            f.write("\n" + "="*60 + "\n")
    else:
        print(f" [+] Finalizado em {runtime:.2f}s | sMAPE: {results.get('smape')}")

    return results, runtime, status

def log_result(model_name, replica_id, seed, metrics, runtime, status, scenario):
    scenario_file = OUTPUT_DIR / f"{scenario}_results.csv"
    file_exists = scenario_file.exists()

    with open(scenario_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "replica", "seed", "smape", "mae", "rmse", "runtime_sec", "status"])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            replica_id,
            seed,
            metrics.get("smape"),
            metrics.get("mae"),
            metrics.get("rmse"),
            round(runtime, 3),
            status
        ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_global_seed(args.seed)

    print(f"\n================ INICIANDO EXPERIMENTOS ({datetime.now().year}) ================")
    
    for replica_id in range(1, N_REPLICAS + 1):
        # Semente variando por réplica para garantir robustez estatística
        current_seed = RANDOM_SEED + (replica_id * 100)
        
        for scenario in SCENARIOS:
            for model_name, script_path in MODELS.items():
                
                # Executa o modelo
                metrics, runtime, status = run_model(
                    model_name, script_path, current_seed, replica_id, scenario
                )

                # 4. Gestão de falhas: Se o script falhar, logamos mas o main continua
                if metrics is not None:
                    log_result(
                        model_name, replica_id, current_seed, 
                        metrics, runtime, status, scenario
                    )
                
                # Controle de memória preventivo (opcional se scripts filhos já fazem gc)
                tf.keras.backend.clear_session()

    print(f"\n================ EXPERIMENTOS FINALIZADOS ================")
    print(f"Logs de erro (se houver): {ERROR_LOG}")
    print(f"Arquivos de resultados gerados em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()