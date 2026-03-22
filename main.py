import tensorflow as tf
import subprocess
import sys
import time
import csv
import re
from pathlib import Path
from datetime import datetime
import argparse
import math
from src.config.experiment_config import DEFAULT_EXPERIMENT_CONFIG
from src.utils.helpers import ensure_dir
from src.utils.reproducibility import set_global_seed

# --- CONFIGURAÇÕES GLOBAIS ---
RANDOM_SEED = DEFAULT_EXPERIMENT_CONFIG["random_seed"]
N_REPLICAS = DEFAULT_EXPERIMENT_CONFIG["n_replicas"]
SCENARIOS = DEFAULT_EXPERIMENT_CONFIG["scenarios"]

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

OUTPUT_DIR = ensure_dir(BASE_DIR / "Resultados")
ERROR_LOG = OUTPUT_DIR / "errors.log"

# Dicionário de mapeamento dos scripts
MODELS = DEFAULT_EXPERIMENT_CONFIG["models"]

def _validate_date_str(date_str: str, field_name: str):
    if date_str is None:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"Valor inválido em --{field_name}: '{date_str}'. Use formato YYYY-MM-DD."
        ) from exc

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

def run_model(
    model_name,
    script_path,
    seed,
    replica_id,
    scenario,
    date_from=None,
    date_to=None,
    timeout_sec=3600,
    max_retries=0,
):
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
    # Nota: passamos as flags suportadas pelos scripts de modelo
    cmd = [sys.executable, str(script_path), "--seed", str(seed)]
    if scenario is not None:
        cmd.extend(["--scenario", str(scenario)])
    if date_from:
        cmd.extend(["--date-from", str(date_from)])
    if date_to:
        cmd.extend(["--date-to", str(date_to)])
    output = ""
    status = "ERRO"
    process = None

    for attempt in range(max_retries + 1):
        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(BASE_DIR),
                timeout=timeout_sec,
            )
            output = (process.stdout or "") + (process.stderr or "")
            status = "OK" if process.returncode == 0 else "ERRO"
            if status == "OK":
                break

            if attempt < max_retries:
                print(f" [!] Tentativa {attempt + 1} falhou para {model_name}. Reexecutando...")
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            status = "TIMEOUT"
            if attempt < max_retries:
                print(f" [!] Timeout na tentativa {attempt + 1} para {model_name}. Reexecutando...")
            else:
                break

    runtime = time.time() - start_time

    # 3. Extração de Métricas Universal
    results = extract_metrics(output)

    if status != "OK":
        print(f" [!] Falha na execução do {model_name}. Verifique o error.log")
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            return_code = None if process is None else process.returncode
            f.write(
                f"\n[{datetime.now()}] {model_name} | {scenario} | replica {replica_id} | "
                f"SEED {seed} | status={status} | returncode={return_code}\n"
            )
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
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--date-from", type=str, default=DEFAULT_EXPERIMENT_CONFIG.get("date_from"))
    parser.add_argument("--date-to", type=str, default=DEFAULT_EXPERIMENT_CONFIG.get("date_to"))
    parser.add_argument(
        "--model-timeout-sec",
        type=int,
        default=3600,
        help="Timeout por execução de modelo em segundos.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Número de tentativas extras para cada execução de modelo (0 = sem retry).",
    )
    args = parser.parse_args()

    date_from = _validate_date_str(args.date_from, "date-from")
    date_to = _validate_date_str(args.date_to, "date-to")
    if date_from and date_to and date_from > date_to:
        raise ValueError(
            f"Intervalo inválido: --date-from ({date_from}) é maior que --date-to ({date_to})."
        )

    set_global_seed(args.seed)

    print(f"\n================ INICIANDO EXPERIMENTOS ({datetime.now().year}) ================")
    
    for replica_id in range(1, N_REPLICAS + 1):
        # Semente variando por réplica para garantir robustez estatística
        current_seed = args.seed + (replica_id * 100)
        
        for scenario in SCENARIOS:
            for model_name, script_path in MODELS.items():
                
                # Executa o modelo
                metrics, runtime, status = run_model(
                    model_name, script_path, current_seed, replica_id, scenario,
                    date_from=args.date_from,
                    date_to=args.date_to,
                    timeout_sec=args.model_timeout_sec,
                    max_retries=args.max_retries,
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

    # Geração automática dos gráficos ao final do pipeline
    try:
        print("\nGerando gráficos dos resultados...")
        subprocess.run(["python", str(SRC_DIR / "utils" / "plots.py")], check=True)
        print("Gráficos salvos em Resultados/plots/")
    except Exception as e:
        print(f"[!] Falha ao gerar gráficos automaticamente: {e}")

if __name__ == "__main__":
    main()
