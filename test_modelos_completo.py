#!/usr/bin/env python3
"""
Script de teste completo dos modelos: Baseline, XGBoost, LSTM, GRU
Executa todos os modelos com diferentes cenários e valida os resultados.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import argparse

# Adiciona o diretório src ao path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Cores para output (Windows compatible)
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_section(title):
    """Imprime um cabeçalho para uma seção."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}{Colors.ENDC}\n")

def log_success(msg):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")

def log_error(msg):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")

def log_info(msg):
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")

def log_warn(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

# ============================================================================
# FASE 1: Verificação de Pré-requisitos
# ============================================================================
def test_prerequisites():
    log_section("FASE 1: Verificação de Pré-requisitos")

    # Verificar se dados pré-processados existem
    preprocessed_path = BASE_DIR / "Dados" / "preprocessed"
    if not preprocessed_path.exists():
        log_error("Diretório 'Dados/preprocessed' não encontrado. Execute o pipeline de dados primeiro.")
        return False

    csv_files = list(preprocessed_path.glob("**/cat*.csv"))
    if len(csv_files) == 0:
        log_error("Nenhum arquivo pré-processado encontrado.")
        return False

    log_success(f"Encontrados {len(csv_files)} arquivos pré-processados")

    # Verificar se features existem (cenário kmeans)
    features_path = BASE_DIR / "Dados" / "features"
    if features_path.exists():
        kmeans_files = list(features_path.glob("kmeans_features.csv"))
        if kmeans_files:
            log_success("Features k-means encontradas")
        else:
            log_warn("Features k-means não encontradas - cenário kmeans pode falhar")
    else:
        log_warn("Diretório de features não encontrado - cenário kmeans pode falhar")

    # Verificar scripts dos modelos
    models_dir = SRC_DIR / "models"
    required_models = ["baseline.py", "xgboost_model.py", "lstm_model.py", "gru_model.py"]

    for model_file in required_models:
        model_path = models_dir / model_file
        if not model_path.exists():
            log_error(f"Script do modelo não encontrado: {model_file}")
            return False

    log_success("Todos os scripts de modelos encontrados")

    # Verificar diretório de resultados
    results_dir = BASE_DIR / "Resultados"
    results_dir.mkdir(exist_ok=True)
    log_success("Diretório de resultados criado/verificado")

    return True

# ============================================================================
# FASE 2: Teste do Modelo Baseline
# ============================================================================
def test_baseline_model(quick=False):
    log_section("FASE 2: Teste do Modelo Baseline")

    script_path = SRC_DIR / "models" / "baseline.py"

    scenarios = ["volume", "price"]
    results = {}

    for scenario in scenarios:
        log_info(f"Testando Baseline com cenário: {scenario}")

        try:
            result = subprocess.run(
                ["python", str(script_path), "--scenario", scenario],
                capture_output=True,
                text=True,
                timeout=300 if quick else 600,  # 5m rápido / 10m normal
                cwd=str(BASE_DIR)
            )

            if result.returncode != 0:
                log_error(f"Baseline falhou no cenário {scenario}")
                print(result.stderr)
                results[scenario] = False
            else:
                log_success(f"Baseline executado com sucesso no cenário {scenario}")
                results[scenario] = True

                # Verificar se arquivos de resultado foram criados
                output_files = list((BASE_DIR / "Resultados" / "baseline_zero_aware").glob("*.csv"))
                if output_files:
                    log_info(f"  → {len(output_files)} arquivos de resultado gerados")
                else:
                    log_warn("  → Nenhum arquivo de resultado encontrado")

        except subprocess.TimeoutExpired:
            log_error(f"Baseline expirou no cenário {scenario}")
            results[scenario] = False
        except Exception as e:
            log_error(f"Erro no Baseline cenário {scenario}: {e}")
            results[scenario] = False

    return all(results.values())

# ============================================================================
# FASE 3: Teste do Modelo XGBoost
# ============================================================================
def test_xgboost_model(quick=False):
    log_section("FASE 3: Teste do Modelo XGBoost")

    script_path = SRC_DIR / "models" / "xgboost_model.py"

    scenarios = ["volume", "price", "kmeans"] if not quick else ["volume"]
    results = {}

    for scenario in scenarios:
        log_info(f"Testando XGBoost com cenário: {scenario}")

        try:
            result = subprocess.run(
                ["python", str(script_path), "--scenario", scenario],
                capture_output=True,
                text=True,
                timeout=450 if quick else 900,  # 7.5m rápido / 15m normal
                cwd=str(BASE_DIR)
            )

            if result.returncode != 0:
                log_error(f"XGBoost falhou no cenário {scenario}")
                print(result.stderr)
                results[scenario] = False
            else:
                log_success(f"XGBoost executado com sucesso no cenário {scenario}")
                results[scenario] = True

                # Verificar se arquivos de resultado foram criados
                output_files = list((BASE_DIR / "Resultados" / "xgb").glob("*.csv"))
                if output_files:
                    log_info(f"  → {len(output_files)} arquivos de resultado gerados")
                else:
                    log_warn("  → Nenhum arquivo de resultado encontrado")

        except subprocess.TimeoutExpired:
            log_error(f"XGBoost expirou no cenário {scenario}")
            results[scenario] = False
        except Exception as e:
            log_error(f"Erro no XGBoost cenário {scenario}: {e}")
            results[scenario] = False

    return all(results.values())

# ============================================================================
# FASE 4: Teste do Modelo LSTM
# ============================================================================
def test_lstm_model(quick=False):
    log_section("FASE 4: Teste do Modelo LSTM")

    script_path = SRC_DIR / "models" / "lstm_model.py"

    scenarios = ["volume", "price"] if not quick else ["volume"]
    results = {}

    for scenario in scenarios:
        log_info(f"Testando LSTM com cenário: {scenario}")

        try:
            result = subprocess.run(
                ["python", str(script_path), "--scenario", scenario, "--epochs", "5" if quick else "50"],
                capture_output=True,
                text=True,
                timeout=900,  # 15m (agora com menos épocas)
                cwd=str(BASE_DIR)
            )

            if result.returncode != 0:
                log_error(f"LSTM falhou no cenário {scenario}")
                print(result.stderr)
                results[scenario] = False
            else:
                log_success(f"LSTM executado com sucesso no cenário {scenario}")
                results[scenario] = True

                # Verificar se arquivos de resultado foram criados
                output_files = list((BASE_DIR / "Resultados" / "lstm").glob("*.csv"))
                if output_files:
                    log_info(f"  → {len(output_files)} arquivos de resultado gerados")
                else:
                    log_warn("  → Nenhum arquivo de resultado encontrado")

        except subprocess.TimeoutExpired:
            log_error(f"LSTM expirou no cenário {scenario}")
            results[scenario] = False
        except Exception as e:
            log_error(f"Erro no LSTM cenário {scenario}: {e}")
            results[scenario] = False

    return all(results.values())

# ============================================================================
# FASE 5: Teste do Modelo GRU
# ============================================================================
def test_gru_model(quick=False):
    log_section("FASE 5: Teste do Modelo GRU")

    script_path = SRC_DIR / "models" / "gru_model.py"

    scenarios = ["volume", "price"] if not quick else ["volume"]
    results = {}

    for scenario in scenarios:
        log_info(f"Testando GRU com cenário: {scenario}")

        try:
            result = subprocess.run(
                ["python", str(script_path), "--scenario", scenario, "--epochs", "5" if quick else "50"],
                capture_output=True,
                text=True,
                timeout=900,  # 15m (agora com menos épocas)
                cwd=str(BASE_DIR)
            )

            if result.returncode != 0:
                log_error(f"GRU falhou no cenário {scenario}")
                print(result.stderr)
                results[scenario] = False
            else:
                log_success(f"GRU executado com sucesso no cenário {scenario}")
                results[scenario] = True

                # Verificar se arquivos de resultado foram criados
                output_files = list((BASE_DIR / "Resultados" / "gru").glob("*.csv"))
                if output_files:
                    log_info(f"  → {len(output_files)} arquivos de resultado gerados")
                else:
                    log_warn("  → Nenhum arquivo de resultado encontrado")

        except subprocess.TimeoutExpired:
            log_error(f"GRU expirou no cenário {scenario}")
            results[scenario] = False
        except Exception as e:
            log_error(f"Erro no GRU cenário {scenario}: {e}")
            results[scenario] = False

    return all(results.values())

# ============================================================================
# FASE 6: Validação dos Resultados
# ============================================================================
def validate_results():
    log_section("FASE 6: Validação dos Resultados")

    results_dir = BASE_DIR / "Resultados"
    expected_models = ["baseline_zero_aware", "xgb", "lstm", "gru"]

    total_files = 0
    valid_results = {}

    for model_dir in expected_models:
        model_path = results_dir / model_dir
        if not model_path.exists():
            log_warn(f"Diretório de resultados não encontrado: {model_dir}")
            valid_results[model_dir] = False
            continue

        csv_files = list(model_path.glob("*.csv"))
        total_files += len(csv_files)

        if csv_files:
            log_success(f"{model_dir}: {len(csv_files)} arquivos encontrados")

            # Validar conteúdo de um arquivo de exemplo
            try:
                sample_df = pd.read_csv(csv_files[0])
                # Aceitar estrutura de métricas: model, arquivo, product_id, mae, rmse, smape
                required_cols = ["product_id", "mae", "rmse", "smape"]

                missing_cols = [col for col in required_cols if col not in sample_df.columns]
                if missing_cols:
                    log_warn(f"  → Colunas ausentes em {csv_files[0].name}: {missing_cols}")
                    valid_results[model_dir] = False
                else:
                    log_success(f"  → Estrutura válida em {csv_files[0].name}")
                    log_info(f"    Shape: {sample_df.shape}")

                    # Verificar se há métricas válidas
                    if sample_df["mae"].notna().any():
                        log_success("    → Métricas de avaliação encontradas")
                        valid_results[model_dir] = True
                    else:
                        log_warn("    → Métricas ausentes")
                        valid_results[model_dir] = False

            except Exception as e:
                log_error(f"Erro ao validar {csv_files[0].name}: {e}")
                valid_results[model_dir] = False
        else:
            log_warn(f"{model_dir}: Nenhum arquivo encontrado")
            valid_results[model_dir] = False

    log_info(f"Total de arquivos de resultado: {total_files}")
    return all(valid_results.values()), valid_results

# ============================================================================
# FASE 7: Comparação de Performance
# ============================================================================
def compare_performance():
    log_section("FASE 7: Comparação de Performance dos Modelos")

    results_dir = BASE_DIR / "Resultados"

    # Tentar ler métricas consolidadas se existirem
    metrics_file = results_dir / "model_comparison.csv"
    if metrics_file.exists():
        try:
            metrics_df = pd.read_csv(metrics_file)
            log_success("Métricas consolidadas encontradas")
            print(metrics_df.to_string(index=False))
            return True
        except Exception as e:
            log_error(f"Erro ao ler métricas: {e}")

    # Caso não haja métricas consolidadas, fazer análise básica
    model_dirs = ["baseline_zero_aware", "xgb", "lstm", "gru"]
    summary = {}

    for model_dir in model_dirs:
        model_path = results_dir / model_dir
        if model_path.exists():
            csv_files = list(model_path.glob("*.csv"))
            summary[model_dir] = len(csv_files)
        else:
            summary[model_dir] = 0

    log_info("Resumo de arquivos gerados por modelo:")
    for model, count in summary.items():
        print(f"  {model}: {count} arquivos")

    return True

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Execute testes de modelos (com modo rápido)")
    parser.add_argument("--quick", action="store_true", help="Executa apenas cenários básicos com timeouts reduzidos")
    args = parser.parse_args()
    quick_mode = args.quick

    log_section("TESTE COMPLETO DOS MODELOS DE MACHINE LEARNING")

    log_info(f"Diretório base: {BASE_DIR}")
    log_info(f"Hora de início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()
    results = {}

    # Executa as fases
    results['prerequisites'] = test_prerequisites()
    if not results['prerequisites']:
        log_error("Pré-requisitos não atendidos. Abortando.")
        return False

    results['baseline'] = test_baseline_model(quick_mode)
    results['xgboost'] = test_xgboost_model(quick_mode)
    results['lstm'] = test_lstm_model(quick_mode)
    results['gru'] = test_gru_model(quick_mode)

    validation_success, validation_details = validate_results()
    results['validation'] = validation_success

    results['comparison'] = compare_performance()

    # Resumo final
    log_section("RESUMO DOS TESTES DOS MODELOS")

    for fase, resultado in results.items():
        status = "PASSOU ✓" if resultado else "FALHOU ✗"
        cor = Colors.OKGREEN if resultado else Colors.FAIL
        print(f"{cor}{fase.upper().ljust(20)} {status}{Colors.ENDC}")

    elapsed = time.time() - start_time
    log_info(f"Tempo total: {elapsed:.2f} segundos")
    log_info(f"Hora de término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Análise detalhada da validação
    if 'validation' in results and results['validation']:
        log_success("TODOS OS MODELOS FORAM EXECUTADOS COM SUCESSO!")
    else:
        log_warn("Alguns modelos falharam ou não geraram resultados válidos.")

        # Mostrar detalhes da validação
        if validation_details:
            log_info("Detalhes da validação:")
            for model, valid in validation_details.items():
                status = "VÁLIDO ✓" if valid else "INVÁLIDO ✗"
                cor = Colors.OKGREEN if valid else Colors.FAIL
                print(f"  {cor}{model.ljust(20)} {status}{Colors.ENDC}")

    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
