#!/usr/bin/env python3
"""
Script de teste completo do pipeline: Carregamento → Pré-processamento → Split → Validação
Executa o pipeline inteiro com logs detalhados.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

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
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}{Colors.ENDC}\n")

def log_success(msg):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")

def log_error(msg):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")

def log_info(msg):
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")

def log_warn(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

# ============================================================================
# FASE 1: Verificação de dados brutos
# ============================================================================
def test_raw_data():
    log_section("FASE 1: Verificação de Dados Brutos")
    
    raw_path = BASE_DIR / "Dados" / "raw"
    
    if not raw_path.exists():
        log_error(f"Diretório 'Dados/raw' não encontrado: {raw_path}")
        return False
    
    log_success(f"Diretório 'Dados/raw' encontrado")
    
    markets = 0
    files = 0
    
    for market_dir in raw_path.iterdir():
        if market_dir.is_dir():
            markets += 1
            market_name = market_dir.name
            log_info(f"Mercado encontrado: {market_name}")
            
            for cat_dir in market_dir.iterdir():
                if cat_dir.is_dir():
                    csv_count = len(list(cat_dir.glob("*.csv")))
                    files += csv_count
                    log_info(f"  → {cat_dir.name}: {csv_count} arquivos CSV")
    
    log_success(f"Total: {markets} mercados, {files} arquivos de dados")
    return True

# ============================================================================
# FASE 2: Testes do Load Raw Data
# ============================================================================
def test_load_raw_data():
    log_section("FASE 2: Teste de Carregamento (load_raw_data.py)")
    
    script_path = SRC_DIR / "data" / "load_raw_data.py"
    
    if not script_path.exists():
        log_error(f"Script não encontrado: {script_path}")
        return False
    
    log_info("Executando script de carregamento...")
    
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutos
            cwd=str(BASE_DIR)
        )
        
        if result.returncode != 0:
            log_error(f"Script falhou com código {result.returncode}")
            print(result.stderr)
            return False
        
        log_success("Script de carregamento executado com sucesso")
        print(result.stdout)
        
        # Verifica se diretório 'processed' foi criado
        processed_path = BASE_DIR / "Dados" / "processed"
        if processed_path.exists():
            log_success("Diretório 'Dados/processed' criado com sucesso")
            
            # Conta arquivos processados
            csv_count = len(list(processed_path.glob("**/cat*.csv")))
            log_info(f"Arquivos processados: {csv_count}")
            return True
        else:
            log_warn("Diretório 'Dados/processed' não encontrado após execução")
            return False
            
    except subprocess.TimeoutExpired:
        log_error("Script expirou (timeout)")
        return False
    except Exception as e:
        log_error(f"Erro ao executar script: {e}")
        return False

# ============================================================================
# FASE 3: Testes do Preprocess
# ============================================================================
def test_preprocess():
    log_section("FASE 3: Teste de Pré-processamento (preprocess.py)")
    
    # Primeiro, verifica se dados processados existem
    processed_path = BASE_DIR / "Dados" / "processed"
    if not processed_path.exists() or len(list(processed_path.glob("**/cat*.csv"))) == 0:
        log_error("Nenhum dado processado encontrado. Execute a Fase 2 primeiro.")
        return False
    
    script_path = SRC_DIR / "data" / "preprocess.py"
    
    if not script_path.exists():
        log_error(f"Script não encontrado: {script_path}")
        return False
    
    log_info("Executando script de pré-processamento...")
    
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutos
            cwd=str(BASE_DIR)
        )
        
        if result.returncode != 0:
            log_error(f"Script falhou com código {result.returncode}")
            print(result.stderr)
            return False
        
        log_success("Script de pré-processamento executado com sucesso")
        print(result.stdout)
        
        # Verifica se diretório 'preprocessed' foi criado
        preprocessed_path = BASE_DIR / "Dados" / "preprocessed"
        if preprocessed_path.exists():
            log_success("Diretório 'Dados/preprocessed' criado com sucesso")
            
            # Conta arquivos pré-processados
            csv_count = len(list(preprocessed_path.glob("**/cat*.csv")))
            log_info(f"Arquivos pré-processados: {csv_count}")
            
            # Valida conteúdo de um arquivo
            sample_files = list(preprocessed_path.glob("**/cat*.csv"))
            if sample_files:
                sample_df = pd.read_csv(sample_files[0])
                log_info(f"Amostra do arquivo: {sample_files[0].name}")
                log_info(f"  → Shape: {sample_df.shape}")
                log_info(f"  → Colunas: {list(sample_df.columns)}")
            
            return True
        else:
            log_warn("Diretório 'Dados/preprocessed' não encontrado após execução")
            return False
            
    except subprocess.TimeoutExpired:
        log_error("Script expirou (timeout)")
        return False
    except Exception as e:
        log_error(f"Erro ao executar script: {e}")
        return False

# ============================================================================
# FASE 4: Testes do Split
# ============================================================================
def test_split_data():
    log_section("FASE 4: Teste de Split de Dados (split_data.py)")
    
    # Verifica se dados pré-processados existem
    preprocessed_path = BASE_DIR / "Dados" / "preprocessed"
    if not preprocessed_path.exists() or len(list(preprocessed_path.glob("**/cat*.csv"))) == 0:
        log_error("Nenhum dado pré-processado encontrado. Execute a Fase 3 primeiro.")
        return False
    
    script_path = SRC_DIR / "data" / "split_data.py"
    
    if not script_path.exists():
        log_error(f"Script não encontrado: {script_path}")
        return False
    
    log_info("Executando script de split...")
    
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutos
            cwd=str(BASE_DIR)
        )
        
        if result.returncode != 0:
            log_error(f"Script falhou com código {result.returncode}")
            print(result.stderr)
            return False
        
        log_success("Script de split executado com sucesso")
        print(result.stdout)
        
        # Verifica se diretório 'split' foi criado
        split_path = BASE_DIR / "Dados" / "split"
        if split_path.exists():
            log_success("Diretório 'Dados/split' criado com sucesso")
            
            # Conta arquivos splitados
            csv_count = len(list(split_path.glob("**/cat*.csv")))
            log_info(f"Arquivos com split train/val/test: {csv_count}")
            
            # Valida conteúdo de um arquivo
            sample_files = list(split_path.glob("**/cat*.csv"))
            if sample_files:
                sample_df = pd.read_csv(sample_files[0])
                log_info(f"Amostra do arquivo: {sample_files[0].name}")
                log_info(f"  → Shape: {sample_df.shape}")
                
                if 'split' in sample_df.columns:
                    splits = sample_df['split'].value_counts().to_dict()
                    log_info(f"  → Distribuição de split: {splits}")
                    
                    train_pct = (splits.get('train', 0) / len(sample_df)) * 100
                    val_pct = (splits.get('val', 0) / len(sample_df)) * 100
                    test_pct = (splits.get('test', 0) / len(sample_df)) * 100
                    log_info(f"  → Percentuais: Train={train_pct:.1f}% | Val={val_pct:.1f}% | Test={test_pct:.1f}%")
            
            return True
        else:
            log_warn("Diretório 'Dados/split' não encontrado após execução")
            return False
            
    except subprocess.TimeoutExpired:
        log_error("Script expirou (timeout)")
        return False
    except Exception as e:
        log_error(f"Erro ao executar script: {e}")
        return False

# ============================================================================
# FASE 5: Validação do Pipeline
# ============================================================================
def test_pipeline_validation():
    log_section("FASE 5: Validação Integrada do Pipeline")
    
    split_path = BASE_DIR / "Dados" / "split"
    
    if not split_path.exists():
        log_error("Diretório 'Dados/split' não encontrado")
        return False
    
    all_df = []
    
    for market_dir in split_path.iterdir():
        if market_dir.is_dir():
            market_name = market_dir.name
            log_info(f"Validando mercado: {market_name}")
            
            for csv_file in market_dir.glob("cat*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    all_df.append(df)
                    
                    # Verificações básicas
                    if 'split' not in df.columns:
                        log_warn(f"  ✗ {csv_file.name}: coluna 'split' ausente")
                        continue
                    
                    if 'date' not in df.columns:
                        log_warn(f"  ✗ {csv_file.name}: coluna 'date' ausente")
                        continue
                    
                    # Verifica integridade
                    null_count = df.isnull().sum().sum()
                    if null_count > 0:
                        log_warn(f"  ⚠ {csv_file.name}: {null_count} valores nulos encontrados")
                    else:
                        log_success(f"  {csv_file.name} validado com sucesso")
                    
                except Exception as e:
                    log_error(f"  Erro lendo {csv_file.name}: {e}")
    
    if all_df:
        combined_df = pd.concat(all_df, ignore_index=True)
        log_success(f"Total de registros validados: {len(combined_df)}")
        log_info(f"Distribuição de splits:")
        print(combined_df['split'].value_counts().to_string())
        return True
    else:
        log_error("Nenhum arquivo foi validado")
        return False

# ============================================================================
# MAIN
# ============================================================================
def main():
    log_section("TESTE COMPLETO DO PIPELINE DE DADOS")
    
    log_info(f"Diretório base: {BASE_DIR}")
    log_info(f"Hora de início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    results = {}
    
    # Executa as fases
    results['raw_data'] = test_raw_data()
    if not results['raw_data']:
        log_error("Falha na verificação de dados brutos. Abortando.")
        return False
    
    results['load_raw_data'] = test_load_raw_data()
    if not results['load_raw_data']:
        log_warn("Falha no carregamento de dados. Continuando...")
    
    results['preprocess'] = test_preprocess()
    if not results['preprocess']:
        log_warn("Falha no pré-processamento. Continuando...")
    
    results['split_data'] = test_split_data()
    if not results['split_data']:
        log_warn("Falha no split. Continuando...")
    
    results['validation'] = test_pipeline_validation()
    
    # Resumo final
    log_section("RESUMO DOS TESTES")
    
    for fase, resultado in results.items():
        status = "PASSOU ✓" if resultado else "FALHOU ✗"
        cor = Colors.OKGREEN if resultado else Colors.FAIL
        print(f"{cor}{fase.upper().ljust(30)} {status}{Colors.ENDC}")
    
    elapsed = time.time() - start_time
    log_info(f"Tempo total: {elapsed:.2f} segundos")
    log_info(f"Hora de término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_passed = all(results.values())
    
    if all_passed:
        log_success("TODOS OS TESTES PASSARAM!")
    else:
        log_warn("Alguns testes falharam. Verifique os logs acima.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
