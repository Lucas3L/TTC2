# Documentação das Telas, Scripts e Funções do Projeto

## Notebooks

### 01_analise_exploratoria.ipynb
- **Objetivo:** Análise exploratória das séries temporais de vendas, identificando padrões, tendências, sazonalidades, variabilidades e anomalias.
- **Blocos principais:**
  - Contextualização do problema de previsão de demanda no varejo.
  - Objetivos detalhados da análise exploratória.
  - (Demais células: análise de padrões, gráficos, identificação de outliers, etc.)

### 02_processamento.ipynb
- **Objetivo:** Estruturar o pipeline de processamento e engenharia de atributos para transformar dados brutos em dados prontos para modelagem.
- **Blocos principais:**
  - Contextualização sobre a importância da engenharia de atributos.
  - Objetivos: criação de atributos temporais, lags, janelamento, normalização e split dos dados.
  - (Demais células: implementação dos passos do pipeline.)

### 03_kmeans.ipynb
- **Objetivo:** Aplicar clusterização (K-Means) para agrupar produtos com padrões de venda semelhantes, reduzindo a complexidade e subsidiando modelagem segmentada.
- **Blocos principais:**
  - Contextualização sobre diversidade de produtos e necessidade de clusterização.
  - Objetivos: identificar grupos homogêneos via K-Means.
  - (Demais células: execução do K-Means, análise dos clusters.)

### 04_modelagem.ipynb
- **Objetivo:** Avaliar e comparar modelos preditivos (árvores de decisão, LSTM, GRU, XGBoost) para previsão de demanda.
- **Blocos principais:**
  - Contextualização sobre desafios da modelagem preditiva em séries temporais.
  - Objetivos: comparar desempenho e robustez dos modelos.
  - (Demais células: treinamento, validação e comparação dos modelos.)

### 05_resultados_e_analise.ipynb
- **Objetivo:** Análise estatística dos resultados dos experimentos de previsão de demanda, com múltiplas execuções para robustez.
- **Blocos principais:**
  - Contextualização e objetivos da análise estatística.
  - Importação de bibliotecas (pandas, numpy, matplotlib, seaborn).
  - (Demais células: análise descritiva, visualizações, discussão dos resultados.)

---

## Scripts principais

### main.py
- **Objetivo:** Script principal para configuração global, execução de experimentos e integração dos módulos do projeto.
- **Funções e blocos:**
  - Definição de seeds para reprodutibilidade (`set_global_seed`).
  - Extração de métricas de resultados (`extract_metrics`).
  - Configuração de diretórios e mapeamento de scripts de modelos.
  - (Demais funções: execução de experimentos, logging de erros, etc.)

### test_modelos_completo.py
- **Objetivo:** Testar todos os modelos (Baseline, XGBoost, LSTM, GRU) em diferentes cenários e validar resultados.
- **Funções e blocos:**
  - Logging colorido para facilitar leitura dos testes.
  - Função para verificação de pré-requisitos (existência de dados, etc.).
  - Execução dos modelos e validação dos resultados.
  - (Demais funções: logging de sucesso, erro, informações, etc.)

### test_pipeline_completo.py
- **Objetivo:** Testar o pipeline completo (carregamento, pré-processamento, split, validação) com logs detalhados.
- **Funções e blocos:**
  - Logging colorido.
  - Função para verificação de dados brutos.
  - Execução do pipeline e validação dos resultados.
  - (Demais funções: logging de sucesso, erro, informações, etc.)

---

## Passo a passo para rodar o modelo

1. **Preparação do ambiente**
   - Instale as dependências do projeto com `pip install -r requirements.txt`.
   - Certifique-se de que a estrutura de pastas de dados (`Dados/raw/Market_1`, `Market_2`, etc.) está preenchida com os arquivos brutos.

2. **Processamento dos dados**
   - Execute o script de carregamento e normalização dos dados brutos:
     - `src/data/load_raw_data.py` — Normaliza e organiza os dados em `Dados/processed`.
   - Execute o pré-processamento:
     - `src/data/preprocess.py` — Corrige datas, trata anomalias e salva em `Dados/preprocessed`.
   - Realize o split dos dados:
     - `src/data/split_data.py` — Separa em treino, validação e teste, salvando em `Dados/split`.

3. **Engenharia de atributos e features**
   - Gere features para cada mercado:
     - `src/features/build_features.py` — Cria atributos temporais, lags, etc., e salva em `Dados/features`.
   - (Opcional) Execute clusterização ou cenários específicos:
     - `src/features/kmeans_features.py` e `src/features/scenarios.py`.

4. **Modelagem**
   - Execute o(s) script(s) de modelagem desejados:
     - `src/models/lstm_model.py` — Treina e avalia modelo LSTM.
     - `src/models/gru_model.py` — Treina e avalia modelo GRU.
     - `src/models/xgboost_model.py` — Treina e avalia modelo XGBoost.
     - `src/models/baseline.py` — Executa baseline para comparação.

5. **Avaliação e análise dos resultados**
   - Os resultados são salvos na pasta `Resultados/`.
   - Utilize os notebooks de análise (`notebooks/05_resultados_e_analise.ipynb`) ou scripts utilitários (`src/utils/aggregation.py`, `src/utils/plots.py`) para consolidar, visualizar e comparar os resultados.

**Observação:**
- O fluxo pode ser automatizado via scripts de teste (`test_pipeline_completo.py`, `test_modelos_completo.py`) para rodar todas as etapas sequencialmente.
- Sempre verifique se as dependências e caminhos estão corretos para o seu ambiente.

---

## Finalidade para o projeto

Esses arquivos e notebooks estruturam o fluxo completo do projeto:
- Desde a análise inicial dos dados,
- Passando pelo processamento e engenharia de atributos,
- Clusterização para segmentação,
- Modelagem preditiva com diferentes algoritmos,
- Até a análise estatística dos resultados.

Os scripts principais automatizam e validam cada etapa, garantindo reprodutibilidade e robustez.

Caso deseje detalhamento de funções/células específicas, consulte o notebook ou script correspondente.

---

## src/data

### load_raw_data.py
- **Objetivo:** Carregar dados brutos de diferentes mercados, normalizar nomes de colunas e salvar em formato processado.
- **Funções:**
  - `normalize_columns(df)`: Normaliza nomes das colunas para snake_case e minúsculas.
- **Fluxo:** Percorre pastas de mercados em Dados/raw, processa arquivos e salva em Dados/processed.

### preprocess.py
- **Objetivo:** Pré-processar dados dos mercados, corrigindo datas, valores e tratando outliers.
- **Funções:** Utiliza funções do arquivo validators.py.
- **Fluxo:** Para cada mercado em Dados/processed, lê arquivos, corrige datas, trata anomalias e salva em Dados/preprocessed.

### split_data.py
- **Objetivo:** Separar os dados pré-processados em conjuntos de treino, validação e teste.
- **Parâmetros:** Proporções de split (train_ratio, val_ratio), janela mínima (WINDOW).
- **Fluxo:** Para cada mercado em Dados/preprocessed, separa os dados e salva em Dados/split.

### validators.py
- **Objetivo:** Fornecer funções utilitárias para validação e correção dos dados.
- **Funções:**
  - `corrigir_datas_temporais(df, max_faltantes, anomalias)`: Corrige datas faltantes/interpoladas por produto, marca anomalias severas.
  - (Outras funções podem estar presentes para valores e outliers.)

---

## src/features

### build_features.py
- **Objetivo:** Gerar e salvar features para cada mercado a partir dos dados já separados (split).
- **Funções:**
  - `process_market(market_dir)`: Para cada arquivo CSV de um mercado, lê os dados, gera features usando a função `build_features` e salva os resultados.
- **Fluxo:** Percorre mercados em Dados/split, gera features e salva em Dados/features.

### feature_engineering.py
- **Objetivo:** Funções utilitárias para engenharia de atributos e preparação de sequências para modelos sequenciais.
- **Funções:**
  - `create_sequences_by_product(df, features, target, window, positive_weight)`: Gera amostras sequenciais (janelas) por produto para LSTM/GRU, com pesos para amostras positivas.
  - `add_price_segments(df)`: Segmenta produtos em quartis de preço.

### kmeans_features.py
- **Objetivo:** Gerar clusters de produtos usando K-Means com base em quantidade vendida e valor médio.
- **Funções:**
  - `generate_product_clusters(df, n_clusters)`: Agrupa produtos em clusters, retornando o ID do produto e o cluster atribuído.

### scenarios.py
- **Objetivo:** Aplicar diferentes cenários de agrupamento de produtos para análise/modelagem.
- **Funções:**
  - `apply_scenario(df, scenario)`: Aplica agrupamento por volume, preço ou K-Means.
  - `group_by_volume(df)`: Agrupa produtos em baixo, médio e alto volume.
  - `group_by_price(df)`: Agrupa produtos em baratos, médios e caros.

---

## src/models

### baseline.py
- **Objetivo:** Implementa o modelo baseline (zero-aware), que faz previsões simples usando o último valor conhecido (lag).
- **Funções:**
  - `naive_lag_forecast(df, feature)`: Previsão baseada no valor defasado.
  - `process_file(csv_file)`: Processa arquivos de dados, aplica baseline e salva resultados.
- **Fluxo:** Lê dados pré-processados, aplica baseline e salva resultados em Resultados/baseline_zero_aware.

### evaluate.py
- **Objetivo:** Funções de avaliação de modelos, incluindo cálculo de métricas.
- **Funções:**
  - `smape(y_true, y_pred)`: Calcula o erro percentual simétrico médio.
  - `evaluate(y_true, y_pred)`: Retorna dicionário com MAE, RMSE e sMAPE.

### gru_model.py
- **Objetivo:** Implementa modelo GRU para previsão de séries temporais.
- **Funções:**
  - Define arquitetura GRU, treinamento, validação e avaliação.
  - Utiliza funções utilitárias para features, métricas e cenários.
- **Fluxo:** Lê dados, prepara sequências, treina GRU, avalia e salva resultados.

### lstm_model.py
- **Objetivo:** Implementa modelo LSTM para previsão de séries temporais.
- **Funções:**
  - Define arquitetura LSTM, treinamento, validação e avaliação.
  - Utiliza funções utilitárias para features, métricas e cenários.
- **Fluxo:** Lê dados, prepara sequências, treina LSTM, avalia e salva resultados.

### predict.py
- **Objetivo:** Função utilitária para realizar predições com modelos treinados e reverter normalização dos resultados.
- **Funções:**
  - `predict_model(model, X_seq, scaler_y)`: Executa inferência e inverte normalização.

### xgboost_model.py
- **Objetivo:** Implementa modelo XGBoost para previsão de séries temporais.
- **Funções:**
  - Define pipeline de features, treinamento, validação e avaliação.
  - Utiliza funções utilitárias para features, métricas e cenários.
- **Fluxo:** Lê dados, prepara features, treina XGBoost, avalia e salva resultados.

---

## src/preprocessing

### sequence.py
- **Objetivo:** Funções para criação de sequências temporais para modelos sequenciais.
- **Funções:**
  - `create_sequences(df, feature_cols, target_col, window)`: Gera amostras de entrada e alvo para cada produto, usando janela deslizante.

---

## src/utils

### aggregation.py
- **Objetivo:** Carregar e consolidar resultados de diferentes modelos para análise comparativa.
- **Funções:**
  - `load_model_results(model_name)`: Lê arquivos de resultados, padroniza e consolida métricas.

### code_hygiene_check.py
- **Objetivo:** Ferramentas para checagem de higiene de código e notebooks (imports duplicados, linhas repetidas).
- **Funções:**
  - `_check_python_file(path)`: Verifica problemas em scripts Python.
  - `_check_notebook(path)`: Verifica problemas em notebooks.

### experiment_logger.py
- **Objetivo:** Classe para registrar e salvar logs de experimentos de modelos.
- **Funções:**
  - `ExperimentLogger`: Permite logar execuções, parâmetros, métricas e salvar em CSV.

### helpers.py
- **Objetivo:** Funções utilitárias para manipulação de diretórios, colunas e features temporais.
- **Funções:**
  - `ensure_dir(path)`: Garante existência de diretório.
  - `normalize_columns(df)`: Normaliza nomes de colunas.
  - `add_lag_features(df, target, lags)`: Adiciona colunas de lag e médias móveis.
  - `add_intermittent_features(df, target)`: Adiciona features para séries intermitentes.

### io.py
- **Objetivo:** Funções para leitura e escrita robusta de arquivos CSV.
- **Funções:**
  - `read_csv(path, parse_dates)`: Lê CSV com validação de existência.
  - `save_csv(df, path, index)`: Salva DataFrame em CSV, criando diretórios se necessário.
  - `list_csv_files(folder)`: Lista arquivos CSV em uma pasta.

### metrics.py
- **Objetivo:** Implementação de métricas de avaliação para modelos de séries temporais.
- **Funções:**
  - `mae(y_true, y_pred)`, `rmse(y_true, y_pred)`, `smape(y_true, y_pred)`: Métricas clássicas.
  - `evaluate_all(y_true, y_pred)`: Retorna todas as métricas em dicionário.

### notebook_audit.py
- **Objetivo:** Ferramentas para auditoria de notebooks (estatísticas, recomendações de refatoração).
- **Funções:**
  - `NotebookStats`: Classe com estatísticas e recomendações para notebooks.

### plots.py
- **Objetivo:** Funções para geração de gráficos e visualizações dos resultados dos modelos.
- **Funções:**
  - Utiliza matplotlib para criar gráficos a partir dos resultados consolidados.

### reproducibility.py
- **Objetivo:** Garantir reprodutibilidade dos experimentos.
- **Funções:**
  - `set_global_seed(seed)`: Fixa seeds para Python, Numpy e TensorFlow.

### statistics.py
- **Objetivo:** Funções para análise estatística dos resultados dos modelos.
- **Funções:**
  - `bootstrap_ci(y_true, y_pred, metric_fn, ...)`: Calcula intervalo de confiança via bootstrap.
  - `diebold_mariano_test(e1, e2, ...)`: Teste estatístico para comparar modelos.