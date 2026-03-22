# Comparativo técnico — estrutura do projeto e tecnologias de modelagem

## Objetivo do documento

Este documento compara a estrutura adotada no projeto e os modelos implementados
com alternativas que resolvem o mesmo problema de previsão de demanda
(séries temporais com intermitência, sazonalidade e múltiplos produtos),
explicando por que a solução atual foi considerada a mais adequada para este contexto.

---

## 1) Estrutura do projeto: o que foi adotado e alternativas

## Estrutura adotada

- **Pipeline em camadas**:
  - ingestão (`src/data/load_raw_data.py`)
  - validação/preprocessamento (`src/data/preprocess.py`, `src/data/validators.py`)
  - features/cenários (`src/features/*`)
  - modelos (`src/models/*`)
  - utilitários/orquestração (`src/utils/*`, `main.py`).
- **Configuração centralizada** em `src/config/` para:
  - experimento (`experiment_config.py`),
  - parâmetros comuns de modelos (`model_params.py`),
  - parâmetros de cenários (`scenario_params.py`).

### Alternativas comparadas

1. **Estrutura monolítica por notebook/script único**
   - Prós: mais rápida para prova de conceito.
   - Contras: baixa reprodutibilidade, alta duplicação, difícil manutenção.

2. **Arquitetura orientada a framework de MLOps completo (ex.: pipelines com orquestradores externos)**
   - Prós: governança avançada e automação robusta.
   - Contras: custo de adoção alto para escopo acadêmico/aplicado e maior complexidade operacional.

### Por que a estrutura atual foi a melhor escolha

- Equilibra robustez e simplicidade para o tamanho do projeto.
- Permite evolução incremental (novos modelos/cenários sem refatoração global).
- Facilita auditoria e rastreabilidade com logs de erro e descarte.
- Mantém custo de operação baixo sem depender de stack externa pesada.

---

## 2) Tecnologias utilizadas e alternativas

## Tecnologias adotadas

- **Python + Pandas/Numpy** para processamento de dados tabulares/temporais.
- **Scikit-learn** para utilitários clássicos (normalização, encoding).
- **XGBoost** para modelo baseado em árvores.
- **TensorFlow/Keras** para redes neurais recorrentes (LSTM/GRU).
- **CLI + subprocess em `main.py`** para orquestração dos experimentos.
- **CSV como artefato padrão** para resultados, predições e auditoria de descartes.

### Alternativas comparadas

1. **PyTorch em vez de TensorFlow/Keras**
   - Prós: flexibilidade alta.
   - Contras: maior custo de implementação para o objetivo atual (pipeline aplicado e rápido de operar).

2. **LightGBM/CatBoost no lugar do XGBoost**
   - Prós: em alguns cenários podem ser mais rápidos.
   - Contras: exigiriam nova calibração de hiperparâmetros e não oferecem vantagem clara garantida no contexto já padronizado.

3. **Banco de dados para artefatos (em vez de CSV)**
   - Prós: versionamento/consulta mais avançada.
   - Contras: overhead operacional desnecessário para o volume e objetivo atual.

### Por que o stack atual foi mais adequado

- Ferramentas maduras e amplamente suportadas.
- Curva de aprendizado compatível com cronograma do projeto.
- Integração direta com o ecossistema já usado no pipeline.
- Boa relação desempenho x complexidade para replicação de experimentos.

---

## 3) Modelos implementados vs alternativas equivalentes

## Modelos implementados

1. **Baseline (lag ingênuo)**
   - Função: referência mínima de desempenho.
   - Valor: estabelece piso técnico para justificar ganho de modelos mais complexos.

2. **XGBoost**
   - Função: capturar não linearidades com features de lags, rolling e calendário.
   - Valor: forte baseline supervisionado em dados tabulares temporais.

3. **LSTM**
   - Função: capturar dependências temporais de médio/longo prazo.
   - Valor: modelagem sequencial robusta com representação temporal contínua.

4. **GRU**
   - Função: alternativa recorrente mais leve que LSTM.
   - Valor: custo computacional menor com desempenho competitivo em muitas séries.

### Alternativas comparadas (mesma finalidade)

1. **Prophet / modelos aditivos clássicos**
   - Bons para sazonalidade interpretável.
   - Menos flexíveis para múltiplos sinais exógenos e relações não lineares complexas por produto.

2. **ARIMA/SARIMA por série**
   - Bons em séries univariadas estáveis.
   - Escalam pior em grande volume de SKUs e múltiplos preditores.

3. **TCN / Transformers (TFT, Informer, etc.)**
   - Alto potencial em séries complexas.
   - Maior custo de engenharia/treino, tuning sensível, risco de overkill para o escopo e hardware disponíveis.

4. **Croston/SBA/TSB (intermitência)**
   - Fortes para séries extremamente esparsas.
   - Menos abrangentes para cenários híbridos com sinais exógenos e comparação ampla entre famílias de modelos.

### Por que os modelos escolhidos foram mais adequados aqui

- Cobrem espectro completo de complexidade:
  - referência simples (Baseline),
  - aprendizado tabular robusto (XGBoost),
  - aprendizado sequencial (LSTM/GRU).
- Permitem comparação justa com parâmetros centralizados.
- São compatíveis com o nível de recursos computacionais e prazos do projeto.
- Entregam saída operacional útil (`*_predictions.csv`) para análise temporal por produto.

---

## 4) Critérios de decisão utilizados

Para a escolha da solução, os critérios práticos foram:

1. **Aderência ao problema de negócio** (demanda por produto, intermitência e sazonalidade).
2. **Reprodutibilidade** (seed, parâmetros centralizados, logs de erro/descarte).
3. **Custo computacional** (tempo de treino/inferência no ambiente disponível).
4. **Manutenibilidade** (baixa duplicação e modularidade).
5. **Auditabilidade** (rastrear descartes, anomalias e artefatos de saída).
6. **Comparabilidade entre modelos** (mesmas bases, splits e features centrais).

---

## 5) Conclusão executiva

A combinação **Baseline + XGBoost + LSTM + GRU** em uma arquitetura modular
com configuração centralizada foi a melhor escolha para este projeto porque:

- equilibra precisão potencial e custo operacional;
- reduz risco de dependência em tecnologia complexa demais para o escopo;
- mantém rastreabilidade dos dados e resultados;
- facilita evolução futura para modelos mais avançados quando houver necessidade comprovada.

Em resumo: a solução atual não é apenas tecnicamente válida, mas também
**mais adequada ao contexto real de implementação e manutenção**.

---

## 6) Teoria do projeto em linguagem simples (para leigos)

O projeto resolve um problema clássico: **prever demanda futura de produtos**.
Na prática, isso significa responder: *“quanto de cada item tende a vender nos próximos dias?”*.

Para fazer isso com confiabilidade, o trabalho foi planejado em etapas:

1. **Organizar os dados brutos** (que vêm “sujos” e com formatos variados).
2. **Limpar e corrigir inconsistências** (datas faltantes, valores inválidos, zeros suspeitos).
3. **Criar variáveis úteis para previsão** (ex.: histórico de vendas, médias móveis, efeitos de calendário).
4. **Treinar diferentes tipos de modelo** (do mais simples ao mais avançado).
5. **Comparar resultados de forma padronizada** (mesmas métricas e mesma lógica de divisão dos dados).
6. **Registrar tudo para auditoria** (erros, descartes, métricas e previsões finais por data/produto).

Essa abordagem segue o princípio da teoria de modelagem preditiva:

- **Dados de qualidade** vêm antes de modelo complexo.
- **Baseline simples** é obrigatório para provar valor real de modelos avançados.
- **Reprodutibilidade** é essencial: duas execuções com mesmas condições devem gerar resultados comparáveis.
- **Auditabilidade** é parte da solução: se algo for descartado/corrigido, precisa ficar registrado.

---

## 7) Fluxo completo: do arquivo de entrada ao registro final

Esta seção descreve o caminho dos dados de ponta a ponta, sem foco em código.

### Etapa A — Entrada (dados brutos)

- Origem: pasta `Dados/raw/`.
- Organização esperada: mercado → categoria → arquivos CSV de produtos.
- Problema típico nessa fase: formatos diferentes, colunas ausentes e valores inconsistentes.

### Etapa B — Ingestão e padronização inicial

- Arquivo principal: `src/data/load_raw_data.py`.
- O que acontece:
  1. leitura dos CSVs brutos;
  2. padronização dos nomes de colunas;
  3. validação de campos essenciais (como data e variáveis de negócio);
  4. tratamento inicial de valores inválidos;
  5. geração de saídas consolidadas por categoria.
- Saída: `Dados/processed/`.
- Auditoria: descartes dessa fase são registrados em `discarded_records_raw_ingestion.csv`.

### Etapa C — Pré-processamento e validação forte

- Arquivos principais:
  - `src/data/preprocess.py`
  - `src/data/validators.py`
- O que acontece:
  1. correção temporal (datas faltantes/irregulares);
  2. imputação de valores inválidos;
  3. tratamento de outliers;
  4. criação de atributos de calendário e atributos cíclicos.
- Saída: `Dados/preprocessed/`.
- Auditoria: descartes dessa fase vão para `discarded_records_preprocess.csv`.

### Etapa D — Engenharia de features e cenários

- Arquivos principais:
  - `src/utils/helpers.py` (lags e médias móveis)
  - `src/features/scenarios.py` (cenários como volume, preço, kmeans)
  - `src/config/scenario_params.py` (parâmetros de cenário)
- O que acontece:
  - o histórico de vendas vira variáveis explicativas;
  - o pipeline pode ser executado em cenários diferentes para comparar comportamento.

### Etapa E — Treino e previsão dos modelos

- Arquivos principais:
  - `src/models/baseline.py`
  - `src/models/xgboost_model.py`
  - `src/models/lstm_model.py`
  - `src/models/gru_model.py`
- O que acontece:
  - cada modelo recebe dados padronizados da etapa anterior;
  - treina com parâmetros centralizados;
  - gera métricas agregadas e também curva temporal de previsão.
- Saídas:
  - métricas por modelo/mercado;
  - arquivos `*_predictions.csv` com `date`, `product_id`, `y_true`, `y_pred`, `scenario`.

### Etapa F — Orquestração, consolidação e registro final

- Orquestrador principal: `main.py`.
- O que ele faz:
  1. executa os modelos por cenário;
  2. aplica filtros de data quando necessário;
  3. controla timeout/retry;
  4. coleta métricas;
  5. registra falhas sem derrubar todo o pipeline.
- Registro final:
  - logs de erro (`Resultados/errors.log`);
  - resultados por cenário/modelo;
  - consolidação final via utilitários (ex.: agregação/plots).

---

## 8) Métodos usados e por que fazem sentido para leigos

Para ficar claro para qualquer leitor:

1. **Baseline**
   - É a referência simples.
   - Se um modelo avançado não superar a baseline, ele não agrega valor prático.

2. **XGBoost**
   - É muito bom em padrões não lineares com dados tabulares.
   - Costuma entregar bom equilíbrio entre desempenho e custo computacional.

3. **LSTM e GRU**
   - São modelos de redes neurais para sequências temporais.
   - Capturam dependências ao longo do tempo melhor que muitos métodos clássicos.
   - A GRU costuma ser mais leve; a LSTM pode capturar memória temporal mais rica.

4. **Heurísticas de qualidade de dados**
   - Antes de prever, é preciso garantir que os dados façam sentido.
   - Por isso existem regras para detectar zeros suspeitos, outliers e problemas de calendário.

---

## 9) Resumo final para leitura rápida

Se alguém nunca viu o projeto, pode entender assim:

- **Entrada:** CSVs brutos em `Dados/raw`.
- **Meio:** limpeza + validação + criação de variáveis + treino de modelos.
- **Saída:** métricas, previsões por data/produto e logs de auditoria.
- **Planejamento:** tudo foi estruturado para ser reproduzível, comparável e auditável.

Isso garante que o resultado não dependa de “tentativa aleatória”, mas de um processo
planejado de ponta a ponta, desde o arquivo de entrada até o registro final.
