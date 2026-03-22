# TTC2 - Fluxo de processamento

## Estrutura principal

- `Dados/raw/`: entrada bruta por mercado/categoria/produto.
- `src/data/load_raw_data.py`: consolida os CSVs brutos em `Dados/processed/market/catXX.csv`.
- `src/data/preprocess.py`: corrige datas/valores/outliers e gera features temporais em `Dados/preprocessed/`.
- `main.py`: orquestra experimentos por cenário e modelo.
- `src/models/*.py`: treinamento e avaliação por modelo (LSTM, GRU, XGBoost, Baseline).
- `Resultados/`: saída consolidada por cenário/modelo e logs de erro.

## Etapas (arquivo de entrada até o fim do `main.py`)

1. **Ingestão e padronização** (`load_raw_data.py`)
   - Lê dados de `Dados/raw`.
   - Normaliza nomes de colunas e garante `date`, `quantity`, `unitvalue`, `productcost`.
   - Corrige valores inválidos em `quantity` (negativos/NaN) e também zeros suspeitos por contexto local; em `unitvalue`, corrige <=0/NaN com média móvel por produto.
   - Gera features calendáricas iniciais.
   - Salva arquivos por mercado/categoria em `Dados/processed`.

2. **Pré-processamento robusto** (`preprocess.py` + `validators.py`)
   - Corrige lacunas de datas por produto (`corrigir_datas_temporais`).
   - Corrige valores inválidos (`corrigir_valores_temporais`).
   - Trata outliers por IQR (`tratar_outliers_iqr_por_produto`).
   - Recalcula features de calendário/cíclicas.
   - Salva em `Dados/preprocessed`.

3. **Orquestração de experimentos** (`main.py`)
   - Define seed global e lista de cenários: `volume`, `price`, `kmeans`.
   - Para cada cenário e para cada modelo (`LSTM`, `GRU`, `XGBoost`, `Baseline`):
     - Executa script de modelo via subprocesso (`python src/models/... --seed ... --scenario ...`).
     - Coleta saída de terminal.
     - Extrai `FINAL sMAPE`, `MAE`, `RMSE` via regex.
     - Registra em `Resultados/<cenario>_results.csv`.
   - Se falhar, escreve detalhes em `Resultados/errors.log` e continua pipeline.
   - Ao final, tenta gerar gráficos com `src/utils/plots.py`.

4. **Treino/avaliação por modelo** (`src/models/*.py`)
   - Cada modelo consome `Dados/preprocessed/<market>/cat*.csv`.
   - Opcionalmente aplica cenário (`src/features/scenarios.py`).
   - Cria lags e médias móveis (`src/utils/helpers.py`).
   - Faz split temporal treino/validação/teste por produto.
   - Treina e calcula métricas (`src/models/evaluate.py`).
   - Salva arquivos de saída em `Resultados/<modelo>/...` e imprime `FINAL sMAPE`.

## Observação importante

`main.py` espera que os dados em `Dados/preprocessed` já existam. Ou seja, o fluxo completo recomendado é:

`Dados/raw` → `load_raw_data.py` → `Dados/processed` → `preprocess.py` → `Dados/preprocessed` → `main.py`.


### Detalhamento completo da Etapa 1 (Ingestão e padronização)

A etapa 1 é executada pelo script `src/data/load_raw_data.py` e tem como objetivo transformar o formato bruto da coleta em um formato tabular único por categoria, pronto para limpeza avançada.

1. **Define origem e destino**
   - Origem: `Dados/raw`.
   - Destino: `Dados/processed`.

2. **Itera por mercado e por categoria**
   - Para cada pasta de mercado encontrada em `Dados/raw`, cria uma pasta espelho em `Dados/processed`.
   - Dentro de cada mercado, percorre as pastas de categoria.

3. **Lê cada CSV de produto da categoria**
   - Extrai `product_id` a partir do nome do arquivo.
   - Faz leitura com parse de data.
   - Se um arquivo estiver corrompido/ilegível, registra erro no terminal e segue para o próximo (não para o pipeline).

4. **Normaliza estrutura de colunas**
   - Converte nomes para snake_case minúsculo e remove caracteres problemáticos.
   - Garante a existência de `date`; caso ausente, o arquivo é descartado.
   - Se `productcost` não existir, cria a coluna com valor nulo para manter schema estável.

5. **Enriquece com metadados de contexto**
   - Adiciona `product_id`, `category` e `market` em cada linha para rastreabilidade posterior.

6. **Concatena todos os produtos da categoria**
   - Junta os DataFrames da categoria em um único DataFrame `df_categoria`.
   - Reaplica normalização e ordena por `product_id` e `date`.

7. **Valida schema mínimo obrigatório**
   - Exige as colunas: `date`, `quantity`, `unitvalue`, `productcost`, `product_id`, `category`, `market`.
   - Se faltar alguma, lança erro para evitar propagar dado estruturalmente inválido.

8. **Corrige valores inválidos iniciais**
   - Em `quantity`, valores negativos/NaN são inválidos e zeros passam por heurística de contexto: se a média local sem zeros (janela centrada de 5 pontos) for alta, o zero é tratado como suspeito e imputado, exceto em domingos e feriados (quando disponíveis), onde zero é considerado plausível por fechamento.
   - Em `unitvalue`, valores `<= 0` ou NaN são inválidos e recebem imputação.
   - O preenchimento usa média móvel (janela de 7 dias) por `product_id`.

9. **Gera features temporais iniciais**
   - `month`, `day_of_week` e variáveis cíclicas (`month_sin`, `month_cos`, `dow_sin`, `dow_cos`).

10. **Grava saída da categoria**
    - Salva em `Dados/processed/<mercado>/cat<codigo>.csv`.
    - Libera memória com `gc.collect()` antes de seguir para próxima categoria/mercado.

> Em resumo: a Etapa 1 **padroniza + consolida + valida** os dados brutos para garantir que a Etapa 2 receba arquivos homogêneos e consistentes.


### Detalhamento completo da Etapa 2 (Pré-processamento forte em `Dados/preprocessed`)

A etapa 2 é executada por `src/data/preprocess.py` com apoio de `src/data/validators.py`. O objetivo é corrigir anomalias temporais e estatísticas antes do treino dos modelos.

1. **Origem e destino**
   - Lê arquivos de `Dados/processed`.
   - Salva resultados em `Dados/preprocessed`.

2. **Loop por mercado e categoria**
   - Para cada mercado, cria pasta de saída correspondente.
   - Processa todos os arquivos `cat*.csv`.

3. **Inicialização de trilha de qualidade**
   - Cria coluna `observation = "ok"`.
   - Inicializa lista `anomalias` para registrar correções e casos severos.

4. **Correção de lacunas de datas (`corrigir_datas_temporais`)**
   - Para cada `product_id`, compara datas observadas com calendário diário esperado.
   - Se faltam poucas datas (até `max_faltantes=2`), insere linhas interpoladas com `observation = date_interpolated` e campos numéricos nulos para posterior imputação robusta.
   - Se faltam muitas datas, marca o grupo como `date_gap_severe` em anomalias para rastreabilidade (sem inventar séries longas).

5. **Correção de valores inválidos (`corrigir_valores_temporais`)**
   - Aplicada a `quantity`, `unitvalue` e `productcost`.
   - Em `quantity`: corrige negativos/NaN e zeros suspeitos por contexto local; preserva zeros plausíveis (domingos/feriados quando disponíveis).
   - Em `unitvalue`/`productcost`: corrige `<= 0` ou `NaN`.
   - Imputação por média móvel por produto (janela 7, vetorizado); fallback com média global do produto e, em caso extremo, preenchimento com 0 + marcação de severidade.

6. **Tratamento de outliers (`tratar_outliers_iqr_por_produto`)**
   - Também aplicado a `quantity`, `unitvalue` e `productcost`.
   - Calcula limites por IQR (Q1/Q3) por produto e substitui extremos pela mediana do produto.
   - Registra linhas afetadas em `anomalias`.

7. **Feature engineering temporal final**
   - Recalcula recursos derivados de data:
     - `month`,
     - `is_weekend`,
     - `day_sin`, `day_cos`,
     - `month_sin`, `month_cos`.

8. **Persistência e rastreabilidade**
   - Salva o CSV limpo em `Dados/preprocessed/<mercado>/catXX.csv`.
   - Se houver anomalias, grava `anomalies_<cat>.csv` no mesmo mercado.

> Em resumo: a Etapa 2 transforma dados apenas “padronizados” em dados **modeláveis**, com lacunas temporais tratadas, outliers controlados e sinais temporais consistentes.


## Observação sobre gráfico histórico (real vs predito)

- Os CSVs em `Dados/processed/` e `Dados/preprocessed/` possuem histórico temporal (`date`) e valores reais (`quantity`), então servem para gráficos de histórico real.
- Além dos arquivos de métricas agregadas (`mae`, `rmse`, `smape`), os modelos agora também salvam arquivos de curva com previsão por data em `Resultados/<modelo>/*_predictions.csv`.
- Esses arquivos de previsão incluem, por registro temporal, colunas como `date`, `product_id`, `y_true`, `y_pred` e `scenario`, permitindo plotar diretamente **real vs predito ao longo do tempo**.


### Detalhamento completo da Etapa 3 (Orquestração em `main.py`)

A etapa 3 é o "controlador" dos experimentos. O `main.py` não treina diretamente os modelos; ele coordena execução, coleta métricas, registra resultados e tenta gerar gráficos no fim.

1. **Configuração global do experimento**
   - Define `RANDOM_SEED`, `N_REPLICAS` e lista de cenários (`volume`, `price`, `kmeans`).
   - Define também o mapeamento de nomes de modelo para scripts (`lstm_model.py`, `gru_model.py`, `xgboost_model.py`, `baseline.py`).

2. **Determinismo/Reprodutibilidade**
   - `set_global_seed` fixa `PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS`, seed do `random`, `numpy` e `tensorflow`.

3. **Loop principal de execução**
   - Itera por réplica (atualmente 1).
   - Para cada cenário, executa todos os modelos mapeados.
   - Cada execução ocorre via subprocesso (`python <script> --seed ... --scenario ...`).

4. **Coleta e parsing de métricas**
   - Junta `stdout` + `stderr` do subprocesso.
   - Extrai `FINAL sMAPE`, `MAE`, `RMSE` com regex (`extract_metrics`).
   - Marca status (`OK` ou `ERRO`) conforme código de retorno do processo filho.

5. **Persistência de resultados por cenário**
   - Registra cada execução em `Resultados/<cenario>_results.csv` com:
     `timestamp, model, replica, seed, smape, mae, rmse, runtime_sec, status`.

6. **Gestão de falhas sem interromper o pipeline**
   - Se um modelo falhar, grava detalhes completos em `Resultados/errors.log`.
   - O loop continua para os demais modelos/cenários (tolerância a falhas parciais).

7. **Pós-processamento automático**
   - Ao final, tenta chamar `src/utils/plots.py` para gerar gráficos em `Resultados/plots/`.
   - Se falhar, emite aviso sem abortar a execução já concluída.

> Em resumo: `main.py` funciona como um orquestrador resiliente e reproduzível, responsável por rodar todos os cenários/modelos, consolidar métricas e manter logs operacionais.


## Configuração centralizada de experimento

Para evitar espalhar parâmetros entre vários scripts, existe um arquivo central:
- `src/config/experiment_config.py`

Nele você pode ajustar:
- `random_seed`
- `n_replicas`
- `scenarios`
- `models` (mapa nome -> script)
- `date_from` e `date_to` (faixa global opcional)

Além disso, o `main.py` aceita `--date-from` e `--date-to` e repassa automaticamente para todos os modelos.


## Modelos — Parte 1: Baseline (`src/models/baseline.py`)

O Baseline é o modelo de referência mais simples do projeto. Ele não aprende pesos; apenas usa a última observação disponível (`lag_1`) como previsão do próximo ponto.

### Objetivo
- Servir de piso de comparação para os modelos mais complexos (LSTM, GRU, XGBoost).
- Detectar rapidamente se um modelo avançado realmente traz ganho sobre uma estratégia ingênua.

### Entradas
- Lê dados em `Dados/preprocessed/<market>/cat*.csv`.
- Aceita filtros opcionais de faixa temporal (`--date-from`, `--date-to`) e cenário (`--scenario`, apenas propagado para rastreio).

### Pipeline interno
1. Carrega CSV e normaliza colunas.
2. Remove linhas sem alvo (`quantity`).
3. Ordena por `product_id` e `date`.
4. Para cada produto:
   - cria lags (`lag_1`, `lag_7`, `lag_14`),
   - mantém apenas linhas completas para lags + alvo,
   - usa divisão temporal padronizada 70/15/15 (treino/validação/teste) e avalia no bloco final de teste (15%),
   - prevê `y_pred = lag_1`.
5. Calcula MAE/RMSE/sMAPE com `evaluate`.
6. Salva:
   - métricas agregadas por arquivo/produto,
   - curva temporal `*_predictions.csv` com `date`, `product_id`, `y_true`, `y_pred`, `scenario`.

### Forças
- Simples, rápido e interpretável.
- Excelente para validar se o pipeline inteiro está coerente.

### Limitações
- Não captura sazonalidade complexa nem efeitos de promoção/preço.
- Em séries com mudanças bruscas, tende a atrasar a resposta (por usar só o último ponto).


## Modelos — Parte 2: GRU (`src/models/gru_model.py`)

O GRU é um modelo recorrente neural para séries temporais. Aqui ele usa sequências de janelas fixas com embedding de produto para aprender padrões compartilhados entre SKUs.

### Objetivo
- Capturar dependências temporais além do baseline.
- Treinar um único modelo global, aprendendo comportamento por produto via embedding.

### Entradas
- Dados de `Dados/preprocessed/<market>/cat*.csv`.
- Filtros opcionais: `--date-from` e `--date-to`.
- Cenário opcional (`--scenario`) aplicado antes da geração de features.

### Pipeline interno
1. Carrega dados, normaliza colunas e remove alvo ausente.
2. Aplica cenário (quando informado).
3. Faz label encoding de `product_id`.
4. Gera lags e médias móveis (`lag_1`, `lag_7`, `rolling_mean_3`, `rolling_mean_7`, `rolling_mean_14`).
5. Split temporal por produto (70% treino, 15% validação, 15% teste) e concatena para treino global.
6. Escala features com `MinMaxScaler`.
7. Monta sequências com janela `WINDOW=7` (padronizada) e preserva datas do alvo.
8. Treina GRU com early stopping monitorando `val_loss`.
9. Prediz em teste, calcula MAE/RMSE/sMAPE e exporta curva real vs predito.

### Arquitetura
- Entrada temporal (`window x features`) + entrada de id de produto.
- Duas camadas GRU (64 -> 32), dropout, embedding de produto, concatenação e camadas densas.
- Saída `softplus` para manter previsão não-negativa.

### Saídas
- Métricas agregadas: `Resultados/gru/<market>_gru_<scenario>.csv`.
- Curva temporal: `Resultados/gru/<market>_gru_<scenario>_predictions.csv` com `date`, `product_id`, `y_true`, `y_pred`, `scenario`.

### Forças
- Aprende padrões temporais não lineares.
- Compartilha aprendizado entre produtos com embedding.

### Limitações
- Custo computacional superior ao baseline e XGBoost em alguns cenários.
- Sensível a volume mínimo de dados (há cláusulas de guarda de amostra).


> Nota: Para comparabilidade, os modelos estão alinhados na proporção temporal 70/15/15 quando aplicável.


## Modelos — Parte 3: LSTM (`src/models/lstm_model.py`)

O LSTM é um modelo recorrente com memória de longo prazo. No projeto, ele é treinado por produto (com embedding de ID) e inclui mecanismo de atenção aditiva para reforçar padrões relevantes da janela temporal.

### Objetivo
- Capturar relações temporais de curto e médio prazo em séries de demanda.
- Melhorar robustez para padrões sazonais e intermitência usando features de lag + embedding.

### Entradas
- `Dados/preprocessed/<market>/cat*.csv`.
- Filtros opcionais de período (`--date-from`, `--date-to`).
- Cenário opcional (`--scenario`) aplicado antes do preparo final.

### Pipeline interno
1. Carrega dados, normaliza colunas, remove alvo ausente e aplica filtro temporal.
2. Aplica cenário (`volume`, `price`, `kmeans`) quando informado.
3. Mapeia `product_id` para índice numérico e mantém mapa inverso para exportação.
4. Para cada produto:
   - normaliza `quantity` por `market_max`,
   - gera lags/médias móveis,
   - seleciona features válidas,
   - split temporal 70/15/15.
5. Gera sequências com janela `WINDOW=7`.
6. Treina LSTM com `EarlyStopping` em validação.
7. Prediz teste, reescala para unidade real, calcula MAE/RMSE/sMAPE e salva curva temporal.

### Arquitetura
- Entrada temporal + entrada de ID do produto.
- Blocos: `LSTM(64, return_sequences=True)` -> `Dropout(0.2)` -> `AdditiveAttention` -> `LSTM(32)`.
- Embedding de produto (`Embedding(..., output_dim=16)`) concatenado ao estado temporal.
- Cabeça densa e saída `softplus` para evitar previsões negativas.

### Saídas
- Métricas agregadas: `Resultados/lstm/<market>_lstm_<scenario>.csv`.
- Curva temporal: `Resultados/lstm/<market>_lstm_<scenario>_predictions.csv` com `date`, `product_id`, `y_true`, `y_pred`, `scenario`.

### Forças
- Boa capacidade de modelar dependências temporais e não linearidades.
- Atenção aditiva pode melhorar foco em trechos relevantes da janela.

### Limitações
- Treino mais custoso que baseline/XGBoost em hardware limitado.
- Sensível a séries muito curtas (há cláusulas mínimas por produto).


### Esclarecimento do passo 4 (LSTM)

No passo 4 do LSTM, o processamento ocorre **por produto** para evitar vazamento temporal entre séries diferentes:
- Primeiro calcula `market_max` (máximo global de `quantity` no mercado) para normalizar a escala.
- Depois, para cada produto, aplica:
  - clipping/normalização de `quantity`,
  - clipping de `unitvalue` (quando existe),
  - criação de lags e rolling means,
  - limpeza de `NaN` nas features escolhidas,
  - split temporal 70/15/15 dentro do próprio produto.

Isso garante que cada SKU tenha histórico tratado de forma consistente antes de gerar sequências.

### Diferenças de código: LSTM vs GRU (e por quê)

1. **Estratégia de treino**
   - LSTM: itera produto a produto e treina em cada grupo dentro do arquivo.
   - GRU: concatena produtos e treina um modelo global (com embedding) em um único fluxo.
   - **Por quê:** GRU foi desenhado para compartilhar sinal entre SKUs; LSTM está mais conservador por produto.

2. **Janela temporal**
   - LSTM e GRU usam `WINDOW=7` (padronizado em `src/config/model_params.py`).
   - **Por quê:** manter comparabilidade entre modelos e facilitar ajustes globais de experimento.

3. **Função de perda**
   - LSTM compila com `loss='mae'`.
   - GRU compila com `loss='poisson'`.
   - **Por quê:** GRU está orientado a contagens/intermitência; LSTM foi configurado para erro absoluto robusto.

4. **Bloco de atenção**
   - LSTM inclui `AdditiveAttention` entre camadas recorrentes.
   - GRU não usa atenção explícita.
   - **Por quê:** LSTM tenta destacar partes mais relevantes da janela temporal.

5. **Escalonamento de features**
   - GRU aplica `MinMaxScaler` explícito antes das sequências.
   - LSTM usa normalização por `market_max` + clipping e não aplica MinMax global da mesma forma.
   - **Por quê:** escolhas de desenho de cada script; ambos funcionam, mas a dinâmica de escala é diferente.


## Padronização global de janela temporal dos modelos

Para garantir consistência entre Baseline, GRU, LSTM e XGBoost, os parâmetros comuns dos modelos foram centralizados em:
- `src/config/model_params.py`

Parâmetros compartilhados:
- `window_size` (janela temporal única para todos os modelos)
- `train_ratio`
- `val_ratio`
- `prediction_horizon_days`
- `lags`
- `rolling_windows`
- `features_base`
- `loss_by_model`
- `training_by_model` (batch, epochs, patience e parâmetros XGBoost)

Assim, ao alterar esses valores em um único arquivo, todos os modelos passam a usar a mesma configuração automaticamente.


> Observação: lags e janelas de médias móveis também foram centralizados para manter o pré-processamento homogêneo entre os modelos.


> Sugestão implementada: features base e funções de perda também foram centralizadas para facilitar comparações e ajustes globais.


## Padronização global de cenários

Os parâmetros dos cenários também foram centralizados em:
- `src/config/scenario_params.py`

Nele ficam configurações como:
- cenários habilitados (`enabled_scenarios`),
- quantis e rótulos de `volume` e `price`,
- parâmetros de clusterização do `kmeans` (número de clusters, seed, colunas de entrada).

Assim, ajustes de cenário são aplicados de forma uniforme em todos os modelos que usam `apply_scenario`.


## Desempenho e tempo de processamento dos modelos

### Como o tempo é medido hoje
- O `main.py` mede `runtime` de cada execução de modelo por cenário via `subprocess.run`.
- Esse tempo é salvo em `Resultados/<cenario>_results.csv` na coluna `runtime_sec`.

### Tendência de custo (do mais leve para o mais pesado, em geral)
1. **Baseline**: mais rápido (sem treino pesado, previsão por `lag_1`).
2. **XGBoost**: custo intermediário (árvores + early stopping).
3. **GRU / LSTM**: mais caros (redes recorrentes com várias épocas e validação).

### Principais alavancas para acelerar
- **Epochs**: reduzir `epochs` em GRU/LSTM (`training_by_model` no `model_params.py`).
- **Batch size**: aumentar lote pode reduzir tempo por época (trade-off de memória).
- **Amostragem**: reduzir período com `--date-from`/`--date-to`.
- **Produtos mínimos**: elevar filtros mínimos de amostra para pular séries muito curtas/ruidosas.
- **XGBoost**: reduzir `n_estimators` e/ou `max_depth`.
- **Cenários**: rodar menos cenários por vez em `enabled_scenarios`.

### Estratégia prática recomendada
- Use um perfil "rápido" para iteração (menos épocas e menos árvores).
- Valide pipeline e outputs.
- Rode perfil completo apenas em execução final.


### Impacto das otimizações no tempo x qualidade

Não há garantia matemática de manter exatamente a mesma qualidade ao reduzir custo, mas a prática recomendada é:

1. **Desenvolvimento rápido**
   - Reduzir `epochs` (GRU/LSTM) e reduzir `n_estimators`/`max_depth` (XGBoost).
   - Usar recorte temporal com `--date-from/--date-to` para validar pipeline e direção de melhoria.

2. **Treino final**
   - Retomar hiperparâmetros completos no `model_params.py`.
   - Executar período completo e comparar métricas (`smape`, `mae`, `rmse`) com baseline da execução anterior.

Regra prática:
- Se a diferença de sMAPE ficar dentro de uma margem pequena (ex.: <= 1-2 p.p.) para o seu negócio, o ganho de tempo geralmente compensa.
- Se houver degradação relevante, aumente gradualmente `epochs` ou árvores até encontrar o melhor compromisso.
