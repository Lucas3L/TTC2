TCC2 - Fluxo de Processamento de Séries Temporais no Varejo
Estrutura Principal
Dados/raw/: entrada bruta por mercado/categoria/produto.

src/data/load_raw_data.py: consolida os CSVs brutos em Dados/processed/market/catXX.csv.

src/data/preprocess.py: corrige datas/valores/outliers e gera features temporais em Dados/preprocessed/.

main.py: orquestrador central de experimentos isolados por cenário e modelo.

src/models/*.py: arquitetura de treinamento e avaliação padronizada (LSTM, GRU, XGBoost, Baseline).

Resultados/: saída consolidada por cenário/modelo, predições diárias e logs de erro.

A Arquitetura de Treinamento Global
Uma das principais inovações deste projeto é a padronização de todos os algoritmos sob uma Arquitetura de Treinamento Global Consciente de Cenário.

Em vez de treinar centenas de modelos isolados para cada produto, todos os modelos (XGBoost, LSTM e GRU) concatenam o histórico de todos os SKUs em grandes matrizes e treinam um único modelo por mercado.
Para que os algoritmos consigam diferenciar o comportamento de um produto barato de um caro, os Cenários (Volume, Price, K-Means) são convertidos numericamente (LabelEncoder) e injetados na lista de features da rede neural. Isso permite ao modelo compartilhar o aprendizado de sazonalidade entre produtos similares sem misturar comportamentos opostos.

Etapas de Execução
1. Ingestão e Padronização (load_raw_data.py)
Lê os dados de Dados/raw.

Normaliza o schema obrigatório (date, quantity, unitvalue, productcost).

Trata zeros suspeitos baseados no contexto da média móvel local, preservando zeros reais (como fechamentos aos domingos).

Gera variáveis cíclicas temporais (month_sin, dow_cos, etc.).

2. Pré-processamento Robusto (preprocess.py)
Mapeia as lacunas temporais. Preenche buracos curtos e isola buracos longos.

Trata outliers severos via IQR (Q1/Q3), substituindo valores aberrantes pela mediana do produto para não distorcer o treinamento das redes neurais.

Salva o dado final pronto para modelagem matemática em Dados/preprocessed/.

3. Orquestração de Experimentos (main.py)
Gerencia a execução em subprocessos (isolamento de memória contra falhas de C++ no Keras/XGBoost).

Roda os cenários habilitados (volume, price, kmeans) com controle de sementes (seed) para garantir reprodutibilidade científica.

Extrai e consolida as métricas reais do terminal (sMAPE, MAE, RMSE) calculando a média global de todos os mercados processados.

Trata timeouts (configuráveis via --model-timeout-sec) para treinamentos densos de Deep Learning.

4. Treino e Avaliação (A Paridade Científica)
Todos os scripts dentro de src/models/ operam sob Regras Estritas de Paridade:

O mesmo Filtro: Todos os modelos aplicam o mesmo dropna para lags e a remoção de sentinelas (-99.0). Isso garante que o Baseline e as Redes Neurais façam previsões exatamente para os mesmos dias, evitando vantagens estatísticas.

A mesma Janela: Divisão temporal travada em 70% Treino, 15% Validação e 15% Teste por produto, preservando a cronologia de cada SKU.

Os Modelos
1. Baseline Zero-Aware (src/models/baseline_zero_aware.py)
Estratégia: Modelo Naive (Ingênuo). Assume que a venda de amanhã é igual à venda de hoje (y_pred = lag_1).

Objetivo: Estabelecer o piso de performance. Na indústria, modelos complexos de ML só têm valor se conseguirem superar essa estratégia de custo zero.

Rigor: Passou a sofrer os mesmos cortes de dados incompletos que a Inteligência Artificial para garantir justiça na comparação de métricas.

2. XGBoost (src/models/xgb_model.py)
Estratégia: Árvore de Decisão Gradiente Global. Treina a base inteira num formato tabular (2D), aproveitando as colunas de cenário e histórico residente (lags).

Performance: Apresenta a melhor relação de Custo-Benefício do projeto. Treina a loja inteira em segundos, isola ruídos com eficácia e não sofre com underfitting crônico, resultando nos menores erros absolutos (MAE).

3. Deep Learning: LSTM e GRU (src/models/lstm_model.py | gru_model.py)
Estratégia: Redes Neurais Recorrentes Globais que utilizam Camadas de Embedding para criar vetores independentes para cada SKU e Cenário.

Hiperparâmetros Agressivos: Para evitar o underfitting (onde a rede prevê apenas uma onda suave ignorando os picos), ambos os modelos utilizam loss='mse' (para penalizar severamente erros em picos) e batch_size=32 (forçando atualizações rápidas de peso durante a volatilidade diária).

Complexidade: Exigem alto tempo de processamento (horas), convergem os dados em matrizes sequenciais em 3D (WINDOW=7) e podem apresentar viés de superestimação (Overprediction Bias) quando submetidos a pesos de amostragem punitivos.

Métricas e a "Armadilha do Zero" (Relevância Acadêmica)
A avaliação da performance em séries temporais de varejo intermitente (com muitos dias sem vendas) sofre do fenômeno estatístico conhecido como Armadilha do Zero no sMAPE.

Embora o projeto calcule e reporte o sMAPE (Symmetric Mean Absolute Percentage Error), esta métrica pune de forma desproporcional modelos contínuos (Redes Neurais). Se a venda real de um dia for 0 e o modelo prever 1 única unidade, a matemática do sMAPE devolve um erro diário de 200%.

Isso resulta em médias percentuais superiores a 120% para LSTM/GRU, mascarando a real capacidade preditiva dos algoritmos. Por este motivo, a avaliação final e o veredito de negócio são baseados em:

MAE (Mean Absolute Error): Reflete com exatidão o número médio de "unidades" que o modelo errou por dia.

WAPE (Weighted Absolute Percentage Error): Calculado no pós-processamento utilizando os arquivos _predictions.csv, diluindo os erros unitários sobre o volume total do período, fornecendo o risco real de ruptura ou excesso de estoque.

Como Executar o Pipeline Final
Garanta que a base limpa existe: python src/data/preprocess.py

Dispare o orquestrador com margem de tempo para o Deep Learning:

Bash
python main.py --seed 142 --max-retries 0 --model-timeout-sec 86400
Acompanhe a evolução do treino em Resultados/errors.log e analise a saída consolidada nos arquivos .csv.