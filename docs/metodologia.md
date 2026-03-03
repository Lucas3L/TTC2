# Metodologia

## 1. Visão Geral

Este trabalho propõe um modelo de previsão baseado em aprendizado de máquina aplicado a séries temporais de produtos. A abordagem combina técnicas de clusterização com modelos preditivos supervisionados, permitindo identificar padrões de comportamento semelhantes entre produtos antes da etapa de previsão.

O fluxo geral do sistema é composto por:

1. Carregamento dos dados
2. Pré-processamento
3. Clusterização com K-Means
4. Preparação das sequências temporais
5. Treinamento dos modelos
6. Avaliação e comparação

---

## 2. Coleta e Estrutura dos Dados

Os dados são organizados por produto e contêm informações históricas utilizadas para gerar janelas temporais (sliding windows), permitindo transformar o problema em aprendizado supervisionado.

Cada produto é tratado inicialmente de forma isolada para evitar mistura de padrões temporais.

---

## 3. Pré-processamento

O pré-processamento inclui:

- Remoção de valores inconsistentes
- Normalização das variáveis
- Organização cronológica
- Criação de janelas temporais (windowing)

Foi utilizada a técnica de janela deslizante, onde uma sequência de tamanho fixo é utilizada para prever o próximo valor da série.

---

## 4. Clusterização com K-Means

Antes da etapa de previsão, foi aplicado o algoritmo K-Means para agrupar produtos com comportamento semelhante.

Objetivos da clusterização:

- Identificar padrões de consumo semelhantes
- Reduzir variabilidade intra-grupo
- Permitir treinamento especializado por cluster

O número de clusters foi definido com base em análise exploratória e critérios de separação entre grupos.

---

## 5. Modelagem Preditiva

Após a clusterização, foram treinados modelos de previsão utilizando os dados organizados em formato de sequências temporais.

Os modelos implementados seguem a seguinte lógica:

- Entrada: sequência temporal normalizada
- Saída: valor futuro da série

O treinamento foi realizado utilizando divisão entre conjunto de treino e teste, garantindo avaliação imparcial.

---

## 6. Pipeline de Execução

O fluxo principal do sistema é controlado pelo `main.py`, que executa:

1. Carregamento global dos dados
2. Preparação das sequências
3. Aplicação do K-Means
4. Treinamento dos modelos
5. Avaliação de desempenho
6. Comparação entre abordagens

Esse fluxo garante reprodutibilidade e organização clara do experimento.