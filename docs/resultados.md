# Resultados

## 1. Avaliação dos Modelos

Os modelos foram avaliados utilizando métricas de erro aplicadas ao conjunto de teste.

As principais métricas consideradas foram:

- Erro Absoluto Médio (MAE)
- Erro Quadrático Médio (MSE)
- Raiz do Erro Quadrático Médio (RMSE)

Essas métricas permitem avaliar tanto a magnitude média do erro quanto penalizar erros maiores.

---

## 2. Impacto da Clusterização

A aplicação do K-Means demonstrou impacto positivo na organização dos dados.

Observou-se que:

- Produtos com comportamento semelhante foram agrupados corretamente.
- Modelos treinados por cluster apresentaram melhor estabilidade.
- Houve redução na variância dos erros dentro dos grupos.

---

## 3. Comparação entre Abordagens

Foram comparados dois cenários:

1. Modelo global (sem clusterização)
2. Modelo com clusterização prévia

De forma geral, o modelo com clusterização apresentou:

- Melhor capacidade de generalização
- Menor erro médio em determinados grupos
- Melhor adaptação a padrões específicos

---

## 4. Análise Crítica

Apesar dos bons resultados, algumas limitações foram observadas:

- Sensibilidade ao número de clusters
- Dependência da qualidade da normalização
- Variação de desempenho entre produtos com pouca informação histórica

Esses fatores indicam oportunidades para melhorias futuras.