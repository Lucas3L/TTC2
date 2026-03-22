# Avaliação de utilidade dos notebooks

Esta avaliação mede utilidade prática para banca/reprodutibilidade.

## Resumo

| Notebook | Células | Markdown | Código | Executadas | Outputs | Score | Recomendação |
|---|---:|---:|---:|---:|---:|---:|---|
| `01_analise_exploratoria.ipynb` | 15 | 5 | 10 | 0 | 0 | 70 | refatorar |
| `02_processamento.ipynb` | 16 | 5 | 11 | 0 | 0 | 70 | refatorar |
| `03_kmeans.ipynb` | 15 | 5 | 10 | 0 | 0 | 70 | refatorar |
| `04_modelagem.ipynb` | 18 | 5 | 13 | 0 | 0 | 70 | refatorar |
| `05_resultados_e_analise.ipynb` | 14 | 5 | 9 | 0 | 0 | 70 | refatorar |

## Diagnóstico

- Todos os notebooks com score baixo devem ser **refatorados** antes de uso em banca.
- Células de texto atualmente como código (comentários) devem virar markdown.
- Adicionar no fim de cada notebook: versão de dados, seed, e célula de exportação de artefatos.

## Recomendação final

- **Não excluir neste momento**: os notebooks representam as etapas do trabalho (EDA, processamento, clusterização, modelagem, resultados).
- **Refatorar estrutura** para leitura acadêmica e rastreabilidade.
- Se após refatoração algum notebook continuar redundante (ex.: clusterização não usada no experimento final), mover para `notebooks/archive/`.
