# Sugestões para evolução do tratamento de dados e registro de descartes

Este guia reúne recomendações práticas para fortalecer qualidade de dados e auditabilidade do pipeline.

## Status de implementação atual

- ✅ Logs de descarte em `raw` e `preprocess` agora incluem `timestamp`, `severity`, `reason` e `rows_affected` (quando aplicável).
- ✅ `preprocess` passou a registrar `stage=preprocess` no schema de descarte.

## 1) Tratamento de dados (evoluções recomendadas)

1. **Imputação hierárquica por fallback**
   - Ordem sugerida para campos numéricos: média móvel por produto → mediana por produto → mediana por categoria → mediana global.
   - Evita que séries curtas fiquem com `NaN` residual ou preenchimento agressivo com zero.

2. **Classificação de severidade das anomalias**
   - Padronizar níveis: `info`, `warning`, `critical`.
   - Exemplos:
     - `warning`: zeros suspeitos imputados.
     - `critical`: ausência de colunas obrigatórias.

3. **Janela adaptativa para zeros suspeitos**
   - Em produtos com baixa frequência de venda, usar janela menor e limiar mais conservador.
   - Em produtos com alta recorrência, janela maior e limiar mais rígido.

4. **Validação de consistência cruzada**
   - `quantity == 0` com `unitvalue > 0` pode ser plausível, mas marcar para revisão quando ocorrer em excesso.
   - `productcost > unitvalue` pode indicar erro de origem dependendo da regra de negócio.

## 2) Registro de descartes (boas práticas)

1. **Schema único de descarte**
   - Campos mínimos:
     - `stage` (raw_ingestion/preprocess/model_input)
     - `market`, `category`, `file_path`/`csv_file`
     - `reason`
     - `rows_affected`
     - `timestamp`

2. **Razões padronizadas**
   - Ex.: `missing_date_column`, `missing_required_columns`, `read_error`, `empty_after_validation`.
   - Facilita dashboards de qualidade e análises históricas.

3. **Métrica de impacto**
   - Registrar percentual de linhas descartadas por arquivo e por mercado.
   - Definir alertas (ex.: >5% do arquivo descartado = investigar fonte).

4. **Trilha temporal**
   - Salvar logs por execução com timestamp no nome do arquivo para auditoria.
   - Ex.: `discarded_records_preprocess_YYYYMMDD_HHMMSS.csv`.

## 3) Próximos passos recomendados

1. Consolidar logs de descarte (`raw` + `preprocess`) em um único relatório diário.
2. Criar gráfico de top razões de descarte por mercado.
3. Criar KPI de qualidade por mercado/categoria para acompanhamento contínuo.
