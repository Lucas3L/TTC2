# Plano de implementação — melhorias de prioridade média

Este documento organiza a implementação das melhorias de prioridade média após os ajustes urgentes no orquestrador.

## Status atual

- ✅ Fase 1 iniciada: criação de utilitário comum para bootstrap de path/imports (`src/utils/project_paths.py`).
- ✅ Fase 1 iniciada: robustez da agregação para cenário sem arquivos de resultado.
- 🔜 Próximo passo: aplicar o bootstrap comum no restante dos scripts fora do escopo de modelos, se necessário.

## Objetivo

Reduzir débito técnico e custo de manutenção sem alterar a lógica principal de modelagem já validada.

## Escopo das melhorias médias

1. **Padronização de bootstrap de path/imports nos modelos**
   - Extrair o bloco repetido de `Path(...)/sys.path` para uma função utilitária comum.
   - Aplicar em `baseline.py`, `gru_model.py`, `lstm_model.py` e `xgboost_model.py`.

2. **Fortalecer agregação de resultados**
   - Tratar caso de ausência de arquivos/linhas para evitar erro de `pd.concat` vazio.
   - Emitir aviso amigável quando não houver resultados disponíveis.

3. **Configuração de execução por ambiente**
   - Mover defaults sensíveis (timeout, retries, diretórios) para config central.
   - Permitir override por CLI e por variável de ambiente.

4. **Higienização incremental da documentação**
   - Separar documentação operacional curta (quickstart) da documentação técnica extensa.
   - Manter README principal enxuto e com links para docs específicos.

## Backlog técnico sugerido

| Item | Entrega | Risco | Estimativa |
|---|---|---|---|
| Bootstrap comum de path/imports | utilitário + refactor dos modelos | baixo | 0.5 dia |
| Guard-clauses em agregação | validação + mensagens de erro claras | baixo | 0.5 dia |
| Config por ambiente | novos campos em `src/config/*` | médio | 1 dia |
| Reorganização da documentação | split do README + índice em `docs/` | baixo | 0.5 dia |

## Ordem de implementação (recomendada)

1. Agregação robusta (rápido e com ganho imediato).
2. Bootstrap comum de modelos (reduz duplicação).
3. Configuração por ambiente.
4. Reorganização da documentação.

## Critérios de aceite

- Nenhum script deve quebrar por ausência de arquivos de resultados.
- Redução de duplicação visível nos modelos (trechos idênticos removidos).
- Novas opções de configuração funcionando sem regressão.
- README principal com menos ruído e links claros para documentação detalhada.
