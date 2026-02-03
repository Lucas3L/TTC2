import pandas as pd
import numpy as np

  # --- Coreção de datas temporais ---

# Detectar datas faltantes e marcar como anomalias caso exceda o limite
def corrigir_datas_temporais(df, max_faltantes=2, anomalias=None):
    if anomalias is None:
        anomalias = []

    # Copia e converte a coluna Date para datetime
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Lista para novas linhas a serem adicionadas
    novos = []

    # Separação de informações por pruduto
    for product_id, g in df.groupby("product_id"):
        # Ordena por data
        g = g.sort_values("Date")
        # Extrai as datas
        datas = g["Date"]

        # Criação de sequencia de datas
        esperado = pd.date_range(
            # inicio
            start=datas.min(),
            # fim
            end=datas.max(),
            # frequência 
            freq="D"
        )
        # Identificação de datas faltantes
        faltantes = esperado.difference(datas)

        # Tratamento conforme quantidade de datas faltantes
        if 0 < len(faltantes) <= max_faltantes:
            # Para cada data faltante, cria uma nova linha
            for data in faltantes:
                # Cópia da última linha do grupo
                linha = g.iloc[-1].copy()
                # Atualiza a data e marca como interpolada
                linha["Date"] = data
                # Marca a observação como interpolada
                linha["observation"] = "date_interpolated"

                # Zera os valores numéricos
                for c in ["Quantity", "UnitValue", "ProductCost"]:
                    # Define como NaN e zera o valor
                    if c in linha:
                        linha[c] = np.nan

                # Guarda a nova linha para adição posterior
                novos.append(linha)

        # Esse item excede o limite de datas faltantes
        elif len(faltantes) > max_faltantes:
            # fazer uma copia do grupo
            g = g.copy()
            # Marca a observação como anomalia severa
            g["observation"] = "date_gap_severe"
            # Adiciona todas as linhas do grupo como anomalias
            anomalias.extend(g.to_dict("records"))

    # Adiciona as novas linhas ao DataFrame original
    if novos:
        # Concatena os novos dados
        df = pd.concat([df, pd.DataFrame(novos)], ignore_index=True)

    # retorna o DataFrame corrigido e as anomalias detectadas
    return df, anomalias


    # --- Coreção de valores temporais ---

# Corrigir valores inválidos (negativos ou nulos) usando média móvel
def corrigir_valores_temporais(
    df,
    coluna,
    window=7,
    anomalias=None
):
    # Inicializa a lista de anomalias se não fornecida
    if anomalias is None:
        anomalias = []

    # Ordena o DataFrame por product_id e Date
    df = df.sort_values(["product_id", "Date"]).copy()

    # Processa cada grupo de product_id
    for product_id, g in df.groupby("product_id"):
        # Obtém os valores da coluna específica
        valores = g[coluna]

        # Itera sobre os índices do grupo
        for idx in g.index:
            # Obtém o valor atual
            valor = df.at[idx, coluna]

            # Se o valor for válido, continua
            if pd.notna(valor) and valor > 0:
                continue

            # Define o intervalo de contexto para cálculo da média móvel
            pos = g.index.get_loc(idx)

            # Calcula os índices de início e fim do contexto
            ini = max(0, pos - window)
            fim = min(len(g), pos + window + 1)

            # Extrai o contexto para cálculo da média móvel
            contexto = valores.iloc[ini:fim].dropna()

            # Se não houver contexto suficiente, marca como anomalia
            if len(contexto) < 5:
                # Marca como anomalia sem contexto suficiente
                df.at[idx, "observation"] = f"{coluna}_invalid_no_context"
                # Adiciona a anomalia à lista
                anomalias.append(df.loc[idx].to_dict())
                continue

            # Calcula a média do contexto
            media = contexto.mean()

            # Se a média for positiva, corrige o valor
            if media > 0:
                df.at[idx, coluna] = abs(media)
                df.at[idx, "observation"] = f"{coluna}_corrected_context"
            # Se a média não for positiva, marca como anomalia severa
            else:
                df.at[idx, "observation"] = f"{coluna}_invalid_severe"
                anomalias.append(df.loc[idx].to_dict())

    return df, anomalias
