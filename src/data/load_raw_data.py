from pathlib import Path
import pandas as pd

# Filtragem apartir da pasta seguinte
BASE_PATH = Path("Dados/raw")

# Pasta onde serão salvos os novos dados
OUTPUT_BASE = Path("Dados/processed")

# Enquanto existir intens dentro da Base_path percorra 1 por 1
for market_path in BASE_PATH.iterdir():

    # Se o item não for uma pasta, ignore
    if not market_path.is_dir():
        continue

    # extração do nome da pasta para depois imprimir 
    market_name = market_path.name
    print(f"Processando {market_name}")

    # Caminho até a pasta de saidado onde serao tratados os dados 
    market_output = OUTPUT_BASE / market_name

    # Criação das pastas caso não existam
    market_output.mkdir(parents=True, exist_ok=True)

    # para cada mercado percorrer o que tem dentro
    for category_path in market_path.iterdir():

        # Caso o item não seja uma pasta ignore
        if not category_path.is_dir():
            continue

        # retorna o nome da pasta, divide o nome em String, pega apenas o elemento 0 da lista   
        category_code = category_path.name.split("-")[0]

        #mostra a categoria divida por espaço
        print(f"  Categoria {category_code}")

        #criação de uma lista para arrays de dados
        dados_categoria = []

        # Estrutura para perccorrer todos os arquivos com padrão .csv
        for csv_file in category_path.glob("*.csv"):

            #Busca o arquivo e salva o nome dele sem a extensão
            product_id = csv_file.stem.split("_")[0]

            # Lê o arquivo e retorna uma tabela de memoria,
            # Parametros de localização e conversão da coluna em data em datetime
            df = pd.read_csv(csv_file, parse_dates=["Date"])

            # coluna de produto
            df["product_id"] = product_id

            # coluna de categoria
            df["category"] = category_code

            # coluna de mercado
            df["market"] = market_name

            # armazena dados representada pela categoria
            dados_categoria.append(df)

        # Se existem dados na categoria
        if dados_categoria:
            # Junte todos esses dados em uma unica tabela com os dados corretamente no tempo
            df_categoria = pd.concat(dados_categoria, ignore_index=True)
            df_categoria = df_categoria.sort_values(
                ["product_id", "Date"]
            )
            # Define o arquivo de saida e salvo tudo no formato .csv
            output_file = market_output / f"cat{category_code}.csv"

            # Salvamento do arquivo csv
            df_categoria.to_csv(output_file, index=False)

            print(f"    Arquivo salvo: {output_file}")
        else:
            print(f"    ⚠️ Categoria {category_code} sem produtos")
