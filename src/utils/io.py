from pathlib import Path
import pandas as pd


def read_csv(path, parse_dates=None):
    # Converte string para objeto Path para manipulação robusta de caminhos
    path = Path(path)

    # Cláusula de guarda para validar a existência do arquivo antes da leitura
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    # Retorna o DataFrame com suporte opcional à conversão de colunas temporais
    return pd.read_csv(path, parse_dates=parse_dates)


def save_csv(df, path, index=False):
    # Garante que o caminho de destino seja um objeto Path
    path = Path(path)
    
    # Cria recursivamente toda a estrutura de pastas pai caso não existam
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Persiste o DataFrame em disco sem o índice padrão do Pandas
    df.to_csv(path, index=index)


def list_csv_files(folder):
    # Converte o diretório alvo para Path
    folder = Path(folder)
    
    # Retorna uma lista contendo os caminhos de todos os arquivos .csv encontrados
    return list(folder.glob("*.csv"))