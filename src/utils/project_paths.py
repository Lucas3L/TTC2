from pathlib import Path
import sys


def get_project_root(current_file: str) -> Path:
    """Retorna a raiz do projeto dado o caminho de um arquivo dentro de src/."""
    return Path(current_file).resolve().parents[2]


def add_project_root_to_sys_path(current_file: str) -> Path:
    """Garante que a raiz do projeto esteja no sys.path e retorna essa raiz."""
    root = get_project_root(current_file)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.append(root_str)
    return root
