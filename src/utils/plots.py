
from pathlib import Path
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    plt.style.use("seaborn-v0_8-whitegrid")
    return plt



DEFAULT_RESULTS_FILE = Path("Resultados/consolidated_results.csv")
DEFAULT_OUTPUT_DIR = Path("Resultados/plots")


# Definição dos caminhos para leitura dos dados consolidados e salvamento das imagens
RESULTS_FILE = Path("Resultados/consolidated_results.csv")
OUTPUT_DIR = Path("Resultados/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Cria a pasta de gráficos caso não exista
COLUMN_MAP = {
    "mae": ["mae", "mae_test", "MAE"],
    "rmse": ["rmse", "rmse_test", "RMSE"],
    "smape": ["smape", "smape_test", "SMAPE", "sMAPE"],
    "model": ["model", "modelo"],
    "category": ["category", "cat", "categoria"],
    "product": ["product", "produto", "sku"],
}

def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def find_column(df, metric):
    for col in COLUMN_MAP[metric]:
        if col in df.columns:
            return col
    raise ValueError(f" Nenhuma coluna válida encontrada para {metric}")

def _find_column(headers: list[str], key: str) -> str:
    for col in COLUMN_MAP[key]:
        if col in headers:
            return col
    raise ValueError(f"Nenhuma coluna válida encontrada para '{key}'. Colunas: {headers}")


def _to_float(value: str) -> float:
    if value is None:
        return float("nan")
    value = str(value).strip().replace(",", ".")
    if value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_results(results_file: Path) -> tuple[list[dict], dict[str, str]]:
    if not results_file.exists():
        raise FileNotFoundError(f"Arquivo de resultados não encontrado: {results_file}")

    with results_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        col = {
            "model": _find_column(headers, "model"),
            "mae": _find_column(headers, "mae"),
            "rmse": _find_column(headers, "rmse"),
            "smape": _find_column(headers, "smape"),
        }
        for optional in ["category", "product"]:
            try:
                col[optional] = _find_column(headers, optional)
            except ValueError:
                pass

        rows = list(reader)

    for row in rows:
        row[col["mae"]] = _to_float(row.get(col["mae"], ""))
        row[col["rmse"]] = _to_float(row.get(col["rmse"], ""))
        row[col["smape"]] = _to_float(row.get(col["smape"], ""))

    return rows, col


def _save_current_figure(output_dir: Path, name: str, plt) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png", dpi=300)
    plt.savefig(output_dir / f"{name}.pdf")
    plt.close()


def _group_metric_mean(rows: list[dict], model_col: str, metric_col: str) -> list[tuple[str, float]]:
    agg = defaultdict(list)
    for r in rows:
        val = r.get(metric_col)
        if isinstance(val, float) and val == val:
            agg[str(r.get(model_col, "desconhecido"))].append(val)
    return sorted([(k, _mean(v)) for k, v in agg.items() if v], key=lambda x: x[1])


def plot_metric_by_model(rows: list[dict], model_col: str, metric_col: str, metric_name: str, output_dir: Path, plt) -> None:
    summary = _group_metric_mean(rows, model_col, metric_col)
    if not summary:
        return

    labels = [x[0] for x in summary]
    values = [x[1] for x in summary]

    _, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values)
    ax.set_title(f"Comparação de modelos por {metric_name.upper()} (média)")
    ax.set_ylabel(metric_name.upper())
    ax.set_xlabel("Modelo")

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    _save_current_figure(output_dir, f"models_{metric_name}", plt)


def plot_smape_distribution(rows: list[dict], smape_col: str, output_dir: Path, plt) -> None:
    data = [r[smape_col] for r in rows if isinstance(r.get(smape_col), float) and r[smape_col] == r[smape_col]]
    if not data:
        return
    _, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data, bins=30, alpha=0.85)
    ax.set_title("Distribuição do sMAPE")
    ax.set_xlabel("sMAPE")
    ax.set_ylabel("Frequência")
    ax.axvline(200, color="red", linestyle="--", linewidth=2, label="Limite teórico 200")
    ax.legend()
    _save_current_figure(output_dir, "smape_distribution", plt)


def plot_mae_vs_smape(rows: list[dict], mae_col: str, smape_col: str, output_dir: Path, plt) -> None:
    mae = [r[mae_col] for r in rows if isinstance(r.get(mae_col), float) and r[mae_col] == r[mae_col]]
    smape = [r[smape_col] for r in rows if isinstance(r.get(mae_col), float) and r.get(mae_col) == r.get(mae_col) and isinstance(r.get(smape_col), float) and r[smape_col] == r[smape_col]]
    if not mae or not smape:
        return

    non_mae = [m for m, s in zip(mae, smape) if s < 199.999]
    non_smape = [s for s in smape if s < 199.999]
    sat_mae = [m for m, s in zip(mae, smape) if s >= 199.999]
    sat_smape = [s for s in smape if s >= 199.999]

    _, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(non_mae, non_smape, alpha=0.7, label="sMAPE < 200")
    if sat_mae:
        ax.scatter(sat_mae, sat_smape, color="red", alpha=0.8, label="sMAPE = 200")
    ax.set_title("Relação entre MAE e sMAPE")
    ax.set_xlabel("MAE")
    ax.set_ylabel("sMAPE")
    ax.legend()
    _save_current_figure(output_dir, "mae_vs_smape", plt)


def plot_smape_by_category(rows: list[dict], category_col: str, smape_col: str, output_dir: Path, plt) -> None:
    grouped = defaultdict(list)
    for r in rows:
        c = str(r.get(category_col, "desconhecido"))
        s = r.get(smape_col)
        if isinstance(s, float) and s == s:
            grouped[c].append(s)

    if not grouped:
        return

    labels = list(grouped.keys())
    series = [grouped[k] for k in labels]

    _, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(series, labels=labels)
    ax.set_title("sMAPE por categoria")
    ax.set_xlabel("Categoria")
    ax.set_ylabel("sMAPE")
    _save_current_figure(output_dir, "smape_by_category", plt)


def _worst_rows(rows: list[dict], mae_col: str, k: int = 5) -> list[dict]:
    valid = [r for r in rows if isinstance(r.get(mae_col), float) and r[mae_col] == r[mae_col]]
    return sorted(valid, key=lambda r: r[mae_col], reverse=True)[:k]


def print_technical_summary(rows: list[dict], col: dict[str, str], output_dir: Path) -> None:
    smape_vals = [r[col["smape"]] for r in rows if isinstance(r.get(col["smape"]), float) and r[col["smape"]] == r[col["smape"]]]
    mae_vals = [r[col["mae"]] for r in rows if isinstance(r.get(col["mae"]), float) and r[col["mae"]] == r[col["mae"]]]
    rmse_vals = [r[col["rmse"]] for r in rows if isinstance(r.get(col["rmse"]), float) and r[col["rmse"]] == r[col["rmse"]]]

    paired = [r for r in rows if all(isinstance(r.get(col[k]), float) and r[col[k]] == r[col[k]] for k in ["mae", "rmse", "smape"])]

    smape_200_ratio = (sum(1 for r in paired if r[col["smape"]] >= 199.999) / len(paired) * 100) if paired else 0.0
    rmse_mae_violations = sum(1 for r in paired if r[col["rmse"]] < r[col["mae"]])

    lines = [
        "===== RESUMO TÉCNICO =====",
        f"Total de linhas avaliadas: {len(rows)}",
        f"sMAPE médio: {_mean(smape_vals):.2f}" if smape_vals else "sMAPE médio: n/a",
        f"MAE médio: {_mean(mae_vals):.2f}" if mae_vals else "MAE médio: n/a",
        f"RMSE médio: {_mean(rmse_vals):.2f}" if rmse_vals else "RMSE médio: n/a",
        f"% de linhas com sMAPE=200: {smape_200_ratio:.2f}%",
        f"Violações da regra RMSE >= MAE: {rmse_mae_violations}",
        "",
        "Top 5 piores MAE:",
    ]

    display_cols = [c for c in [col.get("category"), col.get("product"), col["model"], col["mae"], col["rmse"], col["smape"]] if c]
    for r in _worst_rows(rows, col["mae"], k=5):
        lines.append(" | ".join(f"{c}={r.get(c)}" for c in display_cols))

    lines.extend(["", "Recomendações automáticas:"])
    if smape_200_ratio > 30:
        lines.append("- Alta incidência de sMAPE=200: separar séries intermitentes e avaliar Croston/SBA/Tsb.")
        lines.append("- Incluir métrica adicional robusta a zeros (MASE, WAPE) para banca e comparação.")
    if rmse_mae_violations > 0:
        lines.append("- Existem inconsistências matemáticas entre MAE e RMSE. Verifique o pipeline de métricas.")
    else:
        lines.append("- MAE e RMSE consistentes (RMSE >= MAE em todas as linhas).")

    text = "\n".join(lines)
    print(text)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "technical_summary.txt").write_text(text, encoding="utf-8")


def generate_all_plots(results_file: Path, output_dir: Path) -> None:
    rows, col = load_results(results_file)

    plt = _get_plt()
    if plt is not None:
        plot_metric_by_model(rows, col["model"], col["mae"], "mae", output_dir, plt)
        plot_metric_by_model(rows, col["model"], col["rmse"], "rmse", output_dir, plt)
        plot_metric_by_model(rows, col["model"], col["smape"], "smape", output_dir, plt)
        plot_smape_distribution(rows, col["smape"], output_dir, plt)
        plot_mae_vs_smape(rows, col["mae"], col["smape"], output_dir, plt)
        if "category" in col:
            plot_smape_by_category(rows, col["category"], col["smape"], output_dir, plt)
    else:
        print("Aviso: matplotlib não disponível. Gráficos serão pulados, mas o resumo técnico será gerado.")

    print_technical_summary(rows, col, output_dir)
    print(f"\nSaídas salvas em: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera gráficos e diagnóstico técnico a partir de CSV de resultados.")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE, help="CSV com resultados consolidados")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Diretório de saída dos gráficos")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all_plots(args.results_file, args.output_dir)