from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NotebookStats:
    path: Path
    cells: int
    code_cells: int
    markdown_cells: int
    executed_cells: int
    output_count: int
    empty_cells: int
    comment_only_code_cells: int

    @property
    def score(self) -> int:
        score = 100
        if self.markdown_cells == 0:
            score -= 35
        if self.executed_cells == 0:
            score -= 20
        if self.output_count == 0:
            score -= 10
        if self.code_cells > 0 and self.comment_only_code_cells / self.code_cells > 0.3:
            score -= 10
        if self.empty_cells > 0:
            score -= 5
        return max(score, 0)

    @property
    def recommendation(self) -> str:
        if self.score < 50:
            return "refatorar_urgente"
        if self.score < 75:
            return "refatorar"
        return "manter"


def analyze_notebook(path: Path) -> NotebookStats:
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])

    code_cells = 0
    markdown_cells = 0
    executed_cells = 0
    output_count = 0
    empty_cells = 0
    comment_only_code_cells = 0

    for c in cells:
        ctype = c.get("cell_type", "")
        source_lines = "".join(c.get("source", [])).splitlines()
        stripped = [ln.strip() for ln in source_lines if ln.strip()]

        if not stripped:
            empty_cells += 1

        if ctype == "code":
            code_cells += 1
            if c.get("execution_count") not in (None, 0):
                executed_cells += 1
            output_count += len(c.get("outputs", []))
            if stripped and all(ln.startswith("#") for ln in stripped):
                comment_only_code_cells += 1
        elif ctype == "markdown":
            markdown_cells += 1

    return NotebookStats(
        path=path,
        cells=len(cells),
        code_cells=code_cells,
        markdown_cells=markdown_cells,
        executed_cells=executed_cells,
        output_count=output_count,
        empty_cells=empty_cells,
        comment_only_code_cells=comment_only_code_cells,
    )


def build_markdown_report(stats: list[NotebookStats]) -> str:
    lines = [
        "# Avaliação de utilidade dos notebooks",
        "",
        "Esta avaliação mede utilidade prática para banca/reprodutibilidade.",
        "",
        "## Resumo",
        "",
        "| Notebook | Células | Markdown | Código | Executadas | Outputs | Score | Recomendação |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for s in stats:
        lines.append(
            f"| `{s.path.name}` | {s.cells} | {s.markdown_cells} | {s.code_cells} | {s.executed_cells} | {s.output_count} | {s.score} | {s.recommendation} |"
        )

    lines.extend(
        [
            "",
            "## Diagnóstico",
            "",
            "- Todos os notebooks com score baixo devem ser **refatorados** antes de uso em banca.",
            "- Células de texto atualmente como código (comentários) devem virar markdown.",
            "- Adicionar no fim de cada notebook: versão de dados, seed, e célula de exportação de artefatos.",
            "",
            "## Recomendação final",
            "",
            "- **Não excluir neste momento**: os notebooks representam as etapas do trabalho (EDA, processamento, clusterização, modelagem, resultados).",
            "- **Refatorar estrutura** para leitura acadêmica e rastreabilidade.",
            "- Se após refatoração algum notebook continuar redundante (ex.: clusterização não usada no experimento final), mover para `notebooks/archive/`.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita notebooks e gera relatório de utilidade")
    parser.add_argument("--notebooks-dir", type=Path, default=Path("notebooks"))
    parser.add_argument("--output", type=Path, default=Path("docs/notebooks_avaliacao.md"))
    args = parser.parse_args()

    nb_files = sorted(args.notebooks_dir.glob("*.ipynb"))
    stats = [analyze_notebook(p) for p in nb_files]
    report = build_markdown_report(stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Relatório salvo em: {args.output}")


if __name__ == "__main__":
    main()
