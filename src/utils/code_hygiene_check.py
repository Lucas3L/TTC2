from __future__ import annotations

import argparse
import json
from pathlib import Path


def _check_python_file(path: Path) -> list[str]:
    issues = []
    lines = path.read_text(encoding="utf-8").splitlines()

    # duplicate import statements (same text repeated)
    imports = [ln.strip() for ln in lines if ln.strip().startswith("import ") or ln.strip().startswith("from ")]
    for imp in sorted(set(imports)):
        count = imports.count(imp)
        if count > 1:
            issues.append(f"{path}: import duplicado `{imp}` ({count}x)")

    # consecutive duplicated lines
    for i in range(1, len(lines)):
        if lines[i].strip() and lines[i] == lines[i - 1]:
            issues.append(f"{path}: linha repetida consecutiva na linha {i+1}: `{lines[i].strip()}`")

    return issues


def _check_notebook(path: Path) -> list[str]:
    issues = []
    nb = json.loads(path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb.get("cells", []), start=1):
        src = cell.get("source", [])
        # repeated import lines in the same cell
        imports = [ln.strip() for ln in src if ln.strip().startswith("import ") or ln.strip().startswith("from ")]
        for imp in sorted(set(imports)):
            c = imports.count(imp)
            if c > 1:
                issues.append(f"{path}: célula {idx} com import duplicado `{imp}` ({c}x)")

        for i in range(1, len(src)):
            if src[i].strip() and src[i] == src[i - 1]:
                issues.append(f"{path}: célula {idx} com linha repetida consecutiva: `{src[i].strip()}`")

    return issues


def run_check(root: Path) -> list[str]:
    issues = []
    for p in sorted((root / "src").rglob("*.py")):
        issues.extend(_check_python_file(p))
    for p in sorted((root / "notebooks").glob("*.ipynb")):
        issues.extend(_check_notebook(p))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida higiene de código contra linhas/imports redundantes")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    issues = run_check(args.root)
    if issues:
        print("Foram encontrados problemas de redundância:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Nenhuma redundância detectada (imports/linhas duplicadas).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
