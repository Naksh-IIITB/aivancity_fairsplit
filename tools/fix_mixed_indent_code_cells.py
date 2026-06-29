from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "ipynb_submission"


def fix_mixed_indent(text: str) -> str:
    lines = text.splitlines(keepends=True)
    non_empty_indexes = [i for i, line in enumerate(lines) if line.strip()]
    if len(non_empty_indexes) < 2:
        return text

    first = lines[non_empty_indexes[0]]
    second = lines[non_empty_indexes[1]]
    if first.startswith(" ") or not second.startswith("        "):
        return text

    fixed = []
    for line in lines:
        if line.startswith("        "):
            fixed.append(line[8:])
        else:
            fixed.append(line)
    return "".join(fixed)


def main() -> None:
    for path in sorted(SUBMISSION_DIR.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            text = "".join(cell.get("source", []))
            fixed = fix_mixed_indent(text)
            if fixed != text:
                cell["source"] = fixed.splitlines(keepends=True)
                changed = True
        if changed:
            path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
