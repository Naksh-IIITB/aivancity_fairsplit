from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "ipynb_submission"


def main() -> None:
    for path in sorted(SUBMISSION_DIR.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            text = "".join(cell.get("source", []))
            dedented = textwrap.dedent(text).lstrip("\n")
            if dedented != text:
                cell["source"] = dedented.splitlines(keepends=True)
                changed = True
        if changed:
            path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
