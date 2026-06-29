from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "ipynb_submission"


WARNING_SNIPPET = (
    "import warnings\n"
    "warnings.filterwarnings(\"ignore\", category=RuntimeWarning)\n"
    "warnings.filterwarnings(\"ignore\", category=UserWarning)\n"
)


def main() -> None:
    for path in sorted(SUBMISSION_DIR.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            text = "".join(cell.get("source", []))
            if "from pathlib import Path" in text and "warnings.filterwarnings" not in text:
                text = text.replace("from pathlib import Path\n", "from pathlib import Path\nimport warnings\n", 1)
                text = text.replace(
                    "RANDOM_STATE = 42\n",
                    "warnings.filterwarnings(\"ignore\", category=RuntimeWarning)\n"
                    "warnings.filterwarnings(\"ignore\", category=UserWarning)\n\n"
                    "RANDOM_STATE = 42\n",
                    1,
                )
                changed = True
            if "plt.show()" in text and "plt.close()" not in text:
                text = text.replace("plt.show()", "plt.show()\nplt.close()")
                changed = True
            cell["source"] = text.splitlines(keepends=True)
        if changed:
            path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
