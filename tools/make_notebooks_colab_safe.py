from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "ipynb_submission"


COLAB_DATA_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Colab data setup: upload train.csv and test.csv if the data folder is missing.\n",
        "DATA_DIR = Path(\"data\")\n",
        "required_files = [\"train.csv\", \"test.csv\"]\n",
        "\n",
        "if not all((DATA_DIR / name).exists() for name in required_files):\n",
        "    DATA_DIR.mkdir(exist_ok=True)\n",
        "    try:\n",
        "        from google.colab import files\n",
        "        print(\"Upload train.csv and test.csv. You may also upload sample_submission.csv and data_description.txt.\")\n",
        "        uploaded = files.upload()\n",
        "        for filename, content in uploaded.items():\n",
        "            (DATA_DIR / filename).write_bytes(content)\n",
        "    except ModuleNotFoundError:\n",
        "        print(\"Not running in Colab. Put train.csv and test.csv inside a local data/ folder.\")\n",
        "\n",
        "missing = [name for name in required_files if not (DATA_DIR / name).exists()]\n",
        "if missing:\n",
        "    raise FileNotFoundError(f\"Missing required data files in {DATA_DIR}: {missing}\")\n",
        "\n",
        "print(\"Data folder ready:\", sorted(path.name for path in DATA_DIR.glob(\"*\")))\n",
    ],
}


CATBOOST_INSTALL_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Colab dependency setup for CatBoost.\n",
        "import importlib.util\n",
        "import subprocess\n",
        "import sys\n",
        "\n",
        "if importlib.util.find_spec(\"catboost\") is None:\n",
        "    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"catboost\"])\n",
        "\n",
        "print(\"CatBoost is ready.\")\n",
    ],
}


BUSINESS_NOTE = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Google Colab Note\n",
        "\n",
        "This business notebook is markdown-only, so it does not require dataset upload. If Colab does not render Mermaid diagrams, the surrounding tables still contain the same information.\n",
    ],
}


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def insert_after_first_setup(cells: list[dict], new_cell: dict) -> list[dict]:
    if any("Colab data setup" in cell_text(cell) for cell in cells):
        return cells
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code" and "DATA_DIR = Path" in cell_text(cell):
            return cells[: index + 1] + [new_cell] + cells[index + 1 :]
    return [new_cell] + cells


def main() -> None:
    for path in sorted(SUBMISSION_DIR.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook["cells"]

        if path.name.startswith("00_Business"):
            if not any("Google Colab Note" in cell_text(cell) for cell in cells):
                cells = cells[:1] + [BUSINESS_NOTE] + cells[1:]
        else:
            cells = insert_after_first_setup(cells, COLAB_DATA_CELL)
            if "CatBoost" in path.name and not any("Colab dependency setup for CatBoost" in cell_text(cell) for cell in cells):
                cells = cells[:1] + [CATBOOST_INSTALL_CELL] + cells[1:]

        notebook["cells"] = cells
        path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
