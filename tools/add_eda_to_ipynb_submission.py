from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "ipynb_submission"


def source(text: str) -> list[str]:
    return text.strip().splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


EDA_CELLS = [
    markdown(
        """
        ## Exploratory Data Analysis

        This EDA section is included before model training so each model notebook is self-contained. It checks the target distribution, missing values, strongest numerical correlations, outliers, and the full modeling workflow.
        """
    ),
    code(
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        sns.histplot(train["SalePrice"], kde=True, ax=axes[0], color="#4169e1")
        axes[0].set_title(f"Raw SalePrice | skew = {train['SalePrice'].skew():.2f}")
        axes[0].set_xlabel("SalePrice")

        sns.histplot(np.log(train["SalePrice"]), kde=True, ax=axes[1], color="#2e8b57")
        axes[1].set_title(f"log(SalePrice) | skew = {np.log(train['SalePrice']).skew():.2f}")
        axes[1].set_xlabel("log(SalePrice)")

        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        missing = train.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False).head(20)
        missing_table = pd.DataFrame({
            "missing_count": missing,
            "missing_percent": (missing / len(train) * 100).round(2),
        })

        display(missing_table)

        if not missing_table.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=missing_table.reset_index().rename(columns={"index": "feature"}),
                y="feature",
                x="missing_percent",
                color="#d95f02",
            )
            plt.title("Top Missing-Value Features")
            plt.xlabel("Missing values (%)")
            plt.ylabel("")
            plt.tight_layout()
            plt.show()
        """
    ),
    code(
        """
        numeric_for_corr = train.select_dtypes(include=np.number).columns
        corr_target = train[numeric_for_corr].corr(numeric_only=True)["SalePrice"].abs().sort_values(ascending=False)
        display(corr_target.head(15).to_frame("abs_corr_with_saleprice"))

        top_features = corr_target.head(11).index
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            train[top_features].corr(numeric_only=True),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
        )
        plt.title("Correlation Heatmap: Top SalePrice Drivers")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=train, x="GrLivArea", y="SalePrice", hue="OverallQual", palette="viridis", legend=False)
        plt.title("Outlier Check: Living Area vs Sale Price")
        plt.xlabel("Above-ground living area")
        plt.ylabel("SalePrice")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis("off")

        steps = [
            "Load data",
            "Clean missing values",
            "Feature engineering",
            "Preprocess",
            "Train model",
            "CV RMSLE",
            "Submission CSV",
        ]
        x_positions = np.linspace(0.06, 0.94, len(steps))

        for idx, (x_pos, label) in enumerate(zip(x_positions, steps)):
            ax.text(
                x_pos,
                0.55,
                label,
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f7fb", edgecolor="#4c566a"),
                transform=ax.transAxes,
            )
            if idx < len(steps) - 1:
                ax.annotate(
                    "",
                    xy=(x_positions[idx + 1] - 0.055, 0.55),
                    xytext=(x_pos + 0.055, 0.55),
                    arrowprops=dict(arrowstyle="->", color="#4c566a", lw=1.5),
                    xycoords=ax.transAxes,
                    textcoords=ax.transAxes,
                )

        ax.set_title("End-to-End Modeling Workflow", fontsize=13, pad=16)
        plt.show()
        """
    ),
]


BUSINESS_VISUAL_CELLS = [
    markdown(
        """
        ## Business Diagrams

        The diagrams below summarize how the startup creates value and how the product sits in the market.
        """
    ),
    markdown(
        """
        ```mermaid
        flowchart LR
            A[Property Listing Data] --> B[FairPrice AI Model]
            B --> C[Predicted Fair Value]
            B --> D[Value Driver Explanation]
            C --> E[Underpriced / Fair / Overpriced Verdict]
            D --> F[Buyer Negotiation Report]
            E --> F
        ```
        """
    ),
    markdown(
        """
        ```mermaid
        quadrantChart
            title Real Estate Valuation Positioning
            x-axis Low Explainability --> High Explainability
            y-axis Consumer Tool --> Professional Platform
            quadrant-1 Explainable Pro Tools
            quadrant-2 Black-Box Pro Tools
            quadrant-3 Basic Consumer Tools
            quadrant-4 Explainable Consumer Tools
            Zillow Zestimate: [0.45, 0.45]
            HouseCanary: [0.55, 0.82]
            Realtor.com tools: [0.35, 0.30]
            FairPrice AI: [0.82, 0.38]
        ```
        """
    ),
]


def add_eda_to_model_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    if any("## Exploratory Data Analysis" in "".join(cell.get("source", [])) for cell in cells):
        return

    insert_at = None
    for index, existing in enumerate(cells):
        text = "".join(existing.get("source", []))
        if "## Train, Evaluate, and Generate Submission" in text:
            insert_at = index
            break

    if insert_at is None:
        insert_at = len(cells) - 1

    notebook["cells"] = cells[:insert_at] + EDA_CELLS + cells[insert_at:]
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def add_business_diagrams(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    if any("## Business Diagrams" in "".join(cell.get("source", [])) for cell in cells):
        return

    insert_at = 4
    notebook["cells"] = cells[:insert_at] + BUSINESS_VISUAL_CELLS + cells[insert_at:]
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def main() -> None:
    for path in sorted(SUBMISSION_DIR.glob("*.ipynb")):
        if path.name.startswith("00_Business"):
            add_business_diagrams(path)
        else:
            add_eda_to_model_notebook(path)
        print(path)


if __name__ == "__main__":
    main()
