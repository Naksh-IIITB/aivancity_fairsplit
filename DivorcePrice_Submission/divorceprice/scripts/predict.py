"""Score a single property and emit the court-admissible PDF.

Usage:
    python -m scripts.predict \
        --artifacts artifacts/ \
        --row data/raw/sample_property.json \
        --case-id "Doe v. Doe — 2026-FR-1142" \
        --jurisdiction France \
        --address "12 rue de la Paix, 75002 Paris" \
        --out reports/doe_v_doe.pdf

The --row file is a JSON object whose keys match the training schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from divorceprice import pipeline                              # noqa: E402
from divorceprice.explain import explain_one                   # noqa: E402
from divorceprice.report import ReportContext, render          # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="FairSplit predict + report")
    p.add_argument("--artifacts", type=str, default="artifacts/",
                   help="Directory containing ensemble.joblib (from train.py)")
    p.add_argument("--row", type=str, required=True,
                   help="JSON file with one property's fields")
    p.add_argument("--out", type=str, required=True, help="Output PDF path")
    p.add_argument("--case-id", type=str, required=True)
    p.add_argument("--jurisdiction", type=str, default="France")
    p.add_argument("--address", type=str, default="—")
    p.add_argument("--valuation-date", type=str, default=date.today().isoformat(),
                   help="Date of separation / valuation (YYYY-MM-DD)")
    p.add_argument("--appraiser-name", type=str, default="[ to be co-signed ]")
    p.add_argument("--appraiser-credential", type=str,
                   default="[ expert judiciaire / IBBI valuer ]")
    p.add_argument("--top-k", type=int, default=10,
                   help="How many SHAP factors to include in the report")
    args = p.parse_args()

    print(f"[predict] loading ensemble from {args.artifacts}…")
    ensemble = pipeline.load(args.artifacts)

    print(f"[predict] reading property row from {args.row}…")
    raw = json.loads(Path(args.row).read_text())
    row = pd.DataFrame([raw])

    price, factors = explain_one(ensemble, row, top_k=args.top_k)
    print(f"[predict] valuation: {price:,.0f}")
    print("[predict] top factors:")
    for f in factors:
        print(f"           {f.direction} {f.feature:<20} = {f.value!s:<20} "
              f"impact ≈ {f.impact_eur:+,.0f}")

    ctx = ReportContext(
        case_id=args.case_id,
        valuation_date=args.valuation_date,
        jurisdiction=args.jurisdiction,
        property_address=args.address,
        appraiser_name=args.appraiser_name,
        appraiser_credential=args.appraiser_credential,
    )
    out_pdf = render(args.out, ctx, ensemble, raw, price, factors)
    print(f"[predict] report written to {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
