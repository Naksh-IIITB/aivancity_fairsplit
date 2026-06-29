"""Court-admissible PDF report generator.

Designed to match what the deck describes:
- valuation summary
- methodology block (model, CV RMSLE, dataset)
- SHAP factor table (which features drove the number, by how much)
- input data appendix
- audit trail (model version, timestamp, dataset hash)
- co-signature line for a certified appraiser
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import __version__
from .explain import FactorContribution
from .models import FittedEnsemble


@dataclass
class ReportContext:
    case_id: str
    valuation_date: str       # ISO-8601, e.g. 2026-06-04 (date-of-separation date)
    jurisdiction: str          # e.g. "France", "India"
    property_address: str
    appraiser_name: str = "[ to be co-signed ]"
    appraiser_credential: str = "[ expert judiciaire / IBBI valuer ]"


def _money(amount: float, jurisdiction: str) -> str:
    if jurisdiction.lower().startswith("ind"):
        return f"₹ {amount:,.0f}"
    return f"€ {amount:,.0f}"


def _dataset_hash(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def render(
    out_path: str | Path,
    ctx: ReportContext,
    ensemble: FittedEnsemble,
    property_row: dict,
    predicted_price: float,
    factors: list[FactorContribution],
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"FairSplit Valuation — {ctx.case_id}",
        author="FairSplit",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Kicker", parent=styles["Normal"],
                              textColor=colors.HexColor("#009C9C"),
                              fontSize=9, spaceAfter=2))
    styles.add(ParagraphStyle(name="H1Navy", parent=styles["Heading1"],
                              textColor=colors.HexColor("#0B1F3A"),
                              spaceAfter=6))
    styles.add(ParagraphStyle(name="H2Navy", parent=styles["Heading2"],
                              textColor=colors.HexColor("#0B1F3A"),
                              spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"],
                              fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="Mono", parent=styles["Normal"],
                              fontName="Courier", fontSize=8.5,
                              textColor=colors.HexColor("#555B66")))

    story = []
    story.append(Paragraph("NEUTRAL PROPERTY VALUATION", styles["Kicker"]))
    story.append(Paragraph(
        f"Case {ctx.case_id} — {ctx.jurisdiction}", styles["H1Navy"]))
    story.append(Paragraph(ctx.property_address, styles["Body"]))
    story.append(Spacer(1, 0.4 * cm))

    # ---- valuation banner ------------------------------------------------
    headline = Table(
        [[
            Paragraph("AI VALUATION", styles["Kicker"]),
            Paragraph("DATE OF VALUATION", styles["Kicker"]),
        ], [
            Paragraph(
                f"<b>{_money(predicted_price, ctx.jurisdiction)}</b>",
                ParagraphStyle("big", parent=styles["Body"],
                               fontSize=24, leading=28,
                               textColor=colors.HexColor("#0B1F3A"))),
            Paragraph(ctx.valuation_date, styles["Body"]),
        ]],
        colWidths=[10 * cm, 7 * cm],
    )
    headline.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F6F8")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E6EC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(headline)
    story.append(Spacer(1, 0.4 * cm))

    # ---- methodology -----------------------------------------------------
    story.append(Paragraph("Methodology", styles["H2Navy"]))
    story.append(Paragraph(
        "Valuation produced by the FairSplit ensemble model — a blend of "
        "an ElasticNet linear regression (L1+L2 regularised) and a CatBoost "
        "gradient-boosted decision tree. Both models are trained in log-price "
        "space and combined with a weight chosen by cross-validated grid search "
        "on out-of-fold predictions.", styles["Body"]))
    story.append(Spacer(1, 0.2 * cm))

    method = Table(
        [
            ["Model", "ElasticNet + CatBoost ensemble"],
            ["Blend weight (linear : tree)", f"{ensemble.blend:.2f}  :  {1 - ensemble.blend:.2f}"],
            [f"CV RMSLE — ElasticNet ({ensemble.cv_folds}-fold)",
             f"{ensemble.cv_rmsle_linear:.4f}"],
            [f"CV RMSLE — CatBoost ({ensemble.cv_folds}-fold)",
             f"{ensemble.cv_rmsle_catboost:.4f}"],
            [f"CV RMSLE — Ensemble ({ensemble.cv_folds}-fold)",
             f"{ensemble.cv_rmsle_blend:.4f}"],
            ["Target transform", "log1p"],
        ],
        colWidths=[7 * cm, 10 * cm],
    )
    method.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F4F6F8")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E6EC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E6EC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(method)
    story.append(Spacer(1, 0.4 * cm))

    # ---- SHAP factors ----------------------------------------------------
    story.append(Paragraph("Top contributing factors (SHAP)", styles["H2Navy"]))
    story.append(Paragraph(
        "Each row shows how a single property attribute moved the valuation "
        "above or below the model baseline. Values are local SHAP attributions "
        "from the CatBoost branch, converted from log-price space to currency "
        "for legibility.", styles["Body"]))
    story.append(Spacer(1, 0.2 * cm))

    factor_data = [["Factor", "Value", "Direction", f"Impact ({ctx.jurisdiction})"]]
    for f in factors:
        factor_data.append([
            f.feature,
            str(f.value),
            f.direction,
            _money(f.impact_eur, ctx.jurisdiction),
        ])
    factor_table = Table(factor_data, colWidths=[5.5 * cm, 4.5 * cm, 2 * cm, 5 * cm])
    factor_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1F3A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (2, 1), (3, -1), "RIGHT"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E6EC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E6EC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F4F6F8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(factor_table)
    story.append(PageBreak())

    # ---- appendix: inputs ------------------------------------------------
    story.append(Paragraph("Appendix A — Property inputs", styles["H2Navy"]))
    rows = [["Field", "Value"]]
    for k, v in property_row.items():
        rows.append([k, str(v)])
    appendix = Table(rows, colWidths=[7 * cm, 10 * cm])
    appendix.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1F3A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E6EC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E6EC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F4F6F8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(appendix)
    story.append(Spacer(1, 0.6 * cm))

    # ---- audit trail -----------------------------------------------------
    story.append(Paragraph("Appendix B — Audit trail", styles["H2Navy"]))
    audit = [
        ["Report generated at (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")],
        ["FairSplit version", __version__],
        ["Input hash (SHA-256, 16-char)", _dataset_hash(property_row)],
        ["Model blend weight", f"{ensemble.blend:.4f}"],
        ["Number of features (linear)", str(len(ensemble.feature_names))],
    ]
    audit_t = Table(audit, colWidths=[7 * cm, 10 * cm])
    audit_t.setStyle(TableStyle([
        ("FONTNAME", (1, 0), (1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E6EC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E6EC")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F4F6F8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(audit_t)
    story.append(Spacer(1, 0.8 * cm))

    # ---- co-signature ----------------------------------------------------
    story.append(Paragraph("Co-signature — certified appraiser", styles["H2Navy"]))
    story.append(Paragraph(
        f"<b>Name:</b> {ctx.appraiser_name}<br/>"
        f"<b>Credential:</b> {ctx.appraiser_credential}<br/>"
        f"<b>Signature:</b> ____________________________   <b>Date:</b> __________",
        styles["Body"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<i>Disclaimer — This report is an AI-generated valuation prepared by "
        "FairSplit. It must be co-signed by a registered expert "
        "(expert judiciaire in France, IBBI-registered valuer in India) before "
        "being submitted as evidence. The valuation reflects the best estimate "
        "of fair market value as of the date stated above; it does not constitute "
        "legal advice.</i>", styles["Body"]))

    doc.build(story)
    return out_path
