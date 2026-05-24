from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def df_to_table(df: pd.DataFrame, col_widths=None) -> Table:
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def add_image_if_exists(story, image_path: Path, caption: str, styles) -> None:
    if image_path.exists():
        story.append(Paragraph(caption, styles["CaptionCustom"]))
        img = Image(str(image_path))
        img.drawWidth = 6.4 * inch
        img.drawHeight = 4.0 * inch
        story.append(img)
        story.append(Spacer(1, 0.12 * inch))


def main() -> None:
    md_path = Path("telecom_churn_report.md")
    out_dir = Path("outputs")
    pdf_path = Path("telecom_churn_report.pdf")
    text = md_path.read_text(encoding="utf-8")

    model_df = pd.read_csv(out_dir / "model_comparison_results.csv")
    thr_best_df = pd.read_csv(out_dir / "threshold_analysis_best_model.csv")
    thr_int_df = pd.read_csv(out_dir / "threshold_analysis_interpretable_model.csv")
    cal_df = pd.read_csv(out_dir / "calibration_results.csv")

    model_df = model_df[
        [
            "Model",
            "CV_ROC_AUC",
            "Test_ROC_AUC",
            "Test_PR_AUC",
            "Test_Precision",
            "Test_Recall",
            "Test_F1",
            "Brier_Score",
        ]
    ].copy()
    for col in model_df.columns[1:]:
        model_df[col] = model_df[col].map(lambda v: f"{v:.4f}")

    for dfx in (thr_best_df, thr_int_df):
        for col in ["Threshold", "Precision", "Recall", "F1"]:
            dfx[col] = dfx[col].map(lambda v: f"{float(v):.3f}")

    cal_df["Brier_Score"] = cal_df["Brier_Score"].map(lambda v: f"{float(v):.4f}")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontSize=16, leading=20, spaceAfter=14))
    styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], fontSize=12.5, leading=15, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="H3Custom", parent=styles["Heading3"], fontSize=11.5, leading=14, spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontSize=10.5, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="CaptionCustom", parent=styles["Italic"], fontSize=9.5, leading=12, spaceBefore=6, spaceAfter=4))

    story = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            story.append(Spacer(1, 0.08 * inch))
            continue
        if line.startswith("# "):
            story.append(Paragraph(line[2:].strip(), styles["TitleCustom"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:].strip(), styles["H2Custom"]))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:].strip(), styles["H3Custom"]))
        elif line.startswith("- "):
            body = line[2:].replace("**", "").replace("`", "")
            body = body.replace("approx.", "&asymp;")
            story.append(Paragraph(f"• {body}", styles["BodyCustom"]))
        else:
            body = line.replace("**", "").replace("`", "")
            body = body.replace("approx.", "&asymp;")
            story.append(Paragraph(body, styles["BodyCustom"]))

    story.append(PageBreak())
    story.append(Paragraph("Appendix A: Model Comparison Results", styles["H2Custom"]))
    story.append(df_to_table(model_df))
    story.append(Spacer(1, 0.16 * inch))

    story.append(Paragraph("Appendix B: Threshold Analysis (Best Predictive Model)", styles["H2Custom"]))
    story.append(df_to_table(thr_best_df))
    story.append(Spacer(1, 0.16 * inch))

    story.append(Paragraph("Appendix C: Threshold Analysis (Most Interpretable Model)", styles["H2Custom"]))
    story.append(df_to_table(thr_int_df))
    story.append(Spacer(1, 0.16 * inch))

    story.append(Paragraph("Appendix D: Calibration Results", styles["H2Custom"]))
    story.append(df_to_table(cal_df))
    story.append(PageBreak())

    story.append(Paragraph("Figures", styles["H2Custom"]))
    add_image_if_exists(
        story,
        out_dir / "roc_curves.png",
        "Figure 1. ROC curves across logistic-family models.",
        styles,
    )
    add_image_if_exists(
        story,
        out_dir / "pr_curves.png",
        "Figure 2. Precision-Recall curves showing minority-class retrieval performance.",
        styles,
    )
    add_image_if_exists(
        story,
        out_dir / "calibration_curves.png",
        "Figure 3. Calibration curves (raw vs calibrated probabilities).",
        styles,
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        rightMargin=0.8 * inch,
        leftMargin=0.8 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
    )
    doc.build(story)
    print(f"Created: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
