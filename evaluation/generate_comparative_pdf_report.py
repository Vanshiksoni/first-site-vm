import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter, defaultdict

from pypdf import PdfWriter

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = Path(__file__).parent / "pdf_reports"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_CONFIGS = [
    {"key": "codellama", "name": "CodeLlama 7B", "color": "#2563EB"},
    {"key": "llama3.2_3b", "name": "Llama 3.2 (3B)", "color": "#7C3AED"},
    {"key": "mistral_7b", "name": "Mistral 7B", "color": "#059669"},
]

def similarity(expected, actual):
    if not expected or not actual:
        return 0.0
    return SequenceMatcher(None, expected.lower().strip(), actual.lower().strip()).ratio()

def is_safe_outside_kb(result):
    answer = result.get("answer", "").lower()
    safe_phrases = [
        "not specified", "not explicitly stated", "not available", "not provided",
        "does not provide", "does not specify", "not mentioned", "not stated",
        "information is not available"
    ]
    return any(phrase in answer for phrase in safe_phrases)

def classify_correctness(result):
    category = result.get("category", "")
    if category == "Outside Knowledge Base" and is_safe_outside_kb(result):
        return "SAFE_REFUSAL"
    
    score = similarity(result.get("expected_answer", ""), result.get("answer", ""))
    if score >= 0.80:
        return "CORRECT"
    elif score >= 0.50:
        return "PARTIAL"
    else:
        return "WRONG"

def potential_hallucination(result):
    expected = result.get("expected_answer", "").lower()
    answer = result.get("answer", "").lower()
    unspecified = ["not specified", "not explicitly stated", "not available", "does not specify", "not provided"]
    
    if not any(x in expected for x in unspecified):
        return False
    
    if re.search(r"\b\d+(\.\d+)?\s*%", answer):
        return True
    if re.search(r"\b\d+\s*(days?|weeks?|months?|hours?)\b", answer):
        return True
    return False


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "RAG System Evaluation — Side-by-Side Model Comparison")
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "CodeLlama vs Llama 3.2 vs Mistral 7B")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)
        
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_str)
        self.drawString(54, 36, "Confidential — Comparative LLM RAG Benchmark Analysis")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 8.5 * inch - 54, 46)
        self.restoreState()


def load_model_data():
    model_data = {}
    for m in MODEL_CONFIGS:
        key = m["key"]
        json_path = RESULTS_DIR / f"{key}.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                items = json.load(f)
                # Key items by question id
                model_data[key] = {item["id"]: item for item in items}
    return model_data


def build_comparison_pdf():
    model_data = load_model_data()
    if not model_data:
        print("No model result files found in results directory!")
        return

    output_pdf = OUTPUT_DIR / "model_comparison_report.pdf"
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    doc_title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )

    doc_subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14
    )

    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=14,
        spaceAfter=8
    )

    question_title_style = ParagraphStyle(
        'QuestionTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E293B")
    )

    question_meta_style = ParagraphStyle(
        'QuestionMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569")
    )

    tbl_header_style = ParagraphStyle(
        'TblHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    tbl_cell_style = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#334155")
    )

    tbl_cell_bold = ParagraphStyle(
        'TblCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0F172A")
    )

    status_badge_styles = {
        "CORRECT": ParagraphStyle('BadgeCorrect', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor("#15803D")),
        "SAFE_REFUSAL": ParagraphStyle('BadgeSafe', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor("#0369A1")),
        "PARTIAL": ParagraphStyle('BadgePartial', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor("#B45309")),
        "WRONG": ParagraphStyle('BadgeWrong', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor("#B91C1C")),
    }

    story = []

    # Title Block
    story.append(Paragraph("RAG Model Benchmark — Side-by-Side Comparison", doc_title_style))
    story.append(Paragraph("Comprehensive Question-by-Question Evaluation of CodeLlama 7B vs Llama 3.2 (3B) vs Mistral 7B", doc_subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0F172A"), spaceBefore=0, spaceAfter=14))

    # --- SECTION 1: EXECUTIVE SUMMARY DASHBOARD ---
    story.append(Paragraph("1. Executive Summary & Model Overview", section_heading_style))

    # Compute overall stats for each model
    summary_rows = [
        [
            Paragraph("Model Name", tbl_header_style),
            Paragraph("Accuracy / Pass Rate", tbl_header_style),
            Paragraph("Correct", tbl_header_style),
            Paragraph("Safe Refusal", tbl_header_style),
            Paragraph("Partial", tbl_header_style),
            Paragraph("Wrong", tbl_header_style),
            Paragraph("Avg Sim Score", tbl_header_style),
            Paragraph("Avg Latency", tbl_header_style),
        ]
    ]

    # Collect stats per model
    all_qids = sorted(list(model_data["llama3.2_3b"].keys()))
    
    for mcfg in MODEL_CONFIGS:
        key = mcfg["key"]
        name = mcfg["name"]
        items = model_data.get(key, {})
        
        correct_cnt = 0
        safe_cnt = 0
        partial_cnt = 0
        wrong_cnt = 0
        sim_scores = []
        latencies = []

        for qid in all_qids:
            item = items.get(qid)
            if not item:
                continue
            st = classify_correctness(item)
            if st == "CORRECT":
                correct_cnt += 1
            elif st == "SAFE_REFUSAL":
                safe_cnt += 1
            elif st == "PARTIAL":
                partial_cnt += 1
            else:
                wrong_cnt += 1
            
            sim_scores.append(similarity(item.get("expected_answer", ""), item.get("answer", "")))
            latencies.append(item.get("latency_seconds", 0))

        total_q = len(all_qids)
        pass_rate = ((correct_cnt + safe_cnt) / total_q) * 100 if total_q else 0
        avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        summary_rows.append([
            Paragraph(f"<b>{name}</b>", tbl_cell_bold),
            Paragraph(f"<b>{pass_rate:.1f}%</b>", tbl_cell_bold),
            Paragraph(str(correct_cnt), tbl_cell_style),
            Paragraph(str(safe_cnt), tbl_cell_style),
            Paragraph(str(partial_cnt), tbl_cell_style),
            Paragraph(str(wrong_cnt), tbl_cell_style),
            Paragraph(f"{avg_sim:.3f}", tbl_cell_style),
            Paragraph(f"{avg_lat:.2f}s", tbl_cell_style),
        ])

    sum_table = Table(summary_rows, colWidths=[1.4*inch, 1.1*inch, 0.65*inch, 0.75*inch, 0.6*inch, 0.6*inch, 0.95*inch, 0.95*inch])
    sum_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))

    story.append(sum_table)
    story.append(Spacer(1, 14))

    # --- SECTION 2: CATEGORY BREAKDOWN ---
    story.append(Paragraph("2. Category Performance Comparison", section_heading_style))

    category_map = defaultdict(lambda: defaultdict(list))
    for qid in all_qids:
        # Base category from any model (e.g. llama3.2_3b)
        cat = model_data["llama3.2_3b"][qid].get("category", "General")
        for mcfg in MODEL_CONFIGS:
            mkey = mcfg["key"]
            item = model_data[mkey].get(qid)
            if item:
                category_map[cat][mkey].append(item)

    cat_rows = [
        [
            Paragraph("Category", tbl_header_style),
            Paragraph("CodeLlama 7B (Acc / Lat)", tbl_header_style),
            Paragraph("Llama 3.2 3B (Acc / Lat)", tbl_header_style),
            Paragraph("Mistral 7B (Acc / Lat)", tbl_header_style),
        ]
    ]

    for cat_name, m_items in sorted(category_map.items()):
        row = [Paragraph(f"<b>{cat_name}</b>", tbl_cell_bold)]
        for mcfg in MODEL_CONFIGS:
            items = m_items.get(mcfg["key"], [])
            c_cnt = sum(1 for it in items if classify_correctness(it) in ("CORRECT", "SAFE_REFUSAL"))
            tot = len(items)
            acc = (c_cnt / tot * 100) if tot else 0
            avg_l = sum(it.get("latency_seconds", 0) for it in items) / tot if tot else 0
            row.append(Paragraph(f"<b>{acc:.0f}%</b> ({avg_l:.2f}s)", tbl_cell_style))
        cat_rows.append(row)

    cat_table = Table(cat_rows, colWidths=[1.8*inch, 1.73*inch, 1.73*inch, 1.74*inch])
    cat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))

    story.append(cat_table)
    story.append(Spacer(1, 16))
    story.append(PageBreak())

    # --- SECTION 3: QUESTION-BY-QUESTION SIDE-BY-SIDE COMPARISON ---
    story.append(Paragraph("3. Detailed Question-by-Question Comparison (Q1 – Q30)", section_heading_style))
    story.append(Paragraph("Below is the side-by-side comparison of each question, expected answer, and responses from all 3 models.", doc_subtitle_style))
    story.append(Spacer(1, 8))

    for qid in all_qids:
        q_item_sample = model_data["llama3.2_3b"][qid]
        q_category = q_item_sample.get("category", "General")
        q_text = q_item_sample.get("question", "")
        exp_answer = q_item_sample.get("expected_answer", "")

        q_flowables = []

        # Question Header Box
        header_text = f"<b>Q{qid:02d}</b> &nbsp;|&nbsp; Category: <b>{q_category}</b>"
        q_header_table = Table(
            [[Paragraph(header_text, ParagraphStyle('QH', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, textColor=colors.white))]],
            colWidths=[7.0*inch]
        )
        q_header_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#1E293B")),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        q_flowables.append(q_header_table)

        # Question & Expected Answer Box
        q_body_content = [
            [Paragraph(f"<b>Question:</b> {q_text}", question_title_style)],
            [Paragraph(f"<b>Expected Answer:</b> <i>{exp_answer}</i>", question_meta_style)]
        ]
        q_body_table = Table(q_body_content, colWidths=[7.0*inch])
        q_body_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ]))
        q_flowables.append(q_body_table)

        # 3-Model Answer Comparison Table
        model_comp_rows = [
            [
                Paragraph("Model", tbl_header_style),
                Paragraph("Generated Response", tbl_header_style),
                Paragraph("Status & Metrics", tbl_header_style),
            ]
        ]

        for mcfg in MODEL_CONFIGS:
            mkey = mcfg["key"]
            mname = mcfg["name"]
            mcolor = mcfg["color"]
            item = model_data[mkey].get(qid, {})
            
            ans_text = item.get("answer", "").strip()
            if not ans_text:
                ans_text = "<i>[No Answer Provided]</i>"
            
            sim_val = similarity(exp_answer, ans_text)
            status = classify_correctness(item)
            lat = item.get("latency_seconds", 0)

            # Cell formatting
            m_label = Paragraph(f"<font color=\"{mcolor}\"><b>{mname}</b></font>", tbl_cell_bold)
            
            # Truncate extremely verbose answers if longer than 350 chars to keep report readable
            ans_display = ans_text
            if len(ans_display) > 350:
                ans_display = ans_display[:347] + "..."
            ans_cell = Paragraph(ans_display, tbl_cell_style)

            b_style = status_badge_styles.get(status, tbl_cell_bold)
            status_hex = b_style.textColor.hexval()
            metrics_cell = Paragraph(
                f"<font color=\"#{status_hex}\"><b>{status}</b></font><br/>"
                f"Sim Score: <b>{sim_val:.3f}</b><br/>"
                f"Latency: <b>{lat:.2f}s</b>",
                tbl_cell_style
            )

            model_comp_rows.append([m_label, ans_cell, metrics_cell])

        comp_table = Table(model_comp_rows, colWidths=[1.25*inch, 4.45*inch, 1.30*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#334155")),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))

        q_flowables.append(comp_table)
        q_flowables.append(Spacer(1, 14))

        # Keep each question block clean
        story.append(KeepTogether(q_flowables))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Comparative Model Evaluation PDF generated successfully: {output_pdf}")

if __name__ == "__main__":
    build_comparison_pdf()
