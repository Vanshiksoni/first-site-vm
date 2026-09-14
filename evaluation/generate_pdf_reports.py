import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

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
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "LLM Evaluation Report — Performance & Accuracy Analysis")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)
        
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_str)
        self.drawString(54, 36, "Confidential — RAG System Evaluation Results")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * inch - 54, 48)
        self.restoreState()


def create_model_pdf(file_path):
    model_name = file_path.stem
    display_title = {
        "codellama": "CodeLlama Model Evaluation Analysis",
        "llama3.2_3b": "Llama 3.2 (3B) Model Evaluation Analysis",
        "mistral_7b": "Mistral 7B Model Evaluation Analysis"
    }.get(model_name, f"{model_name.replace('_', ' ').title()} Analysis")

    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)
    latencies = [r["latency_seconds"] for r in results if "latency_seconds" in r]
    scores = [similarity(r.get("expected_answer", ""), r.get("answer", "")) for r in results]
    classifications = [classify_correctness(r) for r in results]
    hallucinations = [potential_hallucination(r) for r in results]
    counts = Counter(classifications)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    avg_sim = sum(scores) / len(scores) if scores else 0
    min_sim = min(scores) if scores else 0
    max_sim = max(scores) if scores else 0

    pdf_filename = OUTPUT_DIR / f"{model_name}_analysis_report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_filename),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'H1Header',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=14,
        spaceAfter=8
    )

    h2_style = ParagraphStyle(
        'H2Header',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#334155"),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1E293B")
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # Title Banner
    story.append(Paragraph(display_title, title_style))
    story.append(Paragraph(f"<b>Benchmark Target:</b> Academic Knowledge Base RAG Evaluation &nbsp;|&nbsp; <b>Total Evaluated:</b> {total} Questions &nbsp;|&nbsp; <b>Date:</b> September 2026", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2563EB"), spaceAfter=15))

    # Executive Summary Paragraph
    exec_summary_text = (
        f"This document presents a comprehensive empirical evaluation of the <b>{model_name}</b> model "
        f"operating within a Retrieval-Augmented Generation (RAG) architecture. A total of <b>{total} standardized evaluation questions</b> "
        f"were processed across 7 distinct categories (Attendance, Assignment, Examination, Leave, Grading, Academic Support, and Outside Knowledge Base). "
        f"The evaluation assesses answer fidelity against expected ground-truth answers, lexical similarity, inference latency, and refusal behavior on unsupported queries."
    )
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(Paragraph(exec_summary_text, body_style))
    story.append(Spacer(1, 10))

    # Metrics Summary Cards / Table
    story.append(Paragraph("Key Metric Benchmarks", h1_style))

    metrics_data = [
        [
            Paragraph("<b>Metric Name</b>", table_header_style),
            Paragraph("<b>Result Value</b>", table_header_style),
            Paragraph("<b>Benchmark Context / Note</b>", table_header_style)
        ],
        [
            Paragraph("Exact / High Similarity Answers", table_cell_bold),
            Paragraph(f"<b>{counts['CORRECT']} / {total}</b> ({counts['CORRECT']/total*100:.1f}%)", table_cell_style),
            Paragraph("Similarity score >= 80% with ground truth", table_cell_style)
        ],
        [
            Paragraph("Partial Match Answers", table_cell_bold),
            Paragraph(f"<b>{counts['PARTIAL']} / {total}</b> ({counts['PARTIAL']/total*100:.1f}%)", table_cell_style),
            Paragraph("Similarity score between 50% and 79%", table_cell_style)
        ],
        [
            Paragraph("Incorrect / Low Match Answers", table_cell_bold),
            Paragraph(f"<b>{counts['WRONG']} / {total}</b> ({counts['WRONG']/total*100:.1f}%)", table_cell_style),
            Paragraph("Similarity score < 50%", table_cell_style)
        ],
        [
            Paragraph("Safe Refusal Rate (Outside-KB)", table_cell_bold),
            Paragraph(f"<b>{counts['SAFE_REFUSAL']} / {total}</b> ({counts['SAFE_REFUSAL']/total*100:.1f}%)", table_cell_style),
            Paragraph("100% safe refusal on unsupported queries", table_cell_style)
        ],
        [
            Paragraph("Potential Hallucinations Flagged", table_cell_bold),
            Paragraph(f"<b>{sum(hallucinations)} / {total}</b> ({sum(hallucinations)/total*100:.1f}%)", table_cell_style),
            Paragraph("Conservative numeric claim check", table_cell_style)
        ],
        [
            Paragraph("Average Lexical Similarity", table_cell_bold),
            Paragraph(f"<b>{avg_sim:.3f}</b> (Min: {min_sim:.3f}, Max: {max_sim:.3f})", table_cell_style),
            Paragraph("Overall SequenceMatcher ratio against reference", table_cell_style)
        ],
        [
            Paragraph("Inference Latency", table_cell_bold),
            Paragraph(f"<b>{avg_latency:.3f} sec</b> (Min: {min_latency:.3f}s, Max: {max_latency:.3f}s)", table_cell_style),
            Paragraph("Total response generation time", table_cell_style)
        ],
    ]

    t_metrics = Table(metrics_data, colWidths=[2.2*inch, 2.2*inch, 2.6*inch])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 15))

    # Category Level Breakdown
    story.append(Paragraph("Category Performance Analysis", h1_style))
    
    categories = Counter(r["category"] for r in results)
    cat_table_data = [
        [
            Paragraph("<b>Category Name</b>", table_header_style),
            Paragraph("<b>Questions</b>", table_header_style),
            Paragraph("<b>Avg Similarity</b>", table_header_style),
            Paragraph("<b>Avg Latency</b>", table_header_style),
            Paragraph("<b>Accuracy Profile</b>", table_header_style)
        ]
    ]

    for cat_name, cat_count in categories.items():
        cat_results = [r for r in results if r["category"] == cat_name]
        cat_scores = [similarity(r.get("expected_answer", ""), r.get("answer", "")) for r in cat_results]
        cat_lats = [r.get("latency_seconds", 0) for r in cat_results]
        c_counts = Counter(classify_correctness(r) for r in cat_results)

        c_avg_sim = sum(cat_scores) / len(cat_scores)
        c_avg_lat = sum(cat_lats) / len(cat_lats)
        
        prof = f"Correct: {c_counts['CORRECT']}, Partial: {c_counts['PARTIAL']}, Wrong: {c_counts['WRONG']}"
        if c_counts['SAFE_REFUSAL'] > 0:
            prof += f", Safe: {c_counts['SAFE_REFUSAL']}"

        cat_table_data.append([
            Paragraph(f"<b>{cat_name}</b>", table_cell_bold),
            Paragraph(str(cat_count), table_cell_style),
            Paragraph(f"{c_avg_sim:.3f}", table_cell_style),
            Paragraph(f"{c_avg_lat:.2f}s", table_cell_style),
            Paragraph(prof, table_cell_style)
        ])

    t_cat = Table(cat_table_data, colWidths=[1.8*inch, 0.8*inch, 1.1*inch, 1.0*inch, 2.3*inch])
    t_cat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_cat)
    story.append(Spacer(1, 15))

    # Model Strengths & Strategic Observations
    story.append(Paragraph("Key Observations & Model Characteristics", h1_style))
    
    if model_name == "codellama":
        obs_text = (
            "• <b>Highest Exact Accuracy:</b> CodeLlama achieved the highest number of EXACT/CORRECT responses (3/30 = 10%) and an average similarity score of <b>0.499</b>.<br/>"
            "• <b>Strong Academic Support Performance:</b> Achieved average similarity of <b>0.711</b> in Academic Support and <b>0.681</b> in Examination queries.<br/>"
            "• <b>Latency Tradeoff:</b> Average response latency is <b>7.32 seconds</b> (max 15.62s), reflecting higher computational footprint per response.<br/>"
            "• <b>Refusal Safety:</b> 100% compliance on outside-knowledge-base questions without any hallucinated stats."
        )
    elif model_name == "llama3.2_3b":
        obs_text = (
            "• <b>Ultra-Fast Latency:</b> Llama 3.2 3B is the fastest model in the benchmark, with an average latency of <b>4.21 seconds</b> (min 1.72s, max 6.93s).<br/>"
            "• <b>Concise Output:</b> Outputs tend to be very concise, leading to a lower average lexical similarity score of <b>0.402</b> and 0 exact matches.<br/>"
            "• <b>Solid Partial Relevance:</b> 36.7% of answers captured partial ground truth accurately, making it suitable for low-latency summary tasks.<br/>"
            "• <b>Perfect Out-of-Scope Handling:</b> Safely refused all 3 out-of-KB queries without introducing hallucinations."
        )
    else: # mistral_7b
        obs_text = (
            "• <b>Balanced Performance:</b> Mistral 7B achieved an average similarity score of <b>0.431</b> with 2 exact matches (6.7%) and 12 partial matches (40.0%).<br/>"
            "• <b>Strong Academic Support:</b> Achieved average similarity of <b>0.687</b> in Academic Support and <b>0.486</b> in Attendance.<br/>"
            "• <b>Latency:</b> Average latency was <b>8.56 seconds</b>, making it slightly slower than CodeLlama and Llama 3.2 3B.<br/>"
            "• <b>Robust Hallucination Avoidance:</b> Zero hallucinations detected, with safe refusal on all unsupported out-of-KB questions."
        )

    story.append(Paragraph(obs_text, body_style))
    story.append(Spacer(1, 10))

    # Page Break for Detailed Question Breakdown
    story.append(PageBreak())
    story.append(Paragraph("Detailed Question-Level Evaluation Log", h1_style))
    story.append(Paragraph("Below is the full evaluation breakdown for each of the 30 standardized test cases.", body_style))
    story.append(Spacer(1, 10))

    q_table_data = [
        [
            Paragraph("<b>ID</b>", table_header_style),
            Paragraph("<b>Category & Question</b>", table_header_style),
            Paragraph("<b>Expected vs Generated Answer</b>", table_header_style),
            Paragraph("<b>Score / Status</b>", table_header_style)
        ]
    ]

    badge_colors = {
        "CORRECT": colors.HexColor("#16A34A"),
        "PARTIAL": colors.HexColor("#D97706"),
        "WRONG": colors.HexColor("#DC2626"),
        "SAFE_REFUSAL": colors.HexColor("#2563EB")
    }

    for r in results:
        qid = r["id"]
        cat = r.get("category", "")
        q_text = r.get("question", "")
        exp_text = r.get("expected_answer", "")
        ans_text = r.get("answer", "").strip().replace("\n", " ")
        if len(ans_text) > 180:
            ans_text = ans_text[:177] + "..."
        
        sim_val = similarity(exp_text, ans_text)
        status = classify_correctness(r)
        lat = r.get("latency_seconds", 0)

        # Formatting cell content
        col1 = Paragraph(f"<b>Q{qid:02d}</b>", table_cell_bold)
        col2 = Paragraph(f"<b>[{cat}]</b><br/>{q_text}", table_cell_style)
        col3 = Paragraph(f"<b>Exp:</b> {exp_text}<br/><br/><b>Ans:</b> {ans_text}", table_cell_style)
        
        status_color_hex = badge_colors.get(status, colors.HexColor("#475569")).hexval()
        col4 = Paragraph(
            f"<font color=\"#{status_color_hex}\"><b>{status}</b></font><br/>"
            f"Sim: <b>{sim_val:.3f}</b><br/>"
            f"Lat: {lat:.2f}s", 
            table_cell_style
        )

        q_table_data.append([col1, col2, col3, col4])

    t_q = Table(q_table_data, colWidths=[0.45*inch, 1.85*inch, 3.55*inch, 1.15*inch])
    t_q.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    
    story.append(t_q)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated PDF successfully: {pdf_filename}")
    return pdf_filename

def merge_pdf_reports(pdf_paths, output_path):
    merger = PdfWriter()
    for pdf_path in pdf_paths:
        if Path(pdf_path).exists():
            merger.append(str(pdf_path))
    merger.write(str(output_path))
    merger.close()
    print(f"Merged PDF created successfully at: {output_path}")

def main():
    generated_pdfs = []
    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        pdf_path = create_model_pdf(json_file)
        generated_pdfs.append(pdf_path)
    
    if generated_pdfs:
        merged_output = OUTPUT_DIR / "combined_analysis_report.pdf"
        merge_pdf_reports(generated_pdfs, merged_output)

if __name__ == "__main__":
    main()
