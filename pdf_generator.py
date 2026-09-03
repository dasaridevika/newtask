import os
import re
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.pdfgen import canvas

def md_to_html(text: str) -> str:
    """Converts markdown to HTML with strict XML escaping for safe ReportLab paragraph parsing."""
    if not text:
        return ""
    
    text = str(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" color="#2563EB"><b>\1</b></a>', text)
    parts = text.split("**")
    result = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            result.append(f"<b>{part}</b>")
        else:
            result.append(part)
    return "".join(result)

PRIMARY_COLOR = colors.HexColor("#0B1938")
SECONDARY_COLOR = colors.HexColor("#2563EB")
ACCENT_COLOR = colors.HexColor("#0D9488")
TEXT_COLOR = colors.HexColor("#1E293B")
BG_LIGHT = colors.HexColor("#F8FAFC")
BG_CARD = colors.HexColor("#F1F5F9")
BORDER_COLOR = colors.HexColor("#CBD5E1")

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
            self.draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def draw_page_elements(self, page_count):
        self.saveState()
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(SECONDARY_COLOR)
            self.drawString(54, 750, "EXECUTIVE LEAD INTELLIGENCE & STRATEGIC MATCH DOSSIER")
            self.setFont("Helvetica", 8)
            self.setFillColor(TEXT_COLOR)
            self.drawRightString(612 - 54, 750, datetime.now().strftime('%B %d, %Y'))
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 36, "Confidential | Blackridge Research & Intelligence Engine")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 36, page_str)
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(54, 48, 612 - 54, 48)
        self.restoreState()

def generate_lead_pdf(dossier: dict, filename: str = "lead_dossier.pdf") -> str:
    """Generates an executive intelligence PDF report."""
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY_COLOR,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=SECONDARY_COLOR,
        spaceAfter=12
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=PRIMARY_COLOR,
        spaceBefore=14,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_COLOR
    )

    story = []

    # Title & Header
    lead_name = dossier.get("name") or dossier.get("lead_name") or "Executive Lead"
    company_name = dossier.get("company") or dossier.get("company_name") or "Target Enterprise"
    story.append(Paragraph(f"{lead_name} &mdash; {company_name}", title_style))
    story.append(Paragraph(f"Executive Lead Intelligence & Strategic Offering Match Dossier | Generated {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY_COLOR, spaceAfter=14))

    # KPI Summary Table
    kpi_data = [
        [
            Paragraph("<b>Email Status:</b>", body_style), Paragraph(dossier.get("email_validity", "Verified"), body_style),
            Paragraph("<b>Buying Role:</b>", body_style), Paragraph(dossier.get("buying_role", "Decision Maker"), body_style)
        ],
        [
            Paragraph("<b>Business Email:</b>", body_style), Paragraph(dossier.get("email", "N/A"), body_style),
            Paragraph("<b>Country/Region:</b>", body_style), Paragraph(dossier.get("country", "Global"), body_style)
        ],
        [
            Paragraph("<b>Archetype:</b>", body_style), Paragraph(dossier.get("archetype", "Enterprise"), body_style),
            Paragraph("<b>Industry:</b>", body_style), Paragraph(dossier.get("industry", "Infrastructure"), body_style)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[90, 160, 90, 160])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 1, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # Executive Profile Analysis
    exec_summary = dossier.get("professional_summary") or dossier.get("executive_profile_analysis") or ""
    if exec_summary:
        story.append(Paragraph("Executive Strategic Synthesis", section_heading))
        for p in exec_summary.split("\n\n"):
            if p.strip():
                story.append(Paragraph(md_to_html(p.strip()), body_style))
                story.append(Spacer(1, 4))
        story.append(Spacer(1, 10))

    # Strategic Offerings Match
    offerings = dossier.get("strategic_offerings") or dossier.get("matched_offerings") or []
    if offerings:
        story.append(Paragraph("Recommended Strategic Offerings", section_heading))
        for idx, off in enumerate(offerings[:3], 1):
            p_name = off.get("product_name", f"Strategic Offering {idx}")
            rel_sum = off.get("relevance_summary") or off.get("value_driver") or ""
            off_data = [
                [Paragraph(f"<b>#{idx} {p_name}</b>", body_style)],
                [Paragraph(md_to_html(rel_sum), body_style)]
            ]
            off_table = Table(off_data, colWidths=[500])
            off_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
                ('BOX', (0, 0), (-1, -1), 1, SECONDARY_COLOR),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(off_table)
            story.append(Spacer(1, 6))

    doc.build(story, canvasmaker=NumberedCanvas)
    return filename
