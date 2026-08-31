from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
root=Path(__file__).resolve().parents[1]; pdfmetrics.registerFont(TTFont('SimHei','C:/Windows/Fonts/simhei.ttf'))
txt=(root/'paper'/'AI工具使用详情.md').read_text(encoding='utf8'); styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='CN',fontName='SimHei',fontSize=10,leading=15,wordWrap='CJK',spaceAfter=5)); story=[]
for l in txt.splitlines():
    if not l.strip(): story.append(Spacer(1,3)); continue
    story.append(Paragraph(l.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'),styles['CN']))
SimpleDocTemplate(str(root/'paper'/'AI工具使用详情.pdf'),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=16*mm,bottomMargin=18*mm).build(story)
