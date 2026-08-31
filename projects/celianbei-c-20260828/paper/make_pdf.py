from pathlib import Path
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak

root=Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(TTFont('SimHei','C:/Windows/Fonts/simhei.ttf'))
md=(root/'paper'/'论文初稿.md').read_text(encoding='utf-8')
out=root/'paper'/'celianbei_C_参赛论文.pdf'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='CNBody',parent=styles['BodyText'],fontName='SimHei',fontSize=9.5,leading=14,spaceAfter=5,wordWrap='CJK'))
styles.add(ParagraphStyle(name='CNH1',parent=styles['Heading1'],fontName='SimHei',fontSize=14,leading=19,spaceBefore=10,spaceAfter=7,wordWrap='CJK'))
styles.add(ParagraphStyle(name='CNH2',parent=styles['Heading2'],fontName='SimHei',fontSize=11.5,leading=16,spaceBefore=8,spaceAfter=5,wordWrap='CJK'))
styles.add(ParagraphStyle(name='CNTitle',parent=styles['Title'],fontName='SimHei',fontSize=17,leading=24,alignment=TA_CENTER,spaceAfter=14,wordWrap='CJK'))

def esc(s):
    s=s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    s=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1',s)
    return s

story=[]
for line in md.splitlines():
    if not line.strip(): story.append(Spacer(1,3)); continue
    if line.startswith('# '): story.append(Paragraph(esc(line[2:]),styles['CNTitle'])); continue
    if line.startswith('## 1 '): story.append(PageBreak())
    if line.startswith('## '): story.append(Paragraph(esc(line[3:]),styles['CNH1'])); continue
    if line.startswith('### '): story.append(Paragraph(esc(line[4:]),styles['CNH2'])); continue
    if line.startswith('- '): line='• '+line[2:]
    story.append(Paragraph(esc(line),styles['CNBody']))
for fn in ['q1_model_mse.png','q3_static_dynamic.png','q4_comparison.png']:
    p=root/'figures'/fn
    if p.exists():
        story.append(Spacer(1,4)); story.append(Image(str(p),width=150*mm,height=75*mm)); story.append(Spacer(1,4))

def footer(canvas,doc):
    canvas.saveState(); canvas.setFont('SimHei',8); canvas.drawCentredString(A4[0]/2,12*mm,str(doc.page)); canvas.restoreState()

SimpleDocTemplate(str(out),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=16*mm,bottomMargin=18*mm,title='策联杯C题数学建模论文').build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
