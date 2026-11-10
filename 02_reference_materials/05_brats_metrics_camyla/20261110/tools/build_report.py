"""Build this report with a scoped Chinese style template and fixed table geometry."""
import os
import base64
import io
from pathlib import Path
import subprocess
import tempfile

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Twips
from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = Path.home() / '.local/bin'


def reference(path):
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2.2)
    section.left_margin = section.right_margin = Cm(2.3)
    for name in ('First Paragraph', 'Compact'):
        if name not in doc.styles:
            doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    if 'Table' not in doc.styles:
        doc.styles.add_style('Table', WD_STYLE_TYPE.TABLE).base_style = doc.styles['Table Grid']
    defaults = doc.styles.element.find(qn('w:docDefaults'))
    if defaults is not None:
        for fonts in defaults.iter(qn('w:rFonts')):
            fonts.attrib.clear()
            for key, value in {'ascii': 'Times New Roman', 'hAnsi': 'Times New Roman',
                               'eastAsia': 'Songti SC', 'cs': 'Times New Roman'}.items():
                fonts.set(qn('w:' + key), value)
    for style in doc.styles:
        if not hasattr(style, 'font'):
            continue
        style.font.name = 'Times New Roman'
        style.font.size = Pt(11)
        style.font.color.rgb = RGBColor(0, 0, 0)
        fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
        fonts.attrib.clear()
        for key, value in {'ascii': 'Times New Roman', 'hAnsi': 'Times New Roman',
                           'eastAsia': 'Songti SC', 'cs': 'Times New Roman'}.items():
            fonts.set(qn('w:' + key), value)
        if style.type == WD_STYLE_TYPE.PARAGRAPH:
            properties = style.element.get_or_add_pPr()
            for name in ('pBdr', 'numPr'):
                node = properties.find(qn('w:' + name))
                if node is not None:
                    properties.remove(node)
            pf = style.paragraph_format
            pf.left_indent = pf.right_indent = pf.first_line_indent = Pt(0)
            pf.space_before, pf.space_after = Pt(0), Pt(5)
            pf.line_spacing = 1.35
            pf.keep_with_next = False
            pf.page_break_before = False
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name, size in [('Title', 20), ('Subtitle', 12),
                       ('Heading 1', 16), ('Heading 2', 13), ('Heading 3', 12)]:
        style = doc.styles[name]
        style.font.size, style.font.bold = Pt(size), name != 'Subtitle'
        style.font.italic = False
        style.font.complex_script = False
        properties = style.element.get_or_add_rPr()
        for name in ('iCs', 'spacing', 'kern'):
            node = properties.find(qn('w:' + name))
            if node is not None:
                properties.remove(node)
        fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
        fonts.set(qn('w:eastAsia'), 'Heiti SC')
        fonts.set(qn('w:ascii'), 'Arial')
        fonts.set(qn('w:hAnsi'), 'Arial')
        pf = style.paragraph_format
        pf.space_before = Pt(14 if name.startswith('Heading') else 0)
        pf.space_after = Pt(6)
        pf.keep_with_next = True
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for name in ('Title', 'Subtitle'):
        doc.styles[name].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    compact = doc.styles['Compact']
    compact.font.size = Pt(10)
    compact.paragraph_format.line_spacing = 1.2
    compact.paragraph_format.space_after = Pt(3)
    compact.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    doc.save(path)


def table_geometry(path):
    doc = Document(path)
    section = doc.sections[0]
    available = int((section.page_width - section.left_margin - section.right_margin) / 635)
    for table in doc.tables:
        table.autofit = False
        table.style = 'Table Grid'
        n = len(table.columns)
        weights = {2: [0.3, 0.7], 3: [0.18, 0.41, 0.41],
                   4: [0.16, 0.28, 0.28, 0.28]}[n]
        widths = [int(available * value) for value in weights]
        widths[-1] += available - sum(widths)
        table._tbl.tblPr.find(qn('w:tblW')).set(qn('w:type'), 'dxa')
        table._tbl.tblPr.find(qn('w:tblW')).set(qn('w:w'), str(available))
        for col, width in zip(table.columns, widths):
            col.width = Twips(width)
        for index, row in enumerate(table.rows):
            trpr = row._tr.get_or_add_trPr()
            trpr.append(OxmlElement('w:cantSplit'))
            for cell, width in zip(row.cells, widths):
                cell.width = Twips(width)
                for paragraph in cell.paragraphs:
                    paragraph.style = doc.styles['Compact']
                    paragraph.paragraph_format.keep_with_next = False
                if index == 0:
                    shade = OxmlElement('w:shd')
                    shade.set(qn('w:fill'), 'EFEFEF')
                    cell._tc.get_or_add_tcPr().append(shade)
    doc.save(path)


def embedded_font(path, index, text):
    font = TTFont(path, fontNumber=index)
    if 'OS/2' in font and font['OS/2'].fsType & 2:
        raise ValueError('Font embedding is restricted')
    options = subset.Options()
    options.recalc_timestamp = False
    worker = subset.Subsetter(options=options)
    worker.populate(text=text)
    worker.subset(font)
    output = io.BytesIO()
    font.save(output)
    return base64.b64encode(output.getvalue()).decode('ascii')


def chrome_pdf(temp):
    text = (ROOT / 'REPORT.md').read_text()
    fonts = [
        ('ReportSong', 400, '/System/Library/Fonts/Supplemental/Songti.ttc', 6),
        ('ReportSong', 700, '/System/Library/Fonts/Supplemental/Songti.ttc', 1),
        ('ReportHei', 700, '/System/Library/Fonts/STHeiti Medium.ttc', 1),
    ]
    css = ''.join('@font-face{font-family:%s;font-weight:%s;src:url(data:font/ttf;base64,%s)}'
                  % (name, weight, embedded_font(path, index, text))
                  for name, weight, path, index in fonts)
    css += '''
    @page {size:A4;margin:20mm 20mm 18mm}
    html,body {margin:0;padding:0;background:white;color:black}
    body {font-family:"Times New Roman",ReportSong,serif;font-size:11pt;line-height:1.55}
    h1,h2,h3 {font-family:Arial,ReportHei,sans-serif;color:black;line-height:1.35;
      break-after:avoid;font-weight:700;letter-spacing:0}
    h1 {font-size:17pt;margin:18pt 0 8pt} h2 {font-size:14pt;margin:14pt 0 6pt}
    h3 {font-size:12pt} header {text-align:center;margin-bottom:22pt}
    header h1 {font-size:21pt} header .subtitle {font-family:Arial,ReportHei;font-size:12pt}
    p {margin:6pt 0} a {color:black;overflow-wrap:anywhere}
    blockquote {margin:8pt 12pt;padding-left:10pt;border-left:2pt solid #999}
    table {border-collapse:collapse;width:100%;table-layout:fixed;font-size:9pt;line-height:1.45;
      margin:10pt 0} th,td {border:.5pt solid #aaa;padding:5pt;vertical-align:top;
      overflow-wrap:anywhere} th {background:#efefef;text-align:left}
    tr {break-inside:avoid} thead {display:table-header-group}
    li {margin:4pt 0} code {font-family:Arial,ReportSong;font-size:.94em;overflow-wrap:anywhere}
    math {font-family:"STIX Two Math";font-size:12pt}
    '''
    style = Path(temp) / 'report.css'
    style.write_text(css)
    html = Path(temp) / 'report.html'
    subprocess.run(['pandoc', str(ROOT / 'REPORT.md'), '--standalone', '--mathml',
                    '--to=html5', '--css=' + style.as_uri(), '--output=' + str(html)], check=True)
    subprocess.run(['node', str(ROOT / 'tools/print_report.mjs'), str(html),
                    str(ROOT / 'REPORT.pdf')], check=True)


def main():
    with tempfile.TemporaryDirectory(prefix='ecnu-report-template-') as temp:
        path = Path(temp) / 'reference-report-zh.docx'
        reference(path)
        env = dict(os.environ, CODEX_DOC_REFERENCE_DOC=str(path))
        output = ROOT / 'REPORT.docx'
        subprocess.run([str(WRAPPERS / 'codex-md-to-docx'), str(ROOT / 'REPORT.md'), str(output)],
                       env=env, check=True)
        table_geometry(output)
        chrome_pdf(temp)


if __name__ == '__main__':
    main()
