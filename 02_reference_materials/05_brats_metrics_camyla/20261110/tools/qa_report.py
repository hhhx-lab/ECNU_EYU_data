"""Inspect the report and render temporary review pages using existing document tools."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render-dir', type=Path, required=True)
    args = parser.parse_args()
    args.render_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(ROOT / 'REPORT.docx') as package:
        document = ET.fromstring(package.read('word/document.xml'))
    namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                  'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    docx = dict(paragraphs=len(document.findall('.//w:p', namespaces)),
                tables=len(document.findall('.//w:tbl', namespaces)),
                office_math=len(document.findall('.//m:oMath', namespaces)))
    assert docx['tables'] == 7 and docx['office_math'] == 3
    records = []
    fonts, colors = Counter(), Counter()
    with pdfplumber.open(ROOT / 'REPORT.pdf') as pdf:
        for index, page in enumerate(pdf.pages):
            text = page.extract_text() or ''
            assert len(text.strip()) > 100, ('Empty report page', index + 1)
            assert '\ufffd' not in text
            assert not any(value in text for value in ('$V=', '\\frac', '\\sum'))
            visible = [c for c in page.chars if c['text'].strip()]
            outside = [c for c in visible if c['x0'] < -0.5 or c['x1'] > page.width + 0.5
                       or c['top'] < -0.5 or c['bottom'] > page.height + 0.5]
            assert not outside, ('Text outside page', index + 1)
            fonts.update(c['fontname'] for c in visible)
            colors.update(str(c.get('non_stroking_color')) for c in visible)
            records.append(dict(page=index + 1, text_chars=len(text),
                tables=len(page.extract_tables()), images=len(page.images),
                outside_page_chars=len(outside)))
    assert any('STSongti-SC-Regular' in name for name in fonts)
    assert not any(any(bad in name for bad in ('WaWa', 'Hannotate', 'Xingkai')) for name in fonts)
    pdf = pdfium.PdfDocument(str(ROOT / 'REPORT.pdf'))
    thumbs = []
    for index in range(len(pdf)):
        page = pdf[index]
        bitmap = page.render(scale=1.8)
        picture = bitmap.to_pil().convert('RGB')
        picture.save(args.render_dir / ('page_%02d.png' % (index + 1)))
        picture.thumbnail((280, 395))
        thumb = Image.new('RGB', (300, 425), 'white')
        thumb.paste(picture, (10, 20))
        ImageDraw.Draw(thumb).text((10, 4), str(index + 1), fill='black')
        thumbs.append(thumb)
        bitmap.close()
        page.close()
    pdf.close()
    for start in range(0, len(thumbs), 8):
        group = thumbs[start:start + 8]
        canvas = Image.new('RGB', (1200, 850), '#dddddd')
        for index, thumb in enumerate(group):
            canvas.paste(thumb, ((index % 4) * 300, (index // 4) * 425))
        canvas.save(args.render_dir / ('contact_%02d.png' % (start + 1)))
    record = dict(reviewed_utc=datetime.now(timezone.utc).isoformat(),
        report_pdf_sha256=hashlib.sha256((ROOT / 'REPORT.pdf').read_bytes()).hexdigest(),
        report_docx_sha256=hashlib.sha256((ROOT / 'REPORT.docx').read_bytes()).hexdigest(),
        docx=docx, pdf_pages=len(records), pages=records, fonts=dict(fonts), colors=dict(colors),
        pdf_route='Pandoc HTML5/native MathML -> Chrome with scoped embedded local fonts',
        global_fonts_or_template_modified=False, new_dependencies_installed=False,
        automated_checks_passed=True, visual_review='pending',
        third_party_pdfs_all_pages_visually_reviewed=False)
    (ROOT / 'DOCUMENT_QA.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(pdf_pages=len(records), docx=docx,
                         table_pages=[x['page'] for x in records if x['tables']])))


if __name__ == '__main__':
    main()
