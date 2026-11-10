"""Derive searchable PDF text and remove irrelevant curl TLS diagnostics."""
import hashlib
import gzip
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    for path in (ROOT / 'brats/threads').rglob('*.md'):
        wire = path.read_bytes()
        if wire.startswith(b'\x1f\x8b'):
            body = gzip.decompress(wire)
            body.decode('utf-8')
            transport = path.with_name(path.name + '.transport.gz')
            assert not transport.exists()
            transport.write_bytes(wire)
            path.write_bytes(body)
            receipt = path.with_name(path.name + '.source.json')
            record = json.loads(receipt.read_text())
            assert record['sha256'] == hashlib.sha256(wire).hexdigest()
            record.update(wire_sha256=record['sha256'], wire_bytes=record['bytes'],
                wire_archive=str(transport.relative_to(ROOT)), bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest(), transport_decoded='gzip')
            receipt.write_text(json.dumps(record, indent=2) + '\n')
    for path in ROOT.rglob('*.source.json'):
        record = json.loads(path.read_text())
        if 'network' in record:
            record['network'] = {key: value for key, value in record['network'].items()
                if key in ('http_code', 'content_type', 'url_effective',
                           'size_download', 'time_total', 'ssl_verify_result')}
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    papers = []
    for path in sorted((ROOT / 'camyla').rglob('*.pdf')):
        target = ROOT / 'camyla/searchable_text' / (path.stem + '.txt')
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['pdftotext', '-layout', str(path), str(target)], check=True)
        info = subprocess.run(['pdfinfo', str(path)], capture_output=True, text=True, check=True)
        pages = int(next(line.split(':', 1)[1] for line in info.stdout.splitlines()
                         if line.startswith('Pages:')))
        papers.append(dict(pdf=str(path.relative_to(ROOT)), text=str(target.relative_to(ROOT)),
            pages=pages, bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            text_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            ligature_extraction_may_be_imperfect=True))
    (ROOT / 'camyla/PDF_TEXT_INVENTORY.json').write_text(json.dumps(papers, indent=2) + '\n')
    print(json.dumps(dict(pdf_count=len(papers), pages=sum(x['pages'] for x in papers))))


if __name__ == '__main__':
    main()
