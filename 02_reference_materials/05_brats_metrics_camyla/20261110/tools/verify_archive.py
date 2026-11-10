"""Verify frozen public references, complete selected forum threads, and local links."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def identity(path):
    body = path.read_bytes()
    return dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


def link_targets(node):
    if isinstance(node, dict):
        if node.get('t') in ('Link', 'Image'):
            yield node['c'][-1][0]
        for child in node.values():
            yield from link_targets(child)
    elif isinstance(node, list):
        for child in node:
            yield from link_targets(child)


def main():
    assets = read(ROOT / 'camyla/PUBLIC_ASSET_MANIFEST.json')
    tree = read(ROOT / 'camyla/sources/frozen_tree.json')
    assert tree['truncated'] is False
    frozen = {entry['path']: entry for entry in tree['tree']}
    for entry in assets['files']:
        path = ROOT / entry['destination']
        actual = identity(path)
        assert actual['bytes'] == entry['bytes'] and actual['sha256'] == entry['sha256']
        body = path.read_bytes()
        blob = hashlib.sha1(('blob %d\0' % len(body)).encode() + body).hexdigest()
        assert blob == entry['git_blob_sha1'] == frozen[entry['source']]['sha']
        assert len(body) == frozen[entry['source']]['size']
    assert len(assets['files']) == 45
    receipts, unsuccessful = [], []
    for path in sorted(ROOT.rglob('*.source.json')):
        receipt = read(path)
        target = path.with_name(path.name[:-len('.source.json')])
        if receipt.get('success') is False:
            unsuccessful.append(dict(receipt=str(path.relative_to(ROOT)),
                url=receipt.get('url'), status=receipt.get('network', {}).get('http_code')))
            continue
        actual = identity(target)
        assert actual['bytes'] == receipt['bytes'] and actual['sha256'] == receipt['sha256'], path
        if receipt.get('transport_decoded') == 'gzip':
            wire = ROOT / receipt['wire_archive']
            assert identity(wire) == dict(bytes=receipt['wire_bytes'], sha256=receipt['wire_sha256'])
            assert gzip.decompress(wire.read_bytes()) == target.read_bytes()
        receipts.append(str(path.relative_to(ROOT)))
    forums = []
    for year, expected in [('2025', (107, 9, 50)), ('2026', (93, 11, 61))]:
        index = read(ROOT / ('brats/FORUM_%s_INDEX.json' % year))
        assert index['pagination_complete'] and len(index['threads']) == expected[0]
        thread_map = {x['id']: x for x in index['threads']}
        assert len(thread_map) == expected[0]
        directories = sorted((ROOT / 'brats/threads' / year).iterdir())
        assert len(directories) == expected[1]
        replies = 0
        for folder in directories:
            thread = read(folder / 'thread.md.source.json')
            assert thread['author_id'] == thread_map[folder.name]['createdBy']
            reply_map = {}
            for page in folder.glob('replies_*.json'):
                if page.name.endswith('.source.json'):
                    continue
                for item in read(page)['results']:
                    assert item['id'] not in reply_map and not item['isDeleted']
                    reply_map[item['id']] = item
            assert set(reply_map) == {p.stem[6:] for p in folder.glob('reply_*.md')}
            for rid, item in reply_map.items():
                source = read(folder / ('reply_%s.md.source.json' % rid))
                assert source['author_id'] == item['createdBy']
                assert source['createdOn'] == item['createdOn']
            replies += len(reply_map)
        assert replies == expected[2]
        forums.append(dict(year=year, indexed_threads=expected[0],
                           complete_selected_threads=len(directories), complete_replies=replies))
    papers = read(ROOT / 'camyla/PDF_TEXT_INVENTORY.json')
    for item in papers:
        assert identity(ROOT / item['pdf']) == dict(bytes=item['bytes'], sha256=item['sha256'])
        assert identity(ROOT / item['text'])['sha256'] == item['text_sha256']
    checked_links = []
    for name in ('REPORT.md', 'README.md', 'camyla/MANUSCRIPTS.md'):
        path = ROOT / name
        ast = json.loads(subprocess.check_output(['pandoc', str(path), '--to=json'], text=True))
        for url in link_targets(ast):
            parsed = urlsplit(url)
            if not parsed.scheme and parsed.path:
                target = (path.parent / unquote(parsed.path)).resolve()
                assert target.exists(), (name, url)
                checked_links.append(dict(document=name, target=url))
    doc_qa = read(ROOT / 'DOCUMENT_QA.json')
    assert doc_qa['automated_checks_passed'] and doc_qa['visual_review'] == 'passed'
    assert doc_qa['report_pdf_sha256'] == identity(ROOT / 'REPORT.pdf')['sha256']
    assert doc_qa['report_docx_sha256'] == identity(ROOT / 'REPORT.docx')['sha256']
    excluded = {'ARCHIVE_MANIFEST.json', 'EXECUTION_AUDIT.json'}
    files = [dict(path=str(path.relative_to(ROOT)), **identity(path))
             for path in sorted(ROOT.rglob('*')) if path.is_file() and path.name not in excluded
             and '__pycache__' not in path.parts]
    record = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=assets['commit'], public_assets_verified=len(assets['files']),
        public_assets_bytes=sum(x['bytes'] for x in assets['files']),
        successful_source_receipts=len(receipts), unsuccessful_source_requests=unsuccessful,
        forums=forums, third_party_pdf_count=len(papers),
        third_party_pdf_pages=sum(x['pages'] for x in papers), checked_local_links=checked_links,
        scope='This batch only; excludes the two generated audit/manifest files to avoid self-reference',
        files=files, archived_file_count=len(files), archived_bytes=sum(x['bytes'] for x in files),
        current_local_filesystem_free_bytes=shutil.disk_usage(ROOT).free,
        all_identity_checks_passed=True, scientific_validation_pass=False)
    (ROOT / 'ARCHIVE_MANIFEST.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('public_assets_verified', 'successful_source_receipts',
        'archived_file_count', 'archived_bytes', 'third_party_pdf_count', 'third_party_pdf_pages')}))


if __name__ == '__main__':
    main()
