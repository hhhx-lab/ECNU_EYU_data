"""Read public Synapse forum pages and selected messages via documented APIs."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://repo-prod.prod.sagebase.org/repo/v1'


def cached_api(url, relative):
    path = ROOT / relative
    if not path.exists():
        subprocess.run([sys.executable, '-B', str(ROOT / 'tools/acquire.py'), url, relative], check=True)
    receipt = json.loads(path.with_name(path.name + '.source.json').read_text())
    assert receipt['url'] == url and receipt['success']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256']
    return json.loads(path.read_text())


def message(kind, entry, relative):
    path = ROOT / relative
    if path.exists():
        receipt = json.loads(path.with_name(path.name + '.source.json').read_text())
        assert receipt['messageKey'] == entry['messageKey']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256']
        return
    api = BASE + '/' + kind + '/messageUrl?' + urlencode({'messageKey':entry['messageKey']})
    query = subprocess.run(['curl', '-fLsS', '--max-time', '30', api], capture_output=True, text=True, timeout=40)
    assert query.returncode == 0, 'public message URL unavailable; stop without auth bypass'
    url = json.loads(query.stdout)['messageUrl']
    response = subprocess.run(['curl', '-fLsS', '--max-time', '30', '-H', 'Content-Type: text/plain; charset=utf-8', url],
        capture_output=True, timeout=40)
    assert response.returncode == 0, 'public message unavailable; no automatic retry'
    path.parent.mkdir(parents=True, exist_ok=True)
    wire = response.stdout
    body = gzip.decompress(wire) if wire.startswith(b'\x1f\x8b') else wire
    body.decode('utf-8')
    path.write_bytes(body)
    # Temporary signed storage URLs are consumed but never archived or exposed.
    receipt = dict(api=api, messageKey=entry['messageKey'], entry_id=entry['id'],
        received_utc=datetime.now(timezone.utc).isoformat(), bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(), signed_url_archived=False,
        wire_sha256=hashlib.sha256(wire).hexdigest(), wire_bytes=len(wire),
        author_id=entry['createdBy'], createdOn=entry['createdOn'], modifiedOn=entry['modifiedOn'])
    path.with_name(path.name + '.source.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', choices=('2025', '2026'), required=True)
    parser.add_argument('--messages', action='store_true')
    parser.add_argument('--thread', action='append', default=[])
    args = parser.parse_args()
    year = args.year
    forum = json.loads((ROOT / ('brats/sources/forum' + year + '_project.json')).read_text())
    threads = []
    for offset in range(0, 1000, 20):
        url = BASE + '/forum/' + forum['id'] + '/threads?limit=20&offset=' + str(offset) + '&filter=EXCLUDE_DELETED'
        page = cached_api(url, 'brats/sources/' + year + '_threads_' + f'{offset:03d}' + '.json')
        entries = page['results']
        assert len(entries) <= 20
        threads += entries
        if len(entries) < 20:
            break
    else:
        raise AssertionError('bounded pagination exhausted; do not claim complete')
    assert len({x['id'] for x in threads}) == len(threads)
    pattern = re.compile(r'mets?|task\s*1|evaluat|metric|rank|leaderboard|score|validation|pre.?train', re.I)
    relevant = [x for x in threads if pattern.search(x['title'])]
    index = dict(year=year, project=forum['projectId'], forum=forum['id'],
        nondeleted_thread_count=len(threads), pagination_complete=True,
        metadata_only_for_other_threads=True, selected_by_title=[x['id'] for x in relevant],
        threads=threads, received_utc=datetime.now(timezone.utc).isoformat())
    index_path = ROOT / ('brats/FORUM_' + year + '_INDEX.json')
    if not index_path.exists():
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + '\n')
    print('forum', year, 'nondeleted threads', len(threads), 'selected', len(relevant), flush=True)
    if args.messages:
        selected = [x for x in threads if x['id'] in args.thread] if args.thread else relevant
        for entry in selected:
            folder = 'brats/threads/' + year + '/' + entry['id']
            message('thread', entry, folder + '/thread.md')
            replies = []
            for offset in range(0, 1000, 100):
                url = BASE + '/thread/' + entry['id'] + '/replies?limit=100&offset=' + str(offset) + '&filter=EXCLUDE_DELETED'
                page = cached_api(url, folder + '/replies_' + f'{offset:03d}' + '.json')
                replies += page['results']
                if len(page['results']) < 100:
                    break
            else:
                raise AssertionError('reply pagination bound')
            assert len({x['id'] for x in replies}) == len(replies)
            for reply in replies:
                message('reply', reply, folder + '/reply_' + reply['id'] + '.md')
            print(entry['id'], entry['title'], 'replies', len(replies), flush=True)


if __name__ == '__main__':
    main()
