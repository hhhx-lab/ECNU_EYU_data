"""Collect the public manuscript samples and dataset descriptions at one revision."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    tree = json.loads((ROOT / 'camyla/sources/frozen_tree.json').read_text())
    commit = json.loads((ROOT / 'camyla/sources/commit.json').read_text())
    assert not tree['truncated'] and tree['sha'] == commit['commit']['tree']['sha']
    revision = commit['sha']
    targets = ['README.md', 'README_zh.md', 'LICENSE', 'assets/camyla.pdf']
    targets += [e['path'] for e in tree['tree'] if e['type'] == 'blob' and e['path'].startswith(('assets/paper_pdf/', 'ideas/'))]
    entries = {e['path']:e for e in tree['tree'] if e['type'] == 'blob'}
    assert len([x for x in targets if x.startswith('assets/paper_pdf/')]) == 10
    assert sum(entries[x]['size'] for x in targets) < 64 * 1024**2
    manifest = []
    for name in targets:
        if name == 'assets/camyla.pdf':
            relative = 'camyla/Camyla_system_paper.pdf'
        elif name.startswith('assets/paper_pdf/'):
            relative = 'camyla/manuscripts/' + Path(name).name
        else:
            relative = 'camyla/sources/' + name
        destination = ROOT / relative
        url = 'https://raw.githubusercontent.com/yifangao112/Camyla/' + revision + '/' + name
        if not destination.exists():
            subprocess.run([sys.executable, '-B', str(ROOT / 'tools/acquire.py'), url, relative], check=True)
        receipt = json.loads(destination.with_name(destination.name + '.source.json').read_text())
        data = destination.read_bytes()
        assert receipt['success'] and receipt['url'] == url and hashlib.sha256(data).hexdigest() == receipt['sha256']
        assert len(data) == entries[name]['size']
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == entries[name]['sha']
        manifest.append(dict(source=name, destination=relative, bytes=len(data), sha256=receipt['sha256'], git_blob_sha1=entries[name]['sha']))
    output = ROOT / 'camyla/PUBLIC_ASSET_MANIFEST.json'
    with output.open('x') as handle:
        json.dump(dict(repository='https://github.com/yifangao112/Camyla', commit=revision,
            public_manuscripts=10, reported_generated_manuscripts=40, full_40_manuscripts_acquired=False,
            files=manifest, total_bytes=sum(r['bytes'] for r in manifest)), handle, indent=2)
        handle.write('\n')
    print('Camyla public asset SHA and Git blob checks pass', len(manifest))


if __name__ == '__main__':
    main()
