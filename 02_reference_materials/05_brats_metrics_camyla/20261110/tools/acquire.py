"""Archive explicitly requested public references with URL/UTC/byte identities."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('destination')
    args = parser.parse_args()
    path = (ROOT / args.destination).resolve()
    assert path.is_relative_to(ROOT) and not path.exists()
    assert args.url.startswith('https://')
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.partial')
    receipt = path.with_name(path.name + '.source.json')
    assert not part.exists() and not receipt.exists()
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(['curl', '--location', '--silent', '--show-error', '--max-time', '60',
        '--max-filesize', str(64 * 1024**2), '--output', str(part), '--write-out', '%{json}', args.url],
        capture_output=True, text=True, timeout=75)
    try:
        raw_network = json.loads(result.stdout)
        network = {key: raw_network[key] for key in (
            'http_code', 'content_type', 'url_effective', 'size_download',
            'time_total', 'ssl_verify_result') if key in raw_network}
    except json.JSONDecodeError:
        network = {'parse_failed': True}
    success = result.returncode == 0 and network.get('http_code') == 200
    if success:
        if path.suffix == '.pdf':
            assert part.read_bytes().startswith(b'%PDF-'), 'not a PDF'
        part.rename(path)
    else:
        if part.exists():
            part.rename(path.with_name(path.name + '.failed_response'))
    record = dict(url=args.url, destination=args.destination, started_utc=started,
        received_utc=datetime.now(timezone.utc).isoformat(), success=success,
        returncode=result.returncode, network=network, stderr=result.stderr,
        bytes=path.stat().st_size if success else None,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest() if success else None,
        untrusted_external_reference=True, project_scientific_validation_pass=False)
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:record[k] for k in ('url', 'destination', 'success', 'bytes', 'sha256')}, ensure_ascii=False))
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
