"""One authorized low-concurrency continuation, with durable verified part ledgers."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from download_assets import IMAGE, DIGEST, BIO_URL, BRAIN_URL, META, ROOT, utc, write
from download_segmented import sha

CHUNK = 64 * 1024**2
PREFIX_BYTES = 256901120
PREFIX_SHA = '67f4856457356ff42210b19571bb85440af2d505e53656c0fc5cddb767864d32'
PREFIX_LAYER = 'ab224c3cf399b42d4c9dfa9bdaa36275478cd38d2738fc03bdd45f8231e4dcf0'


class TransportError(RuntimeError):
    def __init__(self, code, http, stderr):
        # Report diagnostic classes only: curl stderr can contain signed URLs.
        categories = [name for name, pattern in {
            'timeout': r'timed out|Timeout', 'tls': r'SSL|certificate|TLS',
            'connection_reset': r'reset|Recv failure', 'connection_failed': r'Failed to connect',
            'dns': r'resolve host', 'http_failure': r'returned error',
            'partial_response': r'partial|outstanding read|bytes missing',
        }.items() if re.search(pattern, stderr, re.I)]
        self.diagnostic = dict(curl_exit=code, http_status=http, categories=categories)
        super().__init__(json.dumps(self.diagnostic))


def curl_value(value):
    assert '\n' not in value and '\r' not in value
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def range_header(path, start, end, total):
    text = path.read_text(errors='replace')
    blocks = re.split(r'\r?\n\r?\n', text.strip())
    final = next((b for b in reversed(blocks) if b.startswith('HTTP/')), '')
    values = re.findall(r'^content-range:\s*(.*?)\s*$', final, re.I | re.M)
    assert values == [f'bytes {start}-{end}/{total}'], 'final Content-Range mismatch'


def fetch(url, target, proxy, headers=None, bounds=None, total=None, max_time=600):
    assert not target.exists(), 'exclusive download destination required'
    header_path = target.with_suffix(target.suffix + '.headers')
    config = ['url = ' + curl_value(url)]
    for key, value in (headers or {}).items():
        config.append('header = ' + curl_value(key + ': ' + value))
    if bounds:
        config.append('range = ' + curl_value(f'{bounds[0]}-{bounds[1]}'))
    command = ['curl', '--config', '-', '--silent', '--show-error', '--fail', '--location',
               '--retry', '0', '--connect-timeout', '20', '--max-time', str(max_time),
               '--proto', '=https', '--proto-redir', '=https', '--dump-header', str(header_path),
               '--output', str(target), '--write-out', '%{http_code}']
    if proxy:
        command += ['--proxy', proxy]
    try:
        result = subprocess.run(command, input='\n'.join(config) + '\n', text=True,
                                capture_output=True, timeout=max_time + 30)
        http = result.stdout.strip()
        if result.returncode:
            raise TransportError(result.returncode, http, result.stderr)
        assert http == ('206' if bounds else '200'), 'unexpected final HTTP status: ' + http
        if bounds:
            range_header(header_path, *bounds, total)
            assert target.stat().st_size == bounds[1] - bounds[0] + 1, 'range length mismatch'
        else:
            assert target.stat().st_size == total, 'asset length mismatch'
        return dict(bytes=target.stat().st_size, sha256=sha(target), http_status=http)
    finally:
        # Headers may contain ephemeral signed redirects and never enter audit/Git.
        if header_path.exists():
            header_path.unlink()


class RegistryAuth:
    def __init__(self, proxy):
        self.lock = threading.Lock()
        self.token = None
        self.refresh_at = 0
        self.refreshes = 0
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler(
            {'http': proxy, 'https': proxy} if proxy else {}))

    def headers(self):
        with self.lock:
            if time.monotonic() >= self.refresh_at:
                url = 'https://auth.docker.io/token?service=registry.docker.io&scope=repository:' + IMAGE + ':pull'
                with self.opener.open(url, timeout=40) as response:
                    payload = response.read(1024**2)
                    assert len(payload) < 1024**2, 'token metadata size cap'
                data = json.loads(payload)
                self.token = data['token']
                self.refresh_at = time.monotonic() + min(180, max(20, data.get('expires_in', 300) - 60))
                self.refreshes += 1
            return {'Authorization': 'Bearer ' + self.token}


def main():
    temp = Path(os.environ['EXTERNAL_MET_LOCAL_TRANSFER_ROOT'])
    assert temp.parent == Path.home() / 'Library/Caches/ECNUExternalMets' and temp.name == ROOT.name
    assert json.loads((temp / 'OWNERSHIP.json').read_text())['batch'] == ROOT.name
    assert not temp.is_symlink()
    audit = ROOT / 'operator' / ('download_bounded_resume_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    audit.mkdir(exist_ok=False)
    stage = temp / 'bounded_resume'
    stage.mkdir(mode=0o700, exist_ok=False)
    files = []
    current = None
    try:
        free = shutil.disk_usage(temp).free
        assert free > 30 * 1024**3, 'fresh 30GiB Mac temporary capacity gate'
        write(audit / 'STARTED.json', dict(META, utc=utc(), free_bytes_before=free,
            authorization='one_controlled_resume_approved_by_user', concurrency=3,
            segment_bytes=CHUNK, curl_retry=0, old_attempts_preserved=True, jobs_created=False))
        manifest_path = temp / 'docker/manifest.json'
        config_path = temp / 'docker/config.json'
        assert sha(manifest_path) == DIGEST
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest['layers']) == 11 and sum(x['size'] for x in manifest['layers']) == 4446698630
        assert sha(config_path) == manifest['config']['digest'].split(':')[1]
        for path in (manifest_path, config_path):
            files.append(dict(path=str(path.relative_to(temp)), logical_path=str(path.relative_to(temp)),
                              bytes=path.stat().st_size, sha256=sha(path)))
        proxy = os.environ.get('EXTERNAL_MET_LOCAL_HTTP_PROXY', '')
        auth = RegistryAuth(proxy)
        items = [(f'https://registry-1.docker.io/v2/{IMAGE}/blobs/{x["digest"]}',
                  'docker/layers/' + x['digest'].split(':')[1] + '.tar.gz',
                  x['size'], x['digest'].split(':')[1], True) for x in manifest['layers']]
        items += [(BIO_URL, 'weights.zip', 2472704349,
                   'd308d3dd5dff4ec8fa899d64f056d8a18651b4d4e56f6618c66a2d8998e334f3', False),
                  (BRAIN_URL, 'BrainIAC.ckpt', 362119240, None, False)]
        for url, relative, total, digest, registry in items:
            current = relative
            old = temp / relative
            if old.is_file() and not old.is_symlink() and old.stat().st_size == total and digest is not None:
                assert sha(old) == digest, 'cached complete asset SHA mismatch'
                files.append(dict(path=relative, logical_path=relative, bytes=total, sha256=digest,
                                  reused_verified_complete=True))
                print(json.dumps(dict(event='reuse_verified_asset', file=relative, bytes=total)), flush=True)
                continue
            asset_audit = audit / (Path(relative).name + '.parts')
            asset_audit.mkdir()
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            parts_root = stage / 'parts' / Path(relative).name
            parts_root.mkdir(parents=True, exist_ok=False)
            prefix = 0
            parts = {}
            if digest == PREFIX_LAYER and old.is_file():
                assert old.stat().st_size == PREFIX_BYTES and sha(old) == PREFIX_SHA, 'known continuous prefix drift'
                prefix = PREFIX_BYTES
                write(asset_audit / 'PREFIX.json', dict(path=str(old.relative_to(temp)), bytes=prefix,
                    sha256=PREFIX_SHA, full_digest_pending=True, old_sparse_attempt_not_reused=True))
            print(json.dumps(dict(event='bounded_asset_started', file=relative, bytes=prefix,
                                  expected_bytes=total, utc=utc())), flush=True)
            started = time.monotonic()
            if total <= CHUNK or relative == 'BrainIAC.ckpt':
                part = parts_root / 'whole.part'
                record = fetch(url, part, proxy, auth.headers() if registry else {}, total=total, max_time=1800)
                write(asset_audit / 'WHOLE.json', dict(record, path=str(part.relative_to(temp)), utc=utc()))
                parts[0] = part
            else:
                ranges = iter((a, min(a + CHUNK, total) - 1) for a in range(prefix, total, CHUNK))
                done = prefix

                def download_part(a, b):
                    part = parts_root / f'{a:012d}-{b:012d}.part'
                    record = fetch(url, part, proxy, auth.headers() if registry else {}, (a, b), total)
                    record.update(start=a, end=b, total_bytes=total, path=str(part.relative_to(temp)), utc=utc())
                    write(asset_audit / f'{a:012d}-{b:012d}.json', record)
                    return a, part, record['bytes']

                with ThreadPoolExecutor(max_workers=3) as pool:
                    pending = set()

                    def submit_next():
                        bounds = next(ranges, None)
                        if bounds is not None:
                            pending.add(pool.submit(download_part, *bounds))

                    for _ in range(3):
                        submit_next()
                    try:
                        while pending:
                            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                            for future in completed:
                                pending.remove(future)
                                a, part, count = future.result()
                                parts[a] = part
                                done += count
                            print(json.dumps(dict(event='verified_segment_progress', file=relative,
                                bytes=done, expected_bytes=total, utc=utc())), flush=True)
                            assert time.monotonic() - started < 7200, 'bounded per-asset two-hour cap exceeded'
                            for _ in completed:
                                submit_next()
                    except BaseException:
                        for future in pending:
                            future.cancel()
                        raise
            with destination.open('xb') as output:
                if prefix:
                    with old.open('rb') as source:
                        shutil.copyfileobj(source, output, 1024**2)
                for offset, part in sorted(parts.items()):
                    assert output.tell() == offset, 'segment contiguity mismatch'
                    with part.open('rb') as source:
                        shutil.copyfileobj(source, output, 1024**2)
                output.flush()
                os.fsync(output.fileno())
            actual = sha(destination)
            assert destination.stat().st_size == total and (digest is None or actual == digest), 'final full asset SHA mismatch'
            destination.chmod(0o400)
            record = dict(path=str(destination.relative_to(temp)), logical_path=relative, bytes=total, sha256=actual,
                          prefix_bytes=prefix, verified_segment_count=len(parts))
            write(asset_audit / 'FULL_SHA_VERIFIED.json', dict(record, utc=utc()))
            files.append(record)
            print(json.dumps(dict(event='bounded_asset_SHA_verified', file=relative, bytes=total,
                                  sha256=actual, utc=utc())), flush=True)
        bio = next(temp / f['path'] for f in files if f['logical_path'] == 'weights.zip')
        with zipfile.ZipFile(bio) as archive:
            entries = []
            for item in archive.infolist():
                path = PurePosixPath(item.filename)
                assert not path.is_absolute() and '..' not in path.parts and '\\' not in item.filename
                assert ((item.external_attr >> 16) & 0o170000) != 0o120000
                entries.append(dict(path=item.filename, bytes=item.file_size, compressed_bytes=item.compress_size))
            assert len(entries) == len({x['path'] for x in entries})
            assert sum(x['bytes'] for x in entries) < 40 * 1024**3
            write(audit / 'BIO_ZIP_INVENTORY.json', dict(META, entries=entries,
                  expanded_bytes=sum(x['bytes'] for x in entries)))
        assert len(files) == 15
        result = dict(META, status='local_assets_verified_not_server_transferred', temporary_root=str(temp),
            files=files, utc=utc(), image_digest='sha256:' + DIGEST, registry_token_refresh_count=auth.refreshes,
            old_partial_preserved=True, no_model_loaded=True, jobs_created=False,
            cleanup_status='requires_remote_complete_SHA_acceptance')
        write(audit / 'LOCAL_ASSET_MANIFEST.json', result)
        write(stage / 'TRANSFER_MANIFEST.json', result)
        print(json.dumps(dict(event='all_local_assets_verified', audit=str(audit), files=len(files))), flush=True)
    except BaseException as error:
        diagnostic = getattr(error, 'diagnostic', None)
        if isinstance(error, urllib.error.URLError):
            diagnostic = dict(reason_type=type(error.reason).__name__,
                              errno=getattr(error.reason, 'errno', None))
        write(audit / 'DOWNLOAD_FAILED.json', dict(META, status='failed', file=current, error_type=type(error).__name__,
            diagnostic=diagnostic, message=str(error) if isinstance(error, AssertionError) else 'transport stopped; private URL suppressed',
            utc=utc(), verified_files=files, segment_ledgers_preserved=True,
            no_automatic_retry=True, old_partial_preserved=True, jobs_created=False))
        raise RuntimeError('Authorized bounded resume stopped; preserved audit; no further automatic attempt') from None


if __name__ == '__main__':
    main()
