"""Record pinned public text/metadata only; never retrieve model binaries."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BIO = '83c73e0cac0b5a3c77a298547040ea3ea7f99854'
BRAIN = 'ba60f45bed832ee0e5683c678caeaeefe2072f0d'


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    audit = ROOT / 'operator' / f'public_sources_{stamp}'
    audit.mkdir(parents=True, exist_ok=False)
    urls = [
        f'https://api.github.com/repos/thekeramik05/MICCAI-METS_challenge_bi-intel/contents/{name}?ref={BIO}'
        for name in ['environment.yaml', 'requirements.txt', 'Dockerfile', 'docker/install_trainers.py']
    ] + [
        f'https://api.github.com/repos/AIM-KannLab/BrainIAC/contents/{name}?ref={BRAIN}'
        for name in ['README.md', 'LICENSE']
    ] + [
        'https://raw.githubusercontent.com/sylabs/singularity/v4.1.1/internal/pkg/runtime/engine/singularity/container_linux.go',
        'https://raw.githubusercontent.com/sylabs/singularity/v4.1.1/internal/pkg/runtime/engine/singularity/prepare_linux.go',
    ]
    records, raw = [], []
    for url in urls:
        record = dict(url=url, received_utc=datetime.now(timezone.utc).isoformat())
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'ECNU-public-source-audit'})
            with urllib.request.urlopen(request, timeout=30) as response:
                b = response.read(1000000)
                record.update(status=response.status, bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
            text = b.decode()
            if 'api.github.com' in url:
                payload = json.loads(text)
                decoded = base64.b64decode(payload['content'])
                assert len(decoded) == payload['size']
                record.update(blob_sha=payload['sha'], decoded_sha256=hashlib.sha256(decoded).hexdigest(), decoded_bytes=len(decoded))
                text = decoded.decode()
            raw.append(dict(url=url, text=text))
            if '/BrainIAC/' in url:
                record['license_or_author_download_text'] = text
            elif url.endswith('.go'):
                lines = text.splitlines()
                record['network_none_excerpts'] = [dict(line=i+1, text='\n'.join(lines[max(0,i-5):i+12]))
                    for i,line in enumerate(lines) if '"none"' in line and ('network' in '\n'.join(lines[max(0,i-10):i+10]).lower())]
            else:
                record['author_environment_text'] = text
        except Exception as error:
            record.update(status='error', error=str(error), no_automatic_retry=True)
        records.append(record)
    result = dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
        scope='public_text_metadata_only_no_binary_download_no_third_party_execution',source_responses=records,
        status='collected' if all(r['status']==200 for r in records) else 'incomplete',
        environment_installation_authorized_by_user=True,installation_not_performed=True,
        brainiac_browser_file_url='https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AMblt6reQVvlSrORTB3_2lE/BrainIAC.ckpt?rlkey=9w55le6tslwxlfz6c0viylmjb&dl=0',
        brainiac_browser_display_size='345.34 MB',brainiac_exact_bytes_and_sha256='not_yet_verified')
    for name,payload in [('PUBLIC_SOURCE_AUDIT.json',result),('RAW_PUBLIC_TEXT.json',raw)]:
        with (audit/name).open('x',encoding='utf-8') as f:
            json.dump(payload,f,ensure_ascii=False,indent=2);f.write('\n')
    index = {**result, 'source_responses': [
        {key:value for key,value in r.items() if key != 'author_environment_text'} for r in records
    ]}
    with (audit/'PUBLIC_SOURCE_INDEX.json').open('x',encoding='utf-8') as f:
        json.dump(index,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(audit=str(audit),status=result['status'])))


if __name__ == '__main__':
    main()
