"""Single official metadata inspection after mirror preflight failure; no install."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request
from preflight_environment_r4 import EXPECTED
from download_assets import META, write

ROOT = Path(__file__).resolve().parents[1]


def request(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'ECNU-approved-R4-binary-metadata'})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            raw = response.read(16*1024**2+1)
            assert len(raw) <= 16*1024**2, 'official metadata exceeds bound'
            return {'status': response.status, 'url': url, 'bytes': len(raw),
                    'sha256': hashlib.sha256(raw).hexdigest(), 'payload': json.loads(raw)}
    except urllib.error.HTTPError as error:
        return {'status': error.code, 'url': url}
    except (OSError, ValueError) as error:
        return {'status': 'error', 'url': url, 'error_type': type(error).__name__, 'automatic_retry': False}


def main():
    audit = ROOT/'operator'/('official_conda_r4_metadata_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    audit.mkdir(exist_ok=False)
    result = dict(META, status='official_metadata_not_server_solver_pass', packages={},
                  installation=False, environment_created=False, models_loaded=False, jobs_created=False)
    for name in ('numexpr', 'connected-components-3d', 'ndindex'):
        version = EXPECTED[name]
        release = request('https://api.anaconda.org/release/conda-forge/'+name+'/'+version)
        payload = release.pop('payload', {})
        distributions = payload.get('distributions', [])
        release.update(requested_version=version, distributions=[
            {key: item.get(key) for key in ('basename', 'size', 'md5', 'sha256', 'attrs')}
            for item in distributions if item.get('basename', '').startswith(('linux-64/', 'noarch/'))])
        if release['status'] == 404:
            inventory = request('https://api.anaconda.org/package/conda-forge/'+name)
            public = inventory.pop('payload', {})
            inventory.update(published_versions=public.get('versions', []), latest_version=public.get('latest_version'))
            release['package_inventory'] = inventory
        result['packages'][name] = release
    result['local_utc'] = datetime.now(timezone.utc).isoformat()
    write(audit/'RESULT.json', result)
    print(json.dumps(dict(audit=str(audit),packages={
        name: dict(status=item['status'],matching_distributions=len(item['distributions']),
                   latest_version=item.get('package_inventory', {}).get('latest_version'))
        for name,item in result['packages'].items()})),flush=True)


if __name__ == '__main__':
    main()
