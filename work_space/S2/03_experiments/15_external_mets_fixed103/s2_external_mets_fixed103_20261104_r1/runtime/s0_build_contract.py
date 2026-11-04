"""One authorized same-version source-build preparation, separate from inference."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

META = dict(artifact_status='experimental_unvalidated', operator_approved=False,
            formal_gate_status='not_run_not_passed')
CHANNEL = 'https://conda.anaconda.org/conda-forge'
ATTEMPT = 's0_cpu_prepare_r1'
BUILD_ENV = 'bio_build_glibc217_r1'
ARCHIVE_SHA = '731b17a23514bf2460083d4585a42d5ece4137643960c354469790857af52735'
SOURCES = {
    'numexpr': dict(version='2.14.1', filename='numexpr-2.14.1.tar.gz',
        url='https://files.pythonhosted.org/packages/cb/2f/fdba158c9dbe5caca9c3eca3eaffffb251f2fb8674bf8e2d0aed5f38d319/numexpr-2.14.1.tar.gz',
        bytes=119400, sha256='4be00b1086c7b7a5c32e31558122b7b80243fe098579b170967da83f3152b48b'),
    'connected-components-3d': dict(version='4.1.1', filename='connected_components_3d-4.1.1.tar.gz',
        url='https://files.pythonhosted.org/packages/23/66/62eeb4d29188a1161289c58c9129294856ce0f134fb9b49c3d13ae0db1d1/connected_components_3d-4.1.1.tar.gz',
        bytes=85178, sha256='becc200f0b9fd8c16ffe0ddb37c94a963737b96b79705c8ae25dd141f402f990'),
    'ndindex': dict(version='1.10.1', filename='ndindex-1.10.1.tar.gz',
        url='https://files.pythonhosted.org/packages/f5/92/4b9d2f4e0f3eabcfc7b02b48261f6e5ad36a3e2c1bbdcc4e3b7b6c768fa6/ndindex-1.10.1.tar.gz',
        bytes=271395, sha256='0f6113c1f031248f8818cbee1aa92aa3c9472b7701debcce9fddebcd2f610f11'),
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(dict(META, **payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    path.chmod(0o440)


def prior_guard(root):
    root = Path(root)
    result = {}
    for name in ('bio_intel_py310', 'bio_intel_py310_r2', 'bio_intel_py310_r3'):
        prefix = root / 'envs' / name
        require(prefix.is_dir(), 'old environment absent: ' + name)
        result[name] = {str(p.relative_to(prefix)): sha(p)
                        for p in sorted((prefix / 'conda-meta').glob('*')) if p.is_file()}
        result[name]['bin/python'] = sha(prefix / 'bin/python')
    markers = ('operator/ENV_INSTALL_FAILED.json', 'operator/environment_r2/FAILED.json',
               'operator/environment_r3/FAILED.json', 'operator/environment_r4_preflight/FAILED.json',
               'operator/environment_r4_official_preflight_r2/FAILED.json',
               'operator/asset_materialization/FAILED.json')
    result['failures'] = {name: sha(root / name) for name in markers}
    return result


def validate_build_plan(plan):
    require(plan.get('success') is True, 'build environment solver failed')
    selected = plan['actions']['LINK']
    require(len({x['name'] for x in selected}) == len(selected), 'duplicate build packages')
    versions = {x['name']: x['version'] for x in selected}
    for name, value in {'python': '3.10.22', 'numpy': '2.2.6', 'sysroot_linux-64': '2.17'}.items():
        require(versions.get(name) == value, 'build version drift: ' + name)
    for name in ('gcc_linux-64', 'gxx_linux-64', 'binutils_linux-64', 'setuptools', 'wheel', 'cython', 'pip', 'tomli', 'packaging'):
        require(name in versions, 'build tool missing: ' + name)
    require(versions['gcc_linux-64'].split('.')[0] == versions['gxx_linux-64'].split('.')[0] == '13',
            'compiler generation drift')
    for item in [*selected, *plan['actions'].get('FETCH', [])]:
        require(item['name'] not in ('torch', 'pytorch', 'torchvision', 'pytorch-cuda'), 'model dependency in build env')
        require(item.get('platform', item.get('subdir')) in ('linux-64', 'noarch'), 'build platform drift')
        urls = [item.get(k) for k in ('url', 'base_url', 'channel')
                if isinstance(item.get(k), str) and '://' in item[k]]
        require(urls, 'absolute build package origin missing')
        for url in urls:
            parsed = urlsplit(url)
            require(parsed.scheme == 'https' and parsed.netloc == 'conda.anaconda.org'
                    and (parsed.path == '/conda-forge' or parsed.path.startswith('/conda-forge/'))
                    and not parsed.query and not parsed.fragment, 'build package origin drift')
    return selected
