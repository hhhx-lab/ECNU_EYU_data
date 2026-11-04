"""Command-local source isolation and lossless locks for the approved R2."""
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

PREFLIGHT_R2 = 'operator/s0_source_build_preflight_r2'


def isolated_conda_env(audit, inherited):
    env = {k:v for k,v in inherited.items() if not k.startswith('CONDA_OVERRIDE_')
           and k not in ('CONDA_CHANNELS', 'CONDA_CHANNEL_ALIAS', 'CONDA_CUSTOM_CHANNELS',
                         'CONDA_CUSTOM_MULTICHANNELS', 'CONDA_MIGRATED_CUSTOM_CHANNELS',
                         'CONDA_MIGRATED_CHANNEL_ALIASES', 'CONDA_SUBDIR', 'CONDA_SSL_VERIFY')}
    env.update(CONDARC=str(Path(audit)/'condarc.yaml'), CONDA_PKGS_DIRS=str(Path(audit)/'pkgs'),
               CONDA_CHANNEL_ALIAS='https://conda.anaconda.org', CONDA_ADD_ANACONDA_TOKEN='false',
               CONDA_REMOTE_MAX_RETRIES='0', PYTHONDONTWRITEBYTECODE='1', LC_ALL='C',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    return env


def official_config():
    return ('channels:\n  - https://conda.anaconda.org/conda-forge\n'
            'channel_alias: https://conda.anaconda.org\n'
            'custom_channels:\n  conda-forge: https://conda.anaconda.org\n'
            'ssl_verify: true\nadd_anaconda_token: false\nremote_max_retries: 0\n')


def validate_effective_config(config, audit):
    assert config['channel_alias'] == 'https://conda.anaconda.org', 'effective alias drift'
    assert config['custom_channels'].get('conda-forge') == 'https://conda.anaconda.org', 'effective channel drift'
    assert config['pkgs_dirs'] == [str(Path(audit)/'pkgs')], 'isolated cache drift'
    assert config['ssl_verify'] is True, 'TLS verification disabled'
    assert config['add_anaconda_token'] is False, 'public channel auth enabled'
    return dict(channel_alias=config['channel_alias'], conda_forge=config['custom_channels']['conda-forge'],
                pkgs_dirs=config['pkgs_dirs'], ssl_verify=True, add_anaconda_token=False)


def explicit_lock_records(plan):
    fetch = {(x['name'],x['version'],x.get('build',x.get('build_string'))):x
             for x in plan['actions'].get('FETCH', [])}
    records = []
    for item in plan['actions']['LINK']:
        key = item['name'], item['version'], item.get('build',item.get('build_string'))
        other = fetch.get(key, {})
        url = item.get('url') or other.get('url')
        if not url:
            fn = item.get('fn') or other.get('fn')
            assert fn, 'explicit package filename absent; cannot guess archive type'
            url = item['base_url'].rstrip('/')+'/'+item.get('platform',item.get('subdir'))+'/'+fn
        parsed = urlsplit(url)
        assert (parsed.scheme,parsed.netloc) == ('https','conda.anaconda.org'), 'explicit package origin drift'
        assert parsed.path.startswith('/conda-forge/') and not parsed.query and not parsed.fragment
        fn = Path(parsed.path).name
        assert fn.endswith(('.conda','.tar.bz2')), 'explicit package archive type drift'
        checksum = item.get('sha256') or other.get('sha256') or item.get('md5') or other.get('md5')
        assert checksum and re.fullmatch(r'(?:[0-9a-f]{32}|[0-9a-f]{64})', checksum), 'explicit checksum absent'
        records.append(dict(name=key[0],version=key[1],build=key[2],url=url,checksum=checksum,filename=fn))
    assert len({x['url'] for x in records}) == len(records), 'duplicate explicit URLs'
    return records


def render_explicit_lock(records):
    return '@EXPLICIT\n'+''.join(x['url']+'#'+x['checksum']+'\n' for x in records)
