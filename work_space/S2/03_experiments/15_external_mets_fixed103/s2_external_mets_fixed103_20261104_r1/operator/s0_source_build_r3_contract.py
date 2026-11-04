"""Decode Conda's Channel JSON without relaxing the official-source gate."""
from pathlib import Path
from urllib.parse import urlsplit

PREFLIGHT_R3='operator/s0_source_build_preflight_r3'


def validate_official_channel(value, name):
    if isinstance(value, dict):
        assert value.get('scheme')=='https' and value.get('location')=='conda.anaconda.org', 'effective channel origin drift'
        assert value.get('name')==name, 'effective channel name drift'
        assert all(value.get(k) is None for k in ('auth','token','package_filename','platform')), 'unexpected channel credentials or package'
    else:
        assert isinstance(value,str), 'unsupported Conda channel representation'
        parsed=urlsplit(value)
        assert parsed.scheme=='https' and parsed.netloc=='conda.anaconda.org', 'effective channel origin drift'
        assert not parsed.query and not parsed.fragment, 'unexpected channel query'
        allowed={'','/'} if not name else {'','/','/'+name,'/'+name+'/'}
        assert parsed.path in allowed, 'effective channel path drift'
    return 'https://conda.anaconda.org'+('/'+name if name else '')


def effective_config_summary(config):
    def summary(value):
        if isinstance(value,dict):
            return {**{k:value.get(k) for k in ('scheme','location','name','platform','package_filename')},
                    'auth_present':bool(value.get('auth')),'token_present':bool(value.get('token'))}
        if isinstance(value,str):
            parsed=urlsplit(value)
            return dict(scheme=parsed.scheme,hostname=parsed.hostname,path=parsed.path,
                        auth_present=parsed.username is not None or parsed.password is not None,
                        query_present=bool(parsed.query),fragment_present=bool(parsed.fragment))
        return dict(type=type(value).__name__)
    return dict(channel_alias=summary(config.get('channel_alias')),
                conda_forge=summary(config.get('custom_channels',{}).get('conda-forge')),
                pkgs_dirs=config.get('pkgs_dirs'),ssl_verify=config.get('ssl_verify'),
                add_anaconda_token=config.get('add_anaconda_token'))


def validate_effective_config(config,audit):
    alias=validate_official_channel(config['channel_alias'],'')
    channel=validate_official_channel(config['custom_channels']['conda-forge'],'conda-forge')
    assert config['pkgs_dirs']==[str(Path(audit)/'pkgs')], 'isolated cache drift'
    assert config['ssl_verify'] is True, 'TLS verification disabled'
    assert config['add_anaconda_token'] is False, 'public channel auth enabled'
    return dict(channel_alias=alias,conda_forge=channel,pkgs_dirs=config['pkgs_dirs'],
                ssl_verify=True,add_anaconda_token=False)
