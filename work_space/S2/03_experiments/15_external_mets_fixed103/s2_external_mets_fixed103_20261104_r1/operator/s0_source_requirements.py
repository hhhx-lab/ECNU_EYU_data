"""Inspect source build metadata before installing the accepted compiler plan."""
from pathlib import PurePosixPath
import tarfile


def read_sdist_build_config(path):
    try:
        import tomllib as toml
    except ImportError:
        try:
            import tomli as toml
        except ImportError:
            from pip._vendor import tomli as toml
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        assert len(members)<10000 and sum(x.size for x in members)<50*1024**2, 'sdist expansion budget'
        for item in members:
            p=PurePosixPath(item.name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in item.name, 'sdist path traversal'
            assert item.isfile() or item.isdir(), 'sdist special file or link'
        configs=[x for x in members if x.name.count('/')==1 and x.name.endswith('/pyproject.toml')]
        assert len(configs)==1, 'sdist build configuration ambiguous'
        raw=archive.extractfile(configs[0]).read()
        build=toml.loads(raw.decode())['build-system']
    assert build['build-backend'] in ('setuptools.build_meta','setuptools.build_meta:__legacy__'), 'unreviewed build backend'
    assert not build.get('backend-path'), 'local build backend path not reviewed'
    import hashlib
    return dict(build_backend=build['build-backend'],requires=build['requires'],
                configuration_sha256=hashlib.sha256(raw).hexdigest())


def validate_solver_build_requirements(configs,selected):
    try:
        from packaging.requirements import Requirement
        from packaging.markers import default_environment
        from packaging.utils import canonicalize_name
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.markers import default_environment
        from pip._vendor.packaging.utils import canonicalize_name
    versions={canonicalize_name(x['name']):x['version'] for x in selected}
    marker=default_environment()
    marker.update(python_version='3.10',python_full_version='3.10.22',sys_platform='linux',os_name='posix',
                  platform_machine='x86_64',platform_system='Linux',implementation_name='cpython')
    checked={}
    for name,config in configs.items():
        checked[name]=[]
        for text in config['requires']:
            item=Requirement(text)
            assert not item.url, 'unreviewed source build URL'
            if item.marker is not None and not item.marker.evaluate(marker):continue
            selected_version=versions.get(canonicalize_name(item.name))
            assert selected_version and selected_version in item.specifier, 'compiler plan misses source build requirement: '+text
            checked[name].append(dict(requirement=text,selected_version=selected_version))
    return checked
