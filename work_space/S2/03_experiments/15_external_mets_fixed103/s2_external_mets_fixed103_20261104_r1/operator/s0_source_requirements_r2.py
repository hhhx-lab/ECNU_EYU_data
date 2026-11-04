"""PEP 517 setuptools defaults for the independently proposed staging R2."""
import hashlib
from pathlib import PurePosixPath
import tarfile


def normalize_build_system(build, has_setup_py):
    assert isinstance(build,dict), 'build-system table absent or invalid'
    requires=build.get('requires')
    assert isinstance(requires,list) and all(isinstance(x,str) for x in requires), 'invalid or missing build requirements'
    assert not build.get('backend-path'), 'local build backend path not reviewed'
    backend=build.get('build-backend')
    implicit=[]
    if backend is None:
        assert has_setup_py, 'implicit legacy backend requires setup.py'
        backend='setuptools.build_meta:__legacy__'
        implicit=['setuptools>=40.8.0']
    assert backend in ('setuptools.build_meta','setuptools.build_meta:__legacy__'), 'unreviewed build backend'
    return dict(build_backend=backend,declared_requires=requires,implicit_requires=implicit,
                requires=[*requires,*implicit],backend_was_explicit=build.get('build-backend') is not None)


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
        has_setup=any(x.isfile() and x.name.count('/')==1 and x.name.endswith('/setup.py') for x in members)
        build=normalize_build_system(toml.loads(raw.decode()).get('build-system'),has_setup)
    return dict(**build,configuration_sha256=hashlib.sha256(raw).hexdigest())
