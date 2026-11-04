"""Deterministic R2 adaptation of pinned R1 sources; original failures stay immutable."""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PINS={
    'operator/stage_s0_cpu_r1.py':'d726bdf0cc1f4692462cc671875e2ddb45c2927bcf1b93613fd4dc6a3695ca5e',
    'operator/submit_s0_cpu_r1.py':'deadeec0af31fd9b486b1f02a340583255a7bbf85c8431c3bfa5b663435ac528',
    'operator/s0_cpu_submission_contract.py':'9fc8d02f32fcbdbae31e26c12dbe3327ee13a8627d90fa502def95f838270a11',
    'operator/s0_source_requirements.py':'84fab5fbec21063c234c7b454a370c5c58a5534747d5eee2b0ba6035b9ce78c4',
    'runtime/prepare_s0_cpu_r1.py':'274c726c2db8a7dafc02f9e7912dd5b2ac266b3cb886504339635083e996248d',
    'runtime/s0_build_contract.py':'e48acf45fea160ab323377322ea7dd42b3f8abca3792707eba4edf3e64c8633b',
}
PARENT_GUARD=r'''

def parent_staging_guard(root):
    parent=Path(root)/'attempts/s0_cpu_prepare_r1'
    require(sha(parent/'reports/STAGING_FAILED.json')=='be077b174c4975aa0ee7daa26dfd562816d726f0c71be433f19fcf0c54960ee6',
            'parent CPU staging failure SHA drift')
    return {str(p.relative_to(parent)):sha(p) for p in sorted(parent.rglob('*')) if p.is_file()}
'''
VALIDATE_SDIST='''def validate_sdist(path):
    from importlib.metadata import version
    from packaging.requirements import Requirement
    from s0_source_requirements_r2 import read_sdist_build_config
    configuration=read_sdist_build_config(path)
    for text in configuration['requires']:
        item=Requirement(text)
        if item.marker is None or item.marker.evaluate():
            require(not item.url and version(item.name) in item.specifier, 'unstaged build requirement: '+text)
    return configuration
'''


def pinned(path):
    raw=(ROOT/path).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PINS[path], 'original R1 source drift: '+path
    return raw.decode()


def replace_one(source,old,new):
    assert source.count(old)==1, 'R2 adaptation anchor absent or ambiguous'
    return source.replace(old,new)


def replace_function(source,name,replacement):
    matches=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(matches)==1, 'R2 function target absent or ambiguous'
    node=matches[0]
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement+''.join(lines[node.end_lineno:])


def runtime_files():
    contract=replace_one(pinned('runtime/s0_build_contract.py'),"ATTEMPT = 's0_cpu_prepare_r1'","ATTEMPT = 's0_cpu_prepare_r2'")
    contract=replace_one(contract,"BUILD_ENV = 'bio_build_glibc217_r1'","BUILD_ENV = 'bio_build_glibc217_r2'")+PARENT_GUARD
    runtime=replace_function(pinned('runtime/prepare_s0_cpu_r1.py'),'validate_sdist',VALIDATE_SDIST)
    runtime=replace_one(runtime,"root/'attempts/s0_cpu_prepare_r1'","root/'attempts/s0_cpu_prepare_r2'")
    runtime=replace_one(runtime,'from s0_build_contract import META, ARCHIVE_SHA, SOURCES, require, sha, write, prior_guard',
                        'from s0_build_contract import META, ARCHIVE_SHA, SOURCES, require, sha, write, prior_guard, parent_staging_guard')
    runtime=replace_one(runtime,"        require(prior_guard(root) == contract['prior_guard'], 'old environment or failure drift')",
                        "        require(prior_guard(root) == contract['prior_guard'], 'old environment or failure drift')\n"
                        "        require(parent_staging_guard(root)==contract['parent_attempt_files'], 'parent CPU staging files drift')")
    runtime=replace_one(runtime,"        require(prior_guard(root) == contract['prior_guard'], 'old environment/failure changed after preparation')",
                        "        require(prior_guard(root) == contract['prior_guard'], 'old environment/failure changed after preparation')\n"
                        "        require(parent_staging_guard(root)==contract['parent_attempt_files'], 'parent CPU staging files changed after preparation')")
    sources={'s0_build_contract.py':contract,'prepare_s0_cpu_r2.py':runtime,
             's0_source_requirements_r2.py':(ROOT/'operator/s0_source_requirements_r2.py').read_text()}
    for name,source in sources.items():compile(source,name,'exec')
    return sources


def held_helpers():
    helpers=replace_one(pinned('operator/s0_cpu_submission_contract.py'),"attempt.name == 's0_cpu_prepare_r1'","attempt.name == 's0_cpu_prepare_r2'")
    helpers=replace_one(helpers,"Path(build_env).name == 'bio_build_glibc217_r1'","Path(build_env).name == 'bio_build_glibc217_r2'")
    return helpers.replace('s2exts0p1','s2exts0p2').replace('prepare_s0_cpu_r1.py','prepare_s0_cpu_r2.py')
