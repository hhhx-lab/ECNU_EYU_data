"""Exact Conda binary-plan gate; rejects model replacements and toolchains."""


def validate_plan(plan, expected):
    assert plan.get('success') is True, 'compatible binary solver did not pass'
    selected = plan['actions']['LINK']
    assert len({item['name'] for item in selected}) == len(selected), 'duplicate solver packages'
    versions = {item['name']: item['version'] for item in selected}
    assert all(versions.get(name) == version for name, version in expected.items()), 'required version drift'
    forbidden = {'torch', 'pytorch', 'torchvision', 'pytorch-gpu', 'pytorch-cuda', 'gcc', 'gxx', 'gfortran', 'clang',
                 'clangxx', 'compilers', 'c-compiler', 'cxx-compiler', 'fortran-compiler', 'binutils'}
    for item in selected:
        name = item['name']
        assert name not in forbidden and not name.startswith(
            ('gcc_', 'gxx_', 'gfortran_', 'clang-', 'binutils_', 'sysroot_')), 'model or compiler package forbidden'
        assert item.get('platform', item.get('subdir')) in ('linux-64', 'noarch'), 'package platform drift'
        assert item.get('channel', '').rstrip('/').split('/')[-1] == 'conda-forge', 'package source drift'
    return selected
