"""Correct only the historical marker path in a separately authorized preflight."""
import ast
import hashlib
import inspect
import json
import os
import shlex

from preflight_environment_r4_official import CHANNEL, CODE, EXPECTED, ROOT, validate_official_plan
from environment_binary_contract import validate_plan
from session import session, access_module

PASS_STATUS = 'r4_official_binary_preflight_pass_no_environment_created'


def remote_code():
    replacements = {
        'operator/environment/FAILED.json': 'operator/ENV_INSTALL_FAILED.json',
        'operator/environment_r4_official_preflight': 'operator/environment_r4_official_preflight_r2',
    }
    counts = dict.fromkeys(replacements, 0)

    class CorrectMarker(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in replacements:
                counts[node.value] += 1
                return ast.copy_location(ast.Constant(replacements[node.value]), node)
            return node

    tree = CorrectMarker().visit(ast.parse(CODE))
    assert all(count == 1 for count in counts.values()), 'unexpected parent code drift'
    ast.fix_missing_locations(tree)
    source = 'from urllib.parse import urlsplit\n' + inspect.getsource(validate_plan)
    source += '\n' + inspect.getsource(validate_official_plan)
    source += '\nexpected=' + repr(EXPECTED) + '\nchannel=' + repr(CHANNEL)
    source += '\n' + ast.unparse(tree)
    source = 'source_sha=' + repr(hashlib.sha256(source.encode()).hexdigest()) + '\n' + source
    compile(source, 'r4_official_r2_preflight', 'exec')
    return source


def main():
    authorization = json.loads((ROOT/'ENVIRONMENT_R4_OFFICIAL_R2_AUTHORIZATION.json').read_text())
    assert authorization['correct_historical_r1_marker_path_only'] is True
    assert authorization['channel'] == CHANNEL
    code = remote_code()
    with session('preflight_environment_r4_official_r2') as access:
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT=' + shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' EXTERNAL_MET_CONDA_BIN=' + shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command += ' ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(code)
        outcome = json.loads(access.ssh(command,timeout=1080).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',outcome)
        print(json.dumps(dict(audit=str(access.audit),status=outcome['status'],remote_utc=outcome['remote_utc'])),flush=True)
        if outcome['status'] != PASS_STATUS:
            raise RuntimeError('Official R4 R2 preflight failed; preserve failure; no installation or automatic retry')


if __name__ == '__main__':main()
