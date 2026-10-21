"""Complete only an unexecuted partition update after our proven successful hold."""
import ast
import datetime
import json
import shlex
from pathlib import Path

from common import REMOTE, SOURCE_CODE, m, transport


def main():
    parent = Path(__file__).parent / 'prepare_cpu192_correction_20261015T063601Z'
    local_failure = json.loads((parent / 'CORRECTION_RECEIPT.json').read_text())
    assert local_failure['status'] == 'stopped' and local_failure['stage'] == 'hold'
    assert len(local_failure['mutation_responses']) == 1
    assert local_failure['mutation_responses'][0]['argv'] == ['scontrol', 'hold', '4069915']
    assert local_failure['mutation_responses'][0]['returncode'] == 0
    tree = ast.parse(Path(__file__).with_name('change_prepare_partition.py').read_text())
    code = next(n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'code' for t in n.targets))
    access = transport('complete_unexecuted_cpu192_update')
    auth = json.loads((parent / 'AUTHORIZATION.json').read_text())
    auth.update(completion_scope='complete only unexecuted update/release after our successful hold',
                parent_failure_sha256=m.sha(parent / 'CORRECTION_RECEIPT.json'))
    m.exclusive(access.audit / 'AUTHORIZATION.json', auth)
    original = " assert cpu['Reason']=='BadConstraints','No longer the diagnosed blocked job; stop without change'"
    replacement = r'''
 previous=root/'evidence/prepare_cpu192_correction_20261015T063601Z'
 failure=json.loads((previous/'CORRECTION_FAILED.json').read_text())
 assert failure['stage']=='hold' and len(failure['mutation_responses'])==1
 assert failure['mutation_responses'][0]['argv']==['scontrol','hold','4069915']
 assert failure['mutation_responses'][0]['returncode']==0
 assert not (previous/'UPDATE_RESPONSE.json').exists() and not (previous/'RELEASE_RESPONSE.json').exists()
 prior_hold=json.loads((previous/'HOLD_RESPONSE.json').read_text())
 assert prior_hold['returncode']==0 and prior_hold['argv']==['scontrol','hold','4069915']
 assert cpu['Reason'] in ('JobHeldAdmin','JobHeldUser') and cpu['Priority']=='0'
 write('PRESERVED_PARENT_HOLD_AUDIT.json',{'failure':failure,'hold_response':prior_hold,
       'failure_sha256':sha(previous/'CORRECTION_FAILED.json'),
       'original_static_sha256':sha(previous/'STATIC_RESOURCE_PREFLIGHT.json')})
'''
    assert code.count(original) == 1
    code = code.replace(original, replacement)
    hold_block = " stage='hold';r=command(['scontrol','hold','4069915']);commands.append(r);write('HOLD_RESPONSE.json',r)\n assert r['returncode']==0,'Hold rejected; do not update'\n"
    assert code.count(hold_block) == 1
    code = code.replace(hold_block, " stage='verify_owned_hold'\n")
    for key, value in {'__ROOT__': str(REMOTE), '__SOURCE__': str(SOURCE_CODE), '__AUDIT_NAME__': access.audit.name}.items():
        code = code.replace(key, repr(value))
    code = code.replace('__AUTH__', repr(auth))
    compile(code, '<complete_unexecuted_partition_update>', 'exec')
    try:
        access.start_vpn()
        result = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -u -c '+shlex.quote(code), timeout=420).strip().splitlines()[-1])
        result['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'CORRECTION_RECEIPT.json', result)
        print(json.dumps({'audit': str(access.audit), **{k:v for k,v in result.items() if k not in ('evidence','traceback','dependent_checks','eligible_nodes')}}, ensure_ascii=False))
        if result['status'] == 'stopped':
            raise RuntimeError('Completion stopped; see immutable audit, do not repeat')
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
