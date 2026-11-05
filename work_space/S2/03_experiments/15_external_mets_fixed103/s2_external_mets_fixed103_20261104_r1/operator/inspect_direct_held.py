"""Read the already-created 4310575 identity without submitting or releasing."""
import json
from session import session, access_module


def main():
    with session('inspect_direct_held_4310575') as access:
        raw=access.ssh('scontrol show job 4310575 -o',timeout=60).strip().splitlines()[-1]
        access_module.exclusive(access.audit/'RESULT.json',dict(job_id='4310575',raw_scontrol=raw,read_only=True))
        print(json.dumps(dict(audit=str(access.audit),raw_scontrol=raw)),flush=True)


if __name__=='__main__':main()
