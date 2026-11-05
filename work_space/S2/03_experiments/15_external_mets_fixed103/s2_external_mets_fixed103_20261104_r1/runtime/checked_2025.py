"""Execute the unchanged 2025 entry point only inside allocated isolation."""
import hashlib
import json
import os
from pathlib import Path
import random

from gpu_isolation import check_inside, require


def adapt_source(source):
    before = 'tmp_input_path = Path("/input_tmp")'
    require(source.count(before) == 1, '2025 scratch path adapter anchor drift')
    return source.replace(before, 'tmp_input_path = Path("/tmp/input_tmp")')


def main():
    check_inside()
    path = Path('/workspace/main.py')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == os.environ['FULL_2025_MAIN_SHA256'], '2025 entry identity drift')
    import numpy as np
    import torch
    random.seed(20261104)
    np.random.seed(20261104)
    torch.manual_seed(20261104)
    torch.cuda.manual_seed_all(20261104)
    adapted = adapt_source(path.read_text())
    print(json.dumps(dict(status='2025_nonroot_scratch_path_adapter', original_entry_sha256=os.environ['FULL_2025_MAIN_SHA256'],
          adapted_entry_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
          changed_path_only=dict(original='/input_tmp', temporary='/tmp/input_tmp'), model_logic_changed=False)), flush=True)
    exec(compile(adapted, str(path), 'exec'), dict(__name__='__main__', __file__=str(path), __package__=None))


if __name__ == '__main__':
    main()
