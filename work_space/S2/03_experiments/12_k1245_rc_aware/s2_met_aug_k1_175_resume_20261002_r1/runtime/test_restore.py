"""CPU restore probe in a fresh process with an isolated nnUNet result path."""

import json
import sys

from resume_contract import restore_test

if __name__ == '__main__':
    import torch
    torch.set_num_threads(2)
    print('CPU_RESTORE_JSON=' + json.dumps(restore_test(sys.argv[1], sys.argv[2]), allow_nan=False))
