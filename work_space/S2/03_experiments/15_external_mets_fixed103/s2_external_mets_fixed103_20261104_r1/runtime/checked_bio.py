"""Call the author's full seven-fold entry with fixed config and real modalities."""
import json
import os
from pathlib import Path
import random
import runpy
import sys

from gpu_isolation import check_inside, require


def main():
    check_inside()
    expected = json.loads(os.environ['FULL_BIO_CONFIG'])
    source = Path(expected['source'])
    sys.path.insert(0, str(source))
    import config
    require(str(config.MODEL_A_DIR) == expected['model_A_dir'], 'Bio model A path drift')
    require(str(config.MODEL_B_DIR) == expected['model_B_dir'], 'Bio model B path drift')
    require(list(config.MODEL_A_FOLDS) == [0, 1, 2, 3, 4] and list(config.MODEL_B_FOLDS) == [1, 3], 'seven-fold drift')
    require(config.CHECKPOINT_NAME == 'checkpoint_final.pth', 'Bio checkpoint selection drift')
    require(config.DISABLE_TTA and config.ENABLE_POSTPROCESSING and config.TILE_STEP_SIZE == 0.5, 'Bio inference settings drift')
    require(config.ENSEMBLE_W_MODEL_A == {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.15}, 'author ensemble weight drift')
    require(config.SMALL_SIZE_CUTOFF == 50 and config.CONFIDENCE_THRESHOLD == {1: 0.6, 2: 0.6, 3: 0.6, 4: 0.5}, 'author postprocessing drift')
    require(list(config.MODALITY_SUFFIXES) == ['t1c', 't1n', 't2f', 't2w'], 'Bio channel order drift')
    folders = sorted(Path('/input').iterdir())
    require(len(folders) == 103 and all(p.is_dir() for p in folders), 'Bio fixed103 input set drift')
    for folder in folders:
        require(len(list(folder.iterdir())) == 4, 'Bio unexpected input file')
        for suffix in config.MODALITY_SUFFIXES:
            matches = list(folder.glob('*-' + suffix + '.nii.gz'))
            require(len(matches) == 1 and matches[0].is_file(), 'missing modality fallback prohibited')
    import numpy as np
    import torch
    random.seed(20261104)
    np.random.seed(20261104)
    torch.manual_seed(20261104)
    torch.cuda.manual_seed_all(20261104)
    sys.argv = [str(source / 'predict.py'), '--input', '/input', '--output', '/output']
    runpy.run_path(str(source / 'predict.py'), run_name='__main__')


if __name__ == '__main__':
    main()
