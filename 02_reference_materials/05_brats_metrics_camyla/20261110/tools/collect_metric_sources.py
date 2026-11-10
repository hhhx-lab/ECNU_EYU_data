"""Freeze small official metric source excerpts for research, without installation."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    plans = [
        ('BraTS/BraTS_evaluation', 'evaluation_commit.json', 'evaluation2026',
         ['README.md', 'LICENSE', 'brats_evaluation/metrics_parser.py',
          'brats_evaluation/configs/config_mets.yaml']),
        ('Astarakee/brats_evaluator', 'evaluator2025_commit.json', 'evaluation2025',
         ['README.md', 'brats-configs/config_mets.yaml']),
    ]
    for repo, receipt, folder, names in plans:
        commit = json.loads((ROOT / 'brats/sources' / receipt).read_text())['sha']
        for name in names:
            relative = 'brats/code/' + folder + '/' + name
            if not (ROOT / relative).exists():
                subprocess.run([sys.executable, '-B', str(ROOT / 'tools/acquire.py'),
                    'https://raw.githubusercontent.com/' + repo + '/' + commit + '/' + name,
                    relative], check=True)
    for path in (ROOT / 'brats/sources').glob('wiki*.json'):
        if path.name.endswith('.source.json'):
            continue
        obj = json.loads(path.read_text())
        target = ROOT / 'brats/wiki' / (path.stem + '.md')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(obj['markdown'])


if __name__ == '__main__':
    main()
