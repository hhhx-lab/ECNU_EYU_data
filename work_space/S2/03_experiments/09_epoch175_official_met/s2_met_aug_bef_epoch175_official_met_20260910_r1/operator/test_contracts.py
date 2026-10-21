"""Local stdlib tests; never contact Slurm or load a checkpoint."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('workflow', HERE / 'runtime/workflow.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class Contracts(unittest.TestCase):
    def test_launch_is_once_and_release_follows_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'evidence').mkdir()
            names = dict(B='s2b175m1', E_SNFH='s2e175m1', F='s2f175m1', PAIRED='s2x175p1')
            commands = []
            def run(cmd, **kwargs):
                commands.append(cmd)
                if cmd[0] == 'sbatch':
                    self.assertIn('--hold', cmd)
                    return SimpleNamespace(stdout=str(9000 + len(commands)) + '\n')
                self.assertEqual(cmd[:2], ['scontrol', 'release'])
                for method, job in zip((*w.METHODS, 'PAIRED'), ('9001', '9002', '9003', '9004')):
                    tag = method + '_EVALUATION' if method != 'PAIRED' else 'PAIRED_COMPARISON'
                    self.assertEqual(json.loads((root / f'evidence/{tag}_LAUNCH.json').read_text())['job_id'], job)
                self.assertTrue((root / 'evidence/LAUNCH_SUMMARY.json').exists())
                return SimpleNamespace(returncode=0)
            with patch.object(w, 'ROOT', root), patch.object(w, 'require_preflight', return_value={'job_names':names}), patch.object(w, 'sha', return_value='frozen'), patch.object(w.subprocess, 'run', side_effect=run):
                w.launch('frozen')
                self.assertEqual(len(commands), 5)
                self.assertIn('--dependency=afterok:9001:9002:9003', commands[3])
                self.assertEqual(commands[4], ['scontrol', 'release', '9001,9002,9003,9004'])
                with self.assertRaisesRegex(RuntimeError, 'reservation already exists'):
                    w.launch('frozen')
                self.assertEqual(len(commands), 5)

    def test_evidence_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'evidence.json'
            w.write(file, {'status':'failed'})
            with self.assertRaises(FileExistsError):
                w.write(file, {'status':'pass'})
            self.assertEqual(json.loads(file.read_text())['status'], 'failed')

    def test_only_two_paired_cli_changes(self):
        old = HERE.parent / 's2_met_aug_btof_selected103_official_met_20260907_r4/runtime/compare_paired_met.py'
        text = (HERE / 'runtime/compare_paired_met.py').read_text()
        text = text.replace('    parser.add_argument("--expected-methods", type=int, choices=(2, 3), required=True)\n', '')
        text = text.replace('require(len(inputs) == args.expected_methods, "comparison method count drift")', 'require(len(inputs) == 5, "comparison requires exactly five methods")')
        self.assertEqual(text, old.read_text())


if __name__ == '__main__':
    unittest.main()
