"""Local tests for scheduler and submission boundaries; no server access."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import control_resume as control
import resume_contract as contract


class ResumeGuards(unittest.TestCase):
    def test_completed_job_can_have_empty_squeue(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / 'logs').mkdir()
            plan = dict(source_job_id='3827543', source_job_name='s2k1train175r3', source_root=str(source))
            answers = [dict(rc=0, out='', err=''),
                       dict(rc=0, out='3827543|s2k1train175r3|TIMEOUT|0:0|5-00:00:00|5-00:00:00|2026-10-04|', err=''),
                       dict(rc=0, out='disk', err='')]
            with patch.object(control, 'command', side_effect=answers):
                observed = control.snapshot(plan)
            self.assertEqual(observed['source_state'], 'TIMEOUT')
            self.assertEqual(observed['squeue']['out'], '')

    def test_running_source_never_submits(self):
        for state in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'):
            with patch.object(control, 'command') as command:
                result = control.submit({}, dict(source_state=state))
                self.assertEqual(result['status'], 'waiting_no_submission')
                command.assert_not_called()

    def test_completed_source_is_not_resumed(self):
        with patch.object(control, 'command') as command:
            result = control.submit({}, dict(source_state='COMPLETED', exit_code='0:0'))
            self.assertEqual(result['status'], 'source_completed_do_not_resume')
            command.assert_not_called()

    def test_other_failure_blocks_submission(self):
        with patch.object(control, 'command') as command:
            with self.assertRaises(RuntimeError):
                control.submit({}, dict(source_state='FAILED'))
            command.assert_not_called()

    def test_lr_uses_total175_not_remaining_budget(self):
        self.assertEqual(contract.lr(0), 0.001)
        self.assertEqual(contract.lr(175), 0)
        self.assertAlmostEqual(contract.lr(50), 0.0007387283529495573)
        self.assertGreater(contract.lr(50), contract.lr(100))

    def test_no_pretrained_or_missing_checkpoint_fallback(self):
        text = (ROOT / 'runtime/run_resume.py').read_text()
        self.assertIn("'--c'", text)
        self.assertIn('pretrained_weights_file is None', text)
        self.assertIn('silent scratch fallback', text)
        self.assertIn('amp_restore_verified_on_gpu=True', text)

    def test_package_sha_matches(self):
        package = contract.load_json(ROOT / 'PACKAGE_SHA256.json')
        for relative, digest in package['files'].items():
            self.assertEqual(contract.sha(ROOT / relative), digest, relative)


if __name__ == '__main__':
    unittest.main()
