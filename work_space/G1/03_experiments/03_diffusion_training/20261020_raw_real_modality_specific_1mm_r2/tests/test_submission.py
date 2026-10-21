"""Submission parsing checks do not access the network or create jobs."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('raw_remote_submit', ROOT / 'operator/remote_submit.py')
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)


class SubmissionTests(unittest.TestCase):
    def test_no_dependency(self):
        self.assertEqual(submission.dependency_ids('(null)'), set())

    def test_single_afterok(self):
        for text in ('afterok:123', 'afterok:123(unfulfilled)', 'afterok:123(fulfilled)'):
            self.assertEqual(submission.dependency_ids(text), {'123'})

    def test_expanded_dependency(self):
        self.assertEqual(submission.dependency_ids('afterok:123(unfulfilled),afterok:124(unfulfilled)'), {'123', '124'})

    def test_reject_other_type_and_extra_job(self):
        for text in ('afterany:123', 'afterok:123:124', 'afterok:123,afterany:124', ''):
            with self.assertRaises(AssertionError):
                submission.dependency_ids(text)

    def test_job_fields(self):
        f = submission.fields('JobId=123 NumCPUs=8 MinMemoryNode=64G Dependency=afterok:12(unfulfilled) TRES=cpu=8,mem=64G,gres/gpu=1')
        self.assertEqual(f['MinMemoryNode'], '64G')
        self.assertEqual(f['Dependency'], 'afterok:12(unfulfilled)')
        self.assertEqual(submission.fields(f['TRES'].replace(',', ' '))['gres/gpu'], '1')

    def test_exact_one_gpu(self):
        self.assertTrue(submission.one_gpu({'TRES': 'cpu=8,gres/gpu=1', 'TresPerNode': 'gpu:1,gpu:1'}))
        self.assertTrue(submission.one_gpu({'TresPerNode': 'gpu:1,gpu:1'}))
        self.assertFalse(submission.one_gpu({'TRES': 'gres/gpu=2', 'TresPerNode': 'gpu:1'}))
        self.assertFalse(submission.one_gpu({'TresPerNode': 'gres:gpu:10'}))


if __name__ == '__main__':
    unittest.main()
