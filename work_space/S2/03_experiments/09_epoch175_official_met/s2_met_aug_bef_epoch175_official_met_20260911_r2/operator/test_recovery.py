import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recovery', ROOT / 'runtime/recovery.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ContractTests(unittest.TestCase):
    def test_nested_region_ontology_includes_four(self):
        self.assertEqual(r.ontology({'labels':{'background':0,'regions':[1,[2,3],4]}}), {0,1,2,3,4})
        for value in (True, '4', 4.0, None):
            with self.assertRaises(RuntimeError):
                r.label_values(value)

    def test_ontology_missing_four_is_rejected(self):
        with self.assertRaises(RuntimeError):
            r.ontology({'labels':{'all':[0,1,2,3]}})

    def test_reservation_is_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(r,'ROOT',Path(d)), patch.object(r,'sha',return_value='sha'), patch.object(r,'read',return_value={'status':'pass','plan_sha256':'sha'}), patch.object(r.subprocess,'check_output',return_value='12345\n') as submit, patch.object(r.subprocess,'run') as release:
                r.launch('sha')
                self.assertIn('--hold',submit.call_args.args[0])
                self.assertEqual(release.call_args.args[0],['scontrol','release','12345'])
                self.assertTrue((Path(d)/'evidence/CPU_LAUNCH.json').exists())
                with self.assertRaises(FileExistsError):
                    r.launch('sha')
                self.assertEqual(submit.call_count,1)

    def test_audit_failure_keeps_job_held(self):
        with tempfile.TemporaryDirectory() as d:
            def audit(name,*args,**kwargs):
                if name == 'CPU_LAUNCH.json':
                    raise OSError('test audit failure')
            with patch.object(r,'ROOT',Path(d)), patch.object(r,'sha',return_value='sha'), patch.object(r,'read',return_value={'status':'pass','plan_sha256':'sha'}), patch.object(r,'audit',side_effect=audit), patch.object(r.subprocess,'check_output',return_value='12345\n'), patch.object(r.subprocess,'run') as release:
                with self.assertRaises(OSError):
                    r.launch('sha')
                release.assert_not_called()


if __name__ == '__main__':
    unittest.main()
