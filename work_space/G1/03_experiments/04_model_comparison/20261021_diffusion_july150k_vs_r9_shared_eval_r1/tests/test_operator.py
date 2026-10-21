import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
import common
spec=importlib.util.spec_from_file_location('pair_remote_launch_test',ROOT/'operator/remote_launch.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)


class Operator(unittest.TestCase):
    def test_dependency_forms(self):
        self.assertEqual(launch.dependency_ids('(null)'),set())
        self.assertEqual(launch.dependency_ids('afterok:123:124'),{'123','124'})
        self.assertEqual(launch.dependency_ids('afterok:123(unsatisfied),afterok:124(satisfied)'),{'123','124'})
        for bad in ('afterany:123','afterok:123,afterok:123','afterok:123?afterok:124','afterok:'):
            with self.assertRaises(AssertionError):launch.dependency_ids(bad)

    def test_expected_package_not_self_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'source.py').write_text('pass\n')
            (root/'PACKAGE_MANIFEST.json').write_text(json.dumps({'files':{'source.py':common.sha(root/'source.py')}}))
            h=common.sha(root/'PACKAGE_MANIFEST.json')
            with patch.object(common,'ROOT',root),patch.object(common,'cfg',return_value={'remote_root':str(root)}):
                with patch.dict(common.os.environ,{'G1_PAIR_PACKAGE_SHA256':h}):self.assertEqual(common.own_package(),h)
                with patch.dict(common.os.environ,{'G1_PAIR_PACKAGE_SHA256':'0'*64}):
                    with self.assertRaises(AssertionError):common.own_package()

    def test_user_selected_r9_identity(self):
        c=json.loads((ROOT/'RUN.json').read_text())
        expected={
            't1c':(120000,'c66c9610634a6e1a06ffdf5df201f627036000c053100a9b8f02299922f892f4'),
            't1n':(40000,'ec418ac7dbdcb72cf6202af30f10d6b2ca79249294842e85e875f7b895a5d6f9'),
            't2w':(200000,'27ea3175e26b7aeaaba28f30a553cd71fa2eab0c7e79989355da46ef39aa28a5'),
            't2f':(200000,'c642ea88fe73e479a2ddf5d663fe220c5bcada31e03cfea9d981ff24d0acc93b')}
        self.assertEqual({m:(v['step'],v['sha256']) for m,v in c['models']['r9'].items()},expected)
        self.assertEqual(c['selection_sha256'],'541d0fd245ba66841f35574aed3ad227efa9363c580211a1c8733f528d808902')


if __name__=='__main__':unittest.main()
