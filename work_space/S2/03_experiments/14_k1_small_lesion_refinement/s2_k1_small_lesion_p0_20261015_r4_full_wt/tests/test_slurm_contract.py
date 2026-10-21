import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from slurm_contract import validate


class Held(unittest.TestCase):
    def raw(self):
        return 'JobName=x Partition=cpu_192G NumCPUs=4 NumNodes=1-1 MinMemoryNode=65536M TimeLimit=1-00:00:00 JobState=PENDING Reason=JobHeldUser Command=/new/slurm/cpu_supplement.slurm StdOut=/new/logs/a.out StdErr=/new/logs/a.err ReqTRES=cpu=4,mem=64G,node=1 Shared=OK'
    def test_equivalent_units_and_node_range(self):self.assertEqual(validate(self.raw(),'/new','x')['status'],'pass')
    def test_wrong_memory(self):self.assertEqual(validate(self.raw().replace('65536M','32G'),'/new','x')['status'],'failed')
    def test_wrong_path(self):self.assertEqual(validate(self.raw().replace('/new/logs/a.err','/old/logs/a.err'),'/new','x')['status'],'failed')
    def test_gpu_rejected(self):self.assertEqual(validate(self.raw().replace('mem=64G,node=1','mem=64G,node=1,gres/gpu=1'),'/new','x')['status'],'failed')
    def test_not_held(self):self.assertEqual(validate(self.raw().replace('JobHeldUser','Priority'),'/new','x')['status'],'failed')
