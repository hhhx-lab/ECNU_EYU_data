"""Offline single-file VPN tests: no real env, login, sudo, or SSH."""
import ast
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT.parent / '服务器登陆' / 'ecnu-vpn-split.sh'
ARCHIVE = ROOT / '01_project_documents/04_reference_code_notes/atrust_access/history/20261018_two_file/ecnu_atrust.py'
MARKER = "3<<'ECNU_ATRUST_PY'\n"

def embedded():
    text = SCRIPT.read_text()
    source, end = text.split(MARKER, 1)[1].rsplit('\nECNU_ATRUST_PY', 1)
    if end.strip():
        raise ValueError('unexpected shell tail')
    return source + '\n'

a = types.ModuleType('ecnu_atrust_embedded_test')
exec(compile(embedded(), str(SCRIPT), 'exec'), a.__dict__)
ONLINE = {'status':'online','data':{'addr':a.URL,'tunnelStatus':'tunnelAvailable','loginTime':'example'}}
BASE = {'default':{'interface':'en0','gateway':'192.0.2.1'},
        'public':{n:{'interface':'en0','gateway':'192.0.2.1'} for n in a.PUBLIC},
        'dns':'unchanged','proxy':'unchanged','ipv6':''}
STATE = {'interface':'utun7','login_time':'example','ranges':['59.78.189.132/32'],'baseline':BASE,'owns_session':False}

class AdapterTests(unittest.TestCase):
    def test_embedded_body_unchanged(self):
        self.assertEqual(embedded().strip(), ARCHIVE.read_text().strip())
        ast.parse(embedded())

    def test_shell_syntax(self):
        self.assertEqual(subprocess.run(['/bin/bash','-n',str(SCRIPT)],capture_output=True).returncode,0)

    def test_loader_preserves_stdin_arguments_and_fd(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            interpreter=root/'probe'
            interpreter.write_text('#!'+sys.executable+'\nimport os,sys\n'
                'source=os.read(3,200).decode()\n'
                'print("SOURCE="+str(source.startswith(\'"""ECNU aTrust\')))\n'
                'print("ARGS="+repr(sys.argv[-2:]))\n'
                'print("STDIN="+sys.stdin.readline().strip())\n')
            interpreter.chmod(0o700)
            envfile=root/'.env'
            envfile.write_text("ECNU_VPN_PYTHON='"+str(interpreter)+"'\n")
            envfile.chmod(0o600)
            result=subprocess.run(['/bin/bash',str(SCRIPT),'ssh-hpc','59.78.189.133'],
                env={'PATH':os.environ['PATH'],'HOME':str(root),'ECNU_SERVER_ENV_FILE':str(envfile)},
                input='terminal-input\n',text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('SOURCE=True',result.stdout)
            self.assertIn("ARGS=['ssh-hpc', '59.78.189.133']",result.stdout)
            self.assertIn('STDIN=terminal-input',result.stdout)

    def test_help_and_invalid_exit_with_fixture_env(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); envfile=root/'.env'
            envfile.write_text("ECNU_VPN_PYTHON='"+sys.executable+"'\nECNU_VPN_STATE_DIR='"+td+"/state'\n")
            envfile.chmod(0o600)
            env={'PATH':os.environ['PATH'],'HOME':td,'ECNU_SERVER_ENV_FILE':str(envfile)}
            result=subprocess.run(['/bin/bash',str(SCRIPT),'help'],env=env,text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('connect',result.stdout)
            result=subprocess.run(['/bin/bash',str(SCRIPT),'unknown'],env=env,text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,1)

    def test_abbreviated_routes(self):
        for text,expected in (('1.1/16','1.1.0.0/16'),('11','11.0.0.0/8'),('128.0/2','128.0.0.0/2'),('59.78.189.132','59.78.189.132/32')):
            self.assertEqual(str(a.net4(text)),expected)

    def test_login_not_process_presence(self):
        self.assertTrue(a.online(ONLINE))
        for key,value in (('status','logout'),('status','login')):
            v=copy.deepcopy(ONLINE);v[key]=value;self.assertFalse(a.online(v))
        v=copy.deepcopy(ONLINE);v['data']['tunnelStatus']='tunnelUnavailable';self.assertFalse(a.online(v))
        v=copy.deepcopy(ONLINE);v['data']['addr']='https://other.example';self.assertFalse(a.online(v))

    def test_hpc_scope(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(a.scopes(),['59.78.189.132/32','59.78.189.133/32','59.78.189.195/32'])

    def test_custom_rejects_default(self):
        with patch.dict(os.environ,{'ECNU_VPN_SCOPE':'custom','ECNU_IPRANGE':'0.0.0.0/0'},clear=True):
            with self.assertRaises(RuntimeError):a.scopes()

    def test_ecnu_keeps_hpc(self):
        with patch.dict(os.environ,{'ECNU_VPN_SCOPE':'ecnu'},clear=True):
            self.assertIn('219.228.56.0/21',a.scopes())
            self.assertIn('59.78.189.132/32',a.scopes())

    def test_ipv6_detection(self):
        self.assertTrue(a.global_v6('default fe80::%utun7 UGc utun7','utun7'))
        self.assertTrue(a.global_v6('2000::/3 utun7 UGc utun7','utun7'))
        self.assertFalse(a.global_v6('fdff:5341:4e47::/48 utun7 UGc utun7','utun7'))
        self.assertFalse(a.global_v6('default fe80::%utun4 UGc utun4','utun7'))

    def test_check_detects_public_capture(self):
        current=copy.deepcopy(BASE);current['public']['1.1.1.1']['interface']='utun7'
        with patch.object(a,'tunnel',return_value='utun7'),patch.object(a,'snapshot',return_value=current):
            with self.assertRaisesRegex(RuntimeError,'public changed'):a.check(ONLINE,STATE)

    def test_check_detects_route_leak(self):
        leaked=[{'interface':'utun7','gateway':'utun7','network':'128.0.0.0/1'}]
        with patch.object(a,'tunnel',return_value='utun7'),patch.object(a,'snapshot',return_value=BASE),patch.object(a,'routes',return_value=leaked):
            with self.assertRaisesRegex(RuntimeError,'out-of-scope'):a.check(ONLINE,STATE)

    def test_check_rejects_session_change(self):
        v=copy.deepcopy(ONLINE);v['data']['loginTime']='changed'
        with patch.object(a,'tunnel',return_value='utun7'):
            with self.assertRaisesRegex(RuntimeError,'session/interface'):a.check(v,STATE)

    def test_stop_preserves_unowned(self):
        with patch.object(a,'agent',return_value=ONLINE) as agent,patch.object(a,'load_session',return_value=STATE):
            a.stop();self.assertEqual(agent.call_count,1)

    def test_stop_preserves_new_user_login(self):
        state={**STATE,'owns_session':True,'login_time':'older'}
        with patch.object(a,'agent',return_value=ONLINE) as agent,patch.object(a,'load_session',return_value=state):
            a.stop();self.assertEqual(agent.call_count,1)

    def test_cleanup_preserves_new_login(self):
        current=copy.deepcopy(ONLINE);current['data']['loginTime']='new'
        with patch.object(a,'agent',return_value=current) as agent,patch.object(a,'receipt'):
            a.cleanup_session(ONLINE);self.assertEqual(agent.call_count,1)

    def test_session_identity_required(self):
        missing=copy.deepcopy(ONLINE);missing['data'].pop('loginTime')
        self.assertFalse(a.same_session(missing,missing))
        with patch.object(a,'tunnel',return_value='utun7'),patch.object(a,'scopes',return_value=[]),patch.object(a,'routes',return_value=[]),patch.object(a,'edit') as edit:
            with self.assertRaisesRegex(RuntimeError,'identity missing'):a.split(missing,BASE,True)
            edit.assert_not_called()

    def test_new_global_ipv6_rejected_before_edit(self):
        result=types.SimpleNamespace(stdout='default fe80::%utun7 UGc utun7')
        with patch.object(a,'tunnel',return_value='utun7'),patch.object(a,'scopes',return_value=[]),patch.object(a,'routes',return_value=[]),patch.object(a,'run',return_value=result),patch.object(a,'edit') as edit:
            with self.assertRaisesRegex(RuntimeError,'IPv6'):a.split(ONLINE,BASE,True)
            edit.assert_not_called()

    def test_rollback_only_verified_interface(self):
        original=[{'interface':'utun7','gateway':'utun7','network':'1.0.0.0/8'},
                  {'interface':'utun4','gateway':'utun4','network':'100.64.0.0/10'}]
        edits=[]
        def edit(action,network,ifc):
            edits.append((action,network,ifc))
            return types.SimpleNamespace(returncode=int(action=='add' and network=='59.78.189.132/32'))
        with patch.object(a,'tunnel',return_value='utun7'),patch.object(a,'scopes',return_value=['59.78.189.132/32']),patch.object(a,'routes',return_value=original),patch.object(a,'sudo_prime'),patch.object(a,'save'),patch.object(a,'receipt'),patch.object(a,'edit',side_effect=edit),patch.object(a,'agent',return_value=ONLINE),patch.object(a,'run',return_value=types.SimpleNamespace(stdout='')):
            with self.assertRaises(RuntimeError):a.split(ONLINE,copy.deepcopy(BASE),False)
        self.assertIn(('add','1.0.0.0/8','utun7'),edits)
        self.assertTrue(all(row[2]=='utun7' for row in edits))

    def test_rollback_delete_failure_not_success(self):
        def edit(action,network,ifc):return types.SimpleNamespace(returncode=int(action=='delete'))
        with patch.object(a,'tunnel',return_value='utun7'),patch.object(a,'scopes',return_value=['59.78.189.132/32']),patch.object(a,'routes',return_value=[]),patch.object(a,'sudo_prime'),patch.object(a,'save'),patch.object(a,'receipt') as receipt,patch.object(a,'edit',side_effect=edit),patch.object(a,'agent',return_value=ONLINE),patch.object(a,'time'),patch.object(a,'check',side_effect=RuntimeError('verification failed')),patch.object(a,'run',return_value=types.SimpleNamespace(stdout='')):
            with self.assertRaises(RuntimeError):a.split(ONLINE,BASE,True)
            event,data=receipt.call_args.args
            self.assertEqual(event,'split_failed')
            self.assertEqual(data['status'],'restore_incomplete')

if __name__ == '__main__':
    unittest.main()
