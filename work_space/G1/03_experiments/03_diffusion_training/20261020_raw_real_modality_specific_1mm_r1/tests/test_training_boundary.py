"""Focused CPU boundaries; no real cohort, CUDA training, or quality acceptance.

Execute the new train.py functions from its AST so identity.py can be developed
independently. Identity validation itself belongs to the identity test suite.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
import random
import signal
import sys
import tempfile
import time
import traceback
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from monai.transforms import Compose


ROOT = Path(__file__).resolve().parents[1]
R9_TRAIN = ROOT / 'plan/BASELINE_TRAIN_REFERENCE.py'
TRAIN_PATH = ROOT / 'runtime/train.py'
UPSTREAM = ROOT / 'runtime/upstream_source'
BOUNDARY_FUNCTIONS = (
    'request_stop', 'random_nodes', 'capture_random', 'restore_random',
    'new_cursor', 'next_indices', 'save_checkpoint', 'restore_checkpoint',
)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_file('raw_boundary_common', ROOT / 'runtime/common.py')
checkpoints = load_file('raw_boundary_checkpoints', UPSTREAM / 'src/utils/checkpoints.py')
noise = load_file('raw_boundary_noise', UPSTREAM / 'src/utils/gaussian_noise_tumour.py')


def training_functions():
    tree = ast.parse(TRAIN_PATH.read_text(), filename=str(TRAIN_PATH))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    module = ModuleType('raw_training_boundary')
    module.__dict__.update(
        argparse=argparse, copy=copy, json=json, os=os, random=random,
        signal=signal, sys=sys, time=time, traceback=traceback, Path=Path,
        SimpleNamespace=SimpleNamespace, np=np, torch=torch, ROOT=ROOT,
        META=common.META, now=common.now, sha=common.sha, write=common.write,
        verify_package=common.verify_package, atomic_save=checkpoints.atomic_save,
        STOP=False,
    )
    # Use production bodies without importing an unfinished identity or global src package.
    selected = ast.Module(body=[functions[name] for name in (*BOUNDARY_FUNCTIONS, 'main')],
                          type_ignores=[])
    exec(compile(selected, str(TRAIN_PATH), 'exec'), module.__dict__)
    return module


def transform(seed):
    result = Compose([
        Compose([noise.GaussianNoiseTumour('scan_t1c')]),
        noise.GaussianNoiseTumour('scan_t1c'),
    ])
    result.set_random_state(seed=seed)
    return result


class TrainingBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.training = training_functions()
        self.original_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.addCleanup(random.setstate, self.original_rng[0])
        self.addCleanup(np.random.set_state, self.original_rng[1])
        self.addCleanup(torch.set_rng_state, self.original_rng[2])
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cfg = json.loads((ROOT / 'config/RUN.json').read_text())
        self.args = SimpleNamespace(**self.cfg['training'], modality='t1c')
        self.identity = {'modality': 't1c', 'csv_sha256': hashlib.sha256(b't1c train rows').hexdigest()}
        self.schedule = SimpleNamespace(schedule='edm', n_steps=self.args.n_steps,
                                        sigma_data=self.args.sigma_data, timesteps=torch.arange(3))

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def objects(self, seed=123):
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.optim_lr,
                                      betas=(self.args.adam_beta1, .999))
        scaler = torch.amp.GradScaler('cpu')
        return model, optimizer, scaler, transform(seed)

    @staticmethod
    def update(model, optimizer, scaler):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.tensor([[1., 2., 3.]])).square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    def draws(self, augmentation):
        return [random.random(), np.random.rand(), torch.rand(1).item(),
                *[node.R.rand() for node in self.training.random_nodes(augmentation).values()]]

    def saved_fixture(self):
        model, optimizer, scaler, augmentation = self.objects()
        self.update(model, optimizer, scaler)
        cursor = self.training.new_cursor(19, self.cfg['seed'])
        cursor['position'] = 16
        path = self.root / 'checkpoint.pt'
        digest = self.training.save_checkpoint(path, model, optimizer, scaler, 1, cursor,
                                               augmentation, self.identity, self.args, self.schedule)
        return path, digest, cursor, (model, optimizer, scaler, augmentation)

    def test_independent_random_states_roundtrip(self):
        augmentation = transform(20261013)
        nodes = self.training.random_nodes(augmentation)
        self.assertEqual(set(nodes), {'transform', 'transform/0', 'transform/0/0', 'transform/1'})
        self.assertEqual(len({id(node.R) for node in nodes.values()}), len(nodes))
        state = self.training.capture_random(augmentation)
        self.assertEqual(state['cuda'], [])
        expected = self.draws(augmentation)
        self.draws(augmentation)
        fresh = transform(77)
        self.training.restore_random(state, fresh)
        self.assertEqual(self.draws(fresh), expected)
        incomplete = Compose([noise.GaussianNoiseTumour('scan_t1c')])
        with self.assertRaisesRegex(AssertionError, 'random-state graph differs'):
            self.training.restore_random(state, incomplete)

    def test_sampler_resume_tail_and_epoch_are_independent_of_global_rng(self):
        cursor = self.training.new_cursor(19, self.cfg['seed'])
        first, end = self.training.next_indices(cursor, 16)
        self.assertEqual(end, 16)
        self.assertEqual(cursor['position'], 0)
        cursor['position'] = end
        fresh = copy.deepcopy(cursor)
        tail, end = self.training.next_indices(cursor, 16)
        self.assertEqual(len(tail), 3)
        self.assertEqual(sorted(first + tail), list(range(19)))
        self.assertEqual((tail, end), self.training.next_indices(fresh, 16))
        cursor['position'] = fresh['position'] = end
        expected = self.training.next_indices(cursor, 16)
        np.random.seed(918)
        np.random.rand(100)
        self.assertEqual(self.training.next_indices(fresh, 16), expected)
        self.assertEqual(fresh['epoch'], 2)
        self.assert_nested_equal(fresh, cursor)

    def test_invalid_sampler_inputs_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'Empty training dataset'):
            self.training.new_cursor(0, 1)
        for size, position in ((0, 0), (16, -1), (16, 20)):
            with self.subTest(batch_size=size, position=position):
                cursor = self.training.new_cursor(19, 1)
                cursor['position'] = position
                with self.assertRaises(AssertionError):
                    self.training.next_indices(cursor, size)

    def test_full_checkpoint_roundtrip_into_fresh_objects(self):
        path, digest, cursor, original = self.saved_fixture()
        model, optimizer, scaler, augmentation = original
        saved = torch.load(path, map_location='cpu', weights_only=False)
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(len(digest), 64)
        self.assertEqual(set(p.name for p in self.root.iterdir()), {'checkpoint.pt'})
        self.assertEqual(saved['checkpoint_schema'], 'g1_complete_boundary_v1')
        self.assertEqual(saved['step_semantics'], 'completed_optimizer_updates_v2')
        self.assertIs(saved['safe_boundary'], True)
        self.assertEqual(saved['schedule_config'], {'schedule': 'edm', 'n_steps': self.args.n_steps,
                                                   'sigma_data': self.args.sigma_data})
        self.assertEqual(saved['training_config'], vars(self.args))
        for key, value in common.META.items():
            self.assertEqual(saved[key], value)
        expected = self.draws(augmentation)
        restored_model, restored_optimizer, restored_scaler, fresh = self.objects(seed=999)
        step, restored_cursor = self.training.restore_checkpoint(
            path, restored_model, restored_optimizer, restored_scaler, fresh, self.identity, self.args)
        self.assertEqual(step, 1)
        self.assert_nested_equal(restored_cursor, cursor)
        self.assertEqual(self.draws(fresh), expected)
        self.assert_nested_equal(restored_model.state_dict(), model.state_dict())
        self.assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())
        self.assert_nested_equal(restored_scaler.state_dict(), scaler.state_dict())
        self.assertTrue(restored_optimizer.state)
        self.assertTrue(restored_scaler.is_enabled())
        self.update(model, optimizer, scaler)
        self.update(restored_model, restored_optimizer, restored_scaler)
        self.assert_nested_equal(restored_model.state_dict(), model.state_dict())
        self.assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())
        self.assert_nested_equal(restored_scaler.state_dict(), scaler.state_dict())
        self.assertEqual(self.training.next_indices(restored_cursor, 16),
                         self.training.next_indices(cursor, 16))

    def test_sha_identity_and_training_config_drift_rejected_before_loading(self):
        path, _, _, _ = self.saved_fixture()
        drift_cases = [
            ({**self.identity, 'csv_sha256': hashlib.sha256(b'changed rows').hexdigest()}, self.args),
            ({**self.identity, 'modality': 't2w'}, self.args),
        ]
        for key, value in (('optim_lr', 1e-4), ('adam_beta1', .5), ('crop_size', 32),
                           ('num_steps', 100000), ('modality', 't2w')):
            drift_cases.append((self.identity, SimpleNamespace(**{**vars(self.args), key: value})))
        for identity, args in drift_cases:
            with self.subTest(identity=identity, args=vars(args)):
                model, optimizer, scaler, augmentation = self.objects()
                before = copy.deepcopy(model.state_dict())
                rng = self.training.capture_random(augmentation)
                with self.assertRaisesRegex(AssertionError, 'Resume identity drift'):
                    self.training.restore_checkpoint(path, model, optimizer, scaler,
                                                     augmentation, identity, args)
                self.assert_nested_equal(model.state_dict(), before)
                self.assertFalse(optimizer.state)
                self.assert_nested_equal(self.training.capture_random(augmentation), rng)

    def test_unsafe_or_invalid_saved_boundary_rejected(self):
        path, _, _, _ = self.saved_fixture()
        saved = torch.load(path, map_location='cpu', weights_only=False)
        for key, value in (('safe_boundary', False), ('checkpoint_schema', 'weights_only'),
                           ('step_semantics', 'attempted_updates'),
                           ('sampler', {**saved['sampler'], 'position': 20}),
                           ('sampler', {**saved['sampler'], 'n_rows': 0}),
                           ('sampler', {**saved['sampler'], 'permutation': [0] * 19})):
            with self.subTest(key=key, value=value):
                invalid = self.root / 'invalid.pt'
                checkpoints.atomic_save({**saved, key: value}, invalid)
                model, optimizer, scaler, augmentation = self.objects()
                with self.assertRaises(AssertionError):
                    self.training.restore_checkpoint(invalid, model, optimizer, scaler,
                                                     augmentation, self.identity, self.args)

    def test_frozen_package_sha_drift_rejected(self):
        frozen = self.root / 'runtime/train.py'
        frozen.parent.mkdir()
        frozen.write_bytes(b'frozen runtime bytes\n')
        manifest = self.root / 'plan/PACKAGE_MANIFEST.json'
        manifest.parent.mkdir()
        manifest.write_text(json.dumps({'files': {'runtime/train.py': common.sha(frozen)}}))
        with patch.object(common, 'ROOT', self.root):
            self.assertEqual(common.verify_package(), common.sha(manifest))
            frozen.write_bytes(b'changed runtime bytes\n')
            with self.assertRaisesRegex(RuntimeError, 'Frozen package drift: runtime/train.py'):
                common.verify_package()

    def test_safe_term_after_successful_update_never_marks_complete(self):
        training = self.training
        (self.root / 'config').mkdir()
        (self.root / 'config/RUN.json').write_text(json.dumps(self.cfg))
        model, optimizer, scaler, augmentation = self.objects()
        batch = {'scan_t1c_crop_pad': torch.zeros((1, 1, 64, 64, 64)),
                 'label_crop_pad': torch.zeros((1, 4, 64, 64, 64))}
        dataset = [batch] * 19

        class Dataset(list):
            transform = augmentation

        loader = SimpleNamespace(dataset=Dataset(dataset), collate_fn=lambda rows: rows[0])

        def one_update(args, step, batches, net, optim, amp, schedule, output):
            self.assertEqual(len(batches), 1)
            self.update(net, optim, amp)
            training.request_stop(signal.SIGTERM, None)
            return step + 1, net, optim, [0.25], None, None

        identity_reader = Mock(return_value=self.identity)
        loader_factory = Mock(return_value=loader)
        upstream = SimpleNamespace(
            _diffusion_utils=SimpleNamespace(seed_everything=Mock(),
                                            make_noise_schedule=Mock(return_value=self.schedule)),
            get_nets=Mock(return_value=(model, optimizer)), train=one_update,
        )
        output = self.root / 'training/t1c'
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(training, ROOT=self.root, upstream=upstream,
                                              get_loader=loader_factory, data_identity=identity_reader,
                                              verify_package=Mock(return_value='a' * 64), create=True))
            stack.enter_context(patch.object(sys, 'argv', ['train.py', '--modality', 't1c']))
            stack.enter_context(patch.dict(os.environ, {'WORLD_SIZE': '1'}))
            stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
            stack.enter_context(patch.object(torch.cuda, 'device_count', return_value=1))
            stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='A100 CPU test double'))
            stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
            stack.enter_context(patch.object(torch.cuda, 'get_rng_state_all', return_value=[]))
            stack.enter_context(patch.object(torch.amp, 'GradScaler', return_value=scaler))
            # The runtime's real-network size assertion is outside this CPU boundary test.
            stack.enter_context(patch.object(training, 'sum', return_value=4475999, create=True))
            signals = stack.enter_context(patch.object(signal, 'signal'))
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                training.main()
        self.assertEqual(stopped.exception.code, 99)
        self.assertFalse((output / 'COMPLETE.ok').exists())
        self.assertFalse((output / 'FAILED.json').exists())
        marker = json.loads((output / 'SAFE_TERM.json').read_text())
        self.assertIs(marker['training_complete'], False)
        self.assertIs(marker['safe_boundary'], True)
        self.assertEqual(marker['global_step'], 1)
        checkpoint = Path(marker['checkpoint'])
        self.assertEqual(checkpoint.name, 'safe_term_1.pt')
        self.assertEqual(marker['sha256'], common.sha(checkpoint))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.assertEqual(saved['global_step'], 1)
        self.assertEqual(saved['sampler']['position'], 16)
        self.assertTrue(saved['optimizer']['state'])
        self.assertEqual(saved['data_identity'], self.identity)
        progress = [json.loads(line) for line in (output / 'progress.jsonl').read_text().splitlines()]
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]['global_step'], 1)
        self.assertEqual(progress[0]['batch_position'], 16)
        identity_reader.assert_called_once_with('t1c')
        self.assertEqual(loader_factory.call_args.args[0].csv_path,
                         str(self.root / 'dataset/lesions_t1c.csv'))
        self.assertFalse(loader_factory.call_args.args[0].use_all_rows)
        signals.assert_any_call(signal.SIGTERM, training.request_stop)
        signals.assert_any_call(signal.SIGUSR1, training.request_stop)


class TrainingSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(TRAIN_PATH.read_text())
        cls.functions = {node.name: node for node in cls.tree.body if isinstance(node, ast.FunctionDef)}

    def test_r9_boundary_functions_and_successful_update_loop_mechanically_reused(self):
        original = ast.parse(R9_TRAIN.read_text())
        original_functions = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
        for name in BOUNDARY_FUNCTIONS:
            with self.subTest(function=name):
                self.assertEqual(ast.dump(self.functions[name]), ast.dump(original_functions[name]))
        loops = [node for node in ast.walk(self.functions['main']) if isinstance(node, ast.While)]
        old_loops = [node for node in ast.walk(original_functions['main']) if isinstance(node, ast.While)]
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(old_loops), 1)
        self.assertEqual(ast.dump(loops[0]), ast.dump(old_loops[0]))

    def test_new_identity_import_replaces_local_vae_identity(self):
        imports = [node for node in self.tree.body if isinstance(node, ast.ImportFrom)
                   and node.module == 'identity']
        self.assertEqual(len(imports), 1)
        self.assertIn('data_identity', [alias.name for alias in imports[0].names])
        self.assertNotIn('data_identity', self.functions)
        names = {node.id for node in ast.walk(self.tree) if isinstance(node, ast.Name)}
        self.assertNotIn('CHECKPOINT_SHA256', names)
        self.assertNotIn('prepare', names)

    def test_csv_entry_is_modality_specific_not_shared_or_all_rows(self):
        assignments = [node for node in ast.walk(self.functions['main']) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                               and target.value.id == 'args' and target.attr == 'csv_path'
                               for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        expression = compile(ast.Expression(assignments[0].value), str(TRAIN_PATH), 'eval')
        for modality in ('t1c', 't1n', 't2w', 't2f'):
            with self.subTest(modality=modality):
                path = eval(expression, {'ROOT': ROOT, 'str': str,
                                         'options': SimpleNamespace(modality=modality)})
                self.assertEqual(path, str(ROOT / f'dataset/lesions_{modality}.csv'))
                self.assertNotEqual(path, str(ROOT / 'dataset/lesions.csv'))
        loader_calls = [node for node in ast.walk(self.functions['main']) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == 'get_loader']
        self.assertEqual(len(loader_calls), 1)
        self.assertEqual(ast.dump(loader_calls[0].args[0]), ast.dump(ast.Name(id='args', ctx=ast.Load())))
        self.assertLess(assignments[0].lineno, loader_calls[0].lineno)
        config = json.loads((ROOT / 'config/RUN.json').read_text())
        self.assertIs(config['training']['use_all_rows'], False)
        all_rows_writes = [node for node in ast.walk(self.functions['main']) if isinstance(node, ast.Attribute)
                          and node.attr == 'use_all_rows' and isinstance(node.ctx, ast.Store)]
        self.assertEqual(all_rows_writes, [])
        self.assertIs(config['vae_reconstruction'], False)
        self.assertIs(config['synthesized_t2w'], False)

    def test_all_37_upstream_files_match_frozen_byte_sha(self):
        manifest = json.loads((ROOT / 'plan/UPSTREAM_SOURCE_MANIFEST.json').read_text())
        original = json.loads((ROOT / 'plan/UPSTREAM_SOURCE_MANIFEST.json').read_text())
        self.assertEqual(manifest['files_count'], 37)
        self.assertEqual(len(manifest['files']), 37)
        self.assertEqual(manifest['files'], original['files'])
        files = {path.relative_to(UPSTREAM).as_posix() for path in UPSTREAM.rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts}
        self.assertEqual(files, set(manifest['files']))
        for relative, expected in manifest['files'].items():
            with self.subTest(file=relative):
                path = UPSTREAM / relative
                self.assertFalse(path.is_symlink())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
