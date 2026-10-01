import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'matrix'


def module(name):
    spec = importlib.util.spec_from_file_location('matrix_' + name, ROOT / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


build = module('build')
verify = module('verify')
SOURCE = ROOT / 'source/test.jsonl'
HAS_SOURCE = unittest.skipUnless(SOURCE.is_file(), 'Fetch pinned source with matrix/build.py --fetch --fetch-only')
GOLD = {'question': 'Explain diffusion.', 'kind': 'foundational', 'answer': 'Transport down a chemical potential gradient.'}



class MatrixTests(unittest.TestCase):
    @HAS_SOURCE
    def test_source_and_frozen_selection(self):
        rows = build.load_rows()
        selected = build.select(rows)
        self.assertEqual(selected, build.select(list(reversed(rows))))
        manifest = json.loads((ROOT / 'pilot-manifest.json').read_text())
        self.assertEqual([r['qid'] for r in selected], [r['qid'] for r in manifest['tasks']])
        self.assertEqual(len(selected), 10)
        self.assertTrue(all(r['type'] == 'text' for r in selected))
        self.assertIn('diffusion', next(r['answer'] for r in rows if r['qid'] == 'd8773d95691e4137b000334cd952b5bf'))

    @HAS_SOURCE
    def test_generation_preserves_native_and_hides_gold(self):
        from harbor.models.task.config import TaskConfig
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'tasks'
            build.build(out)
            for row in build.select(build.load_rows()):
                task = out / ('matrix-' + row['qid'])
                self.assertTrue((task / 'instruction.md').read_text().startswith(row['question'] + '\n\n'))
                self.assertEqual(json.loads((task / 'tests/gold.json').read_text()), row)
                self.assertEqual(list((task / 'environment').iterdir()), [task / 'environment/Dockerfile'])
                self.assertNotIn('COPY', (task / 'environment/Dockerfile').read_text())
                config = TaskConfig.model_validate(tomllib.loads((task / 'task.toml').read_text()))
                self.assertEqual(config.verifier.environment_mode.value, 'separate')
                self.assertEqual(config.environment.env, {})

    def test_fetch_rejects_bad_bytes_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'bad.jsonl'
            source.write_bytes(b'not the pinned dataset')
            destination = root / 'cache/test.jsonl'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                build.fetch_source(destination, local_file=source)
            self.assertFalse(destination.exists())
            destination.parent.mkdir()
            destination.write_bytes(b'corrupt cache')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                build.fetch_source(destination, local_file=source)
            self.assertEqual(destination.read_bytes(), b'corrupt cache')

    @HAS_SOURCE
    def test_offline_fetch_cache_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / 'cache/test.jsonl'
            build.fetch_source(destination, local_file=SOURCE)
            self.assertEqual(destination.read_bytes(), SOURCE.read_bytes())
            build.fetch_source(destination, local_file=root / 'nonexistent')
            self.assertEqual(list(destination.parent.iterdir()), [destination])
            command = [sys.executable, str(ROOT / 'build.py'), '--fetch',
                       '--source-file', str(SOURCE), '--source', str(destination),
                       '--output', str(root / 'tasks')]
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(len(list((root / 'tasks').iterdir())), 10)
            before = (root / 'pilot-manifest.json').read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((root / 'pilot-manifest.json').read_bytes(), before)

    def test_validation_survives_optimized_python(self):
        command = [sys.executable, '-O', '-c',
                   "import runpy; runpy.run_path(" + repr(str(ROOT / 'build.py')) + ")[\"validate_rows\"]([])"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ValueError', result.stderr)

    def test_harbor_network_semantics(self):
        from harbor.models.task.config import TaskConfig, VerifierEnvironmentMode
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig
        from harbor.trial.network_policy import resolve_trial_network_plan

        task = TaskConfig.model_validate({
            'environment': {'network_mode': 'no-network'},
            'verifier': {'environment_mode': 'separate', 'environment': {'cpus': 1, 'memory_mb': 512}},
        })

        def plan(agent=None, environment=None):
            return resolve_trial_network_plan(
                task, agent or AgentConfig(), environment or EnvironmentConfig(), None,
                verifier_mode=VerifierEnvironmentMode.SEPARATE)

        baseline = plan()
        self.assertEqual(baseline.agent_phase.network_mode.value, 'no-network')
        self.assertEqual(baseline.verifier_phase.network_mode.value, 'public')
        phase = plan(agent=AgentConfig(extra_allowed_hosts=['llm.example.org']))
        self.assertEqual(phase.agent_env_baseline.network_mode.value, 'no-network')
        self.assertEqual(phase.agent_phase.network_mode.value, 'allowlist')
        self.assertEqual(phase.agent_phase.allowed_hosts, ['llm.example.org'])
        startup = plan(environment=EnvironmentConfig(extra_allowed_hosts=['llm.example.org']))
        self.assertEqual(startup.agent_env_baseline, startup.agent_phase)
        self.assertEqual(startup.agent_phase.allowed_hosts, ['llm.example.org'])
        self.assertEqual(startup.verifier_phase, baseline.verifier_phase)
        self.assertEqual(task.environment.network_mode.value, 'no-network')

    def test_strict_judge_parser(self):
        for value, expected in [('true', 1), ('false', 0), ('1', 1), ('0', 0)]:
            self.assertEqual(verify.parse_judgment('{"correct":' + value + '}'), expected)
        for text in ['{"correct":"false"}', '{"correct":0.0}', '{"correct":2}',
                     '{"correct":null}', '{"correct":true,"extra":0}',
                     '{"correct":true,"correct":false}', 'true', '[]',
                     '```json\n{"correct":true}\n```', '{"correct":NaN}',
                     '{"correct": true} trailing', '__import__("os").system("true")']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                verify.parse_judgment(text)

    def test_missing_and_invalid_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'answer.txt'
            gold = GOLD
            self.assertEqual(verify.evaluate(gold, path)['reward'], 0)
            for content in [b'   ', b'\xff', b'a' * 65537]:
                path.write_bytes(content)
                self.assertEqual(verify.evaluate(gold, path)['reward'], 0)
            code = '__import__("os").system("touch NEVER_EXECUTE")'
            path.write_text(code)
            self.assertEqual(verify.read_answer(path), code)
            path.unlink()
            os.mkfifo(path)
            with self.assertRaises(ValueError):
                verify.read_answer(path)

    def test_cli_infrastructure_not_wrong_and_stale_reward_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / 'gold.json'
            gold.write_text(json.dumps(GOLD))
            answer = root / 'answer.txt'
            logs = root / 'logs'
            command = [sys.executable, str(ROOT / 'verify.py'), '--gold', str(gold),
                       '--answer', str(answer), '--logs', str(logs)]
            env = {k: v for k, v in os.environ.items() if not k.startswith('MATRIX_JUDGE_')}
            self.assertEqual(subprocess.run(command, env=env).returncode, 0)
            self.assertEqual((logs / 'reward.txt').read_text(), '0')
            answer.write_text('A nonempty scientific response')
            self.assertEqual(subprocess.run(command, env=env).returncode, 2)
            self.assertFalse((logs / 'reward.txt').exists())
            result = json.loads((logs / 'result.json').read_text())
            self.assertEqual(result['status'], 'infrastructure_error')
            self.assertIsNone(result['reward'])


if __name__ == '__main__':
    unittest.main()
