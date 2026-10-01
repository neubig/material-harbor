import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_omni_cohort as cohort
import run_modal


class CohortTests(unittest.TestCase):
    @unittest.skipUnless((cohort.ROOT / 'datasets/measurements/omni-sdk-01/trials/smoke/result.json').exists() and (cohort.OUT / 'model/cal-15-015/trials/smoke/result.json').exists(), "Historical local measurement artifacts unavailable")
    def test_existing_completed_config_equivalence(self):
        smoke = cohort.read(cohort.ROOT / 'datasets/measurements/omni-sdk-01/trials/smoke/result.json')
        other = cohort.read(cohort.OUT / 'model/cal-15-015/trials/smoke/result.json')
        self.assertEqual(cohort.effective_config(smoke), cohort.effective_config(other))
        other['agent_info']['version'] = '1.50.2'
        with self.assertRaises(AssertionError):
            cohort.effective_config(other)

    @unittest.skipUnless((cohort.OUT / 'model/cal-18-003/trials/smoke/result.json').exists(), "Historical local measurement artifacts unavailable")
    def test_real_previous_failure_is_retryable(self):
        raw = cohort.read(cohort.OUT / 'model/cal-18-003/trials/smoke/result.json')
        self.assertTrue(cohort.retryable(raw))

    def test_unique_names(self):
        names = {cohort.trial_name('cal/02/001', run, attempt)
                 for run in ('abc', 'def') for attempt in (1, 2)}
        self.assertEqual(len(names), 4)
        self.assertTrue(all('/' not in n and len(n) < 64 for n in names))

    def test_only_pre_model_collision_retry(self):
        result = {'exception_info': {'exception_type': 'AlreadyExistsError'}, 'agent_result': None}
        self.assertTrue(cohort.retryable(result))
        result['agent_result'] = {'n_output_tokens': 1}
        self.assertFalse(cohort.retryable(result))
        result['exception_info']['exception_type'] = 'AgentTimeoutError'
        self.assertFalse(cohort.retryable(result))

    def test_timeout_does_not_pause(self):
        self.assertEqual(cohort.failure_kind({'exception_info': {'exception_type': 'AgentTimeoutError'}}, {}), 'agent_timeout')
        self.assertEqual(cohort.failure_kind({'exception_info': {'exception_type': 'AgentSetupTimeoutError'}}, {}), 'system')

    @unittest.skipUnless(cohort.task('cal/02/001').exists(), "Historical local measurement artifacts unavailable")
    def test_real_staging_and_command(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'attempt'
            staged = cohort.stage('cal/02/001', out)
            self.assertEqual(cohort.hashes(staged), cohort.hashes(cohort.task('cal/02/001')))
            self.assertEqual(set(p.relative_to(staged).as_posix() for p in staged.rglob('*') if p.is_file()), set(cohort.FILES))
            cmd = cohort.build_command(out, staged, 'unique-test')
            self.assertEqual(cmd[cmd.index('--trial-name') + 1], 'unique-test')
            self.assertIn('version=1.50.1', cmd)

    def test_real_run_trial_and_redaction(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = out / 'result.json'
            script = "import pathlib; print('test-secret'); pathlib.Path(%r).write_text(%r)" % (str(result), json.dumps({'exception_info': None}))
            summary = run_modal.run_trial([sys.executable, '-c', script], {'LLM_API_KEY': 'test-secret'}, out, result)
            self.assertEqual(summary['status'], 'completed')
            self.assertNotIn('test-secret', (out / 'runner.log').read_text())


if __name__ == '__main__':
    unittest.main()
