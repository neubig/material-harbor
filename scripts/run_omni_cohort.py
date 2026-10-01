"""Fixed100 amendment with native gold checks and bounded Modal trials."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tomllib
import uuid

import subprocess
import sys

import run_modal
import summarize_accuracy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/measurements/omni-cohort'
FILES = ['environment/Dockerfile', 'environment/case.json', 'instruction.md',
         'solution/solve.sh', 'task.toml', 'tests/gold.json',
         'tests/native/eval_cal_results.py', 'tests/native/omnimat_paths.py',
         'tests/test.sh', 'tests/verify.py']
CONFIG = {'agent': 'openhands-sdk', 'model': 'openai/deepseek-v4-flash',
          'base_url': 'https://llm-proxy.app.all-hands.dev/v1',
          'modal_environment': 'paper2rlenv', 'max_iterations': 20,
          'load_skills': False, 'temperature': 0, 'timeout': 600,
          'sandbox_timeout': 1800, 'concurrency': 4, 'repetitions': 1}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task(key):
    return OUT / 'tasks' / ('omnimatbench-' + key.replace('/', '-'))


def hashes(directory):
    return {name: digest(directory / name) for name in FILES}


FIXED = OUT / 'fixed100'
HANDOFF = ROOT.parent / 'pragmatic-measurement-handoff.json'


def read(path):
    return json.loads(path.read_text())


def trial_name(key, run_id, attempt):
    return f"omni-{run_id}-{key.replace('/', '-')}-a{attempt}"


def retryable(result):
    return ((result.get('exception_info') or {}).get('exception_type') == 'AlreadyExistsError'
            and result.get('agent_result') is None and result.get('agent_execution') is None)


def failure_kind(result, summary):
    exception = (result.get('exception_info') or {}).get('exception_type')
    if exception == 'AgentTimeoutError':
        return 'agent_timeout'
    if exception or summary.get('status') != 'completed':
        return 'system'
    return None


def stage(key, directory):
    source = task(key)
    if source.is_symlink():
        raise ValueError('Symlink task root')
    files = run_modal.task_files(source, OUT / 'upload-manifest.json')
    assert {str(p) for p in files} == set(FILES)
    tomllib.loads((source / 'task.toml').read_text())
    directory.mkdir(parents=True, mode=0o700)
    staged = directory / 'task'
    for name in files:
        target = staged / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    write(directory / 'source-hashes.json', hashes(staged))
    write(directory / 'manifest.json', [str(p) for p in files])
    return staged


def build_command(directory, staged, name):
    args = argparse.Namespace(output=directory, agent=CONFIG['agent'], model=CONFIG['model'],
                              timeout=600, sandbox_timeout=1800, max_iterations=20,
                              app_name='material-harbor')
    return run_modal.command(args, staged, trial_name=name) + ['--agent-kwarg', 'version=1.50.1']


def effective_config(result):
    config = json.loads(json.dumps(result['config']))
    # An explicit install pin is equivalent only when the observed SDK version matches.
    assert result['agent_info']['version'] == '1.50.1'
    kwargs = config['agent']['kwargs']
    assert kwargs.get('version', '1.50.1') == '1.50.1'
    kwargs['version'] = '1.50.1'
    return summarize_accuracy.config_signature(config)


def record(key, path, oracle, expected, reused=False):
    raw = read(path) if path.exists() else {}
    directory = path.parents[2]
    summary = read(directory / 'summary.json') if (directory / 'summary.json').exists() else {}
    verifier_path = path.parent / 'verifier/omnimatbench-status.json'
    verifier = read(verifier_path) if verifier_path.exists() else {}
    kind = failure_kind(raw, summary)
    log_path = path.parent / 'agent/openhands_sdk.txt'
    log = log_path.read_text(errors='replace') if log_path.exists() else ''
    system_markers = ('litellm.AuthenticationError',
                      'APIConnectionError', 'RateLimitError', 'InternalServerError',
                      'litellm.BadRequestError', 'litellm.APIError')
    if not kind and (any(marker in log for marker in system_markers)
                     or not (raw.get('agent_result') or {}).get('n_output_tokens')):
        kind = 'system'
    issues = []
    if kind:
        issues.append(kind)
    if oracle.get('status') != 'scored' or oracle.get('reward') != 1:
        issues.append('oracle_failure')
    if verifier.get('status') not in ('scored', 'malformed', 'missing_or_unreadable'):
        issues.append('native_verifier_unknown')
    if not kind:
        try:
            assert effective_config(raw) == expected
            assert raw['task_name'] == 'omnimatbench/' + key.replace('/', '-')
        except (AssertionError, KeyError):
            issues.append('config_mismatch')
            kind = 'system'
    reward = (raw.get('verifier_result') or {}).get('rewards', {}).get('reward')
    if issues or reward not in (0, 1):
        reward = None
    return {'id': key, 'status': 'unknown' if reward is None else 'scored', 'reward': reward,
            'failure_kind': kind, 'infrastructure_failure': kind == 'system',
            'validation_issues': issues, 'oracle': oracle, 'verifier': verifier,
            'result_path': str(path), 'reused': reused, 'agent_result': raw.get('agent_result'),
            'exception_info': raw.get('exception_info')}


def freeze100():
    if FIXED.exists():
        raise ValueError('Fixed100 already exists; never overwrite frozen protocol')
    original = read(OUT / 'protocol.json')
    ids = original['selected_ids'][:100]
    assert len(ids) == len(set(ids)) == 100
    assert original['selected_ids'] == read(OUT / 'tasks/manifest.json')['selected']
    previous = read(OUT / 'progress.json')['records']
    smoke = read(ROOT / 'datasets/measurements/omni-sdk-01/trials/smoke/result.json')
    expected = effective_config(smoke)
    reuse = {}
    failures = {}
    for item in previous:
        key = item['id']
        if key not in ids:
            continue
        path = Path(item['result_path'])
        raw = read(path)
        if raw.get('exception_info'):
            failures[key] = {'result_path': str(path), 'sha256': digest(path), 'retryable': retryable(raw)}
            continue
        assert effective_config(raw) == expected
        assert raw['agent_result']['n_output_tokens'] > 0
        directory = path.parents[2]
        assert hashes(directory / 'task') == original['task_hashes'][key]
        batch = read(directory / 'batch-summary.json')
        for field in ['agent', 'model', 'base_url', 'modal_environment', 'max_iterations', 'load_skills', 'temperature']:
            assert batch[field] == CONFIG[field]
        reuse[key] = {'result_path': str(path), 'sha256': digest(path)}
    protocol = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'selected_ids': ids,
                'parent_protocol_sha256': digest(OUT / 'protocol.json'),
                'parent_progress_sha256': digest(OUT / 'progress.json'),
                'selection': 'Fixed first100 of frozen random.sample360 order; no outcome selection or adaptive expansion',
                'observed_before_amendment': [{'id': r['id'], 'reward': r['reward']} for r in previous if r['status'] == 'scored'],
                'config': {**CONFIG, 'sdk_version': '1.50.1'}, 'effective_config': expected,
                'expected_agent_info': smoke['agent_info'], 'reused': reuse, 'prior_failures': failures,
                'task_hashes': {key: original['task_hashes'][key] for key in ids},
                'runner_sha256': digest(ROOT / 'scripts/run_modal.py'),
                'orchestrator_sha256': digest(Path(__file__)), 'run_id': uuid.uuid4().hex[:12],
                'failure_policy': 'Retry only pre-model AlreadyExistsError once total; preserve failures. AgentTimeoutError unknown/no retry/continue. System init/API failures pause and drain.',
                'unknown_policy': 'Full denominator100, no dropping; conservative bounds and inconclusive if missing',
                'bootstrap': {'confidence': .95, 'resamples': 100000, 'seed': 20260915, 'strict_range': [.10, .70]},
                'config_normalization': 'Report-only copies add version pin to reused config only after observed SDK1.50.1 match; original results immutable'}
    write(FIXED / 'protocol.json', protocol)
    write(FIXED / 'frozen-ids.json', ids)
    for key in ids:
        assert hashes(task(key)) == protocol['task_hashes'][key]
        directory = FIXED / 'oracle' / key.replace('/', '-')
        directory.mkdir(parents=True)
        cmd = [sys.executable, str(task(key) / 'tests/verify.py'), '--answer', str(task(key) / 'tests/gold.json'),
               '--gold', str(task(key) / 'tests/gold.json'), '--reward', str(directory / 'reward.txt')]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        (directory / 'stdout.txt').write_text(proc.stdout)
        (directory / 'stderr.txt').write_text(proc.stderr)
        status = directory / 'omnimatbench-status.json'
        write(directory / 'result.json', {'id': key, 'exit_code': proc.returncode,
                                         **(read(status) if status.exists() else {'status': 'oracle_process_failure'})})
    write(FIXED / 'preflight.json', {'oracle_count': 100, 'protocol_sha256': digest(FIXED / 'protocol.json'),
                                   'oracle_failures': [key for key in ids if read(FIXED / 'oracle' / key.replace('/', '-') / 'result.json').get('reward') != 1]})
    print('Frozen first100 and checked all100 native gold oracles', flush=True)


def run_one(key, protocol, env):
    prior = protocol['prior_failures'].get(key)
    if prior:
        assert digest(Path(prior['result_path'])) == prior['sha256']
    oracle = read(FIXED / 'oracle' / key.replace('/', '-') / 'result.json')
    attempts = [prior] if prior else []
    if prior and not prior['retryable']:
        return record(key, Path(prior['result_path']), oracle, protocol['effective_config'])
    first = 2 if prior else 1
    for attempt in range(first, 3):
        name = trial_name(key, protocol['run_id'], attempt)
        directory = FIXED / 'model' / key.replace('/', '-') / f'attempt-{attempt}'
        staged = stage(key, directory)
        assert hashes(staged) == protocol['task_hashes'][key]
        path = directory / 'trials' / name / 'result.json'
        run_modal.run_trial(build_command(directory, staged, name), env, directory, path)
        raw = read(path) if path.exists() else {}
        attempts.append({'result_path': str(path), 'sha256': digest(path) if path.exists() else None})
        item = record(key, path, oracle, protocol['effective_config'])
        item['attempts'] = attempts.copy()
        write(directory / 'record.json', item)
        if not retryable(raw) or attempt == 2:
            return item


def report(protocol, results):
    by_id = {r['id']: r for r in results}
    tasks = []
    for key in protocol['selected_ids']:
        row = by_id.get(key, {})
        path = Path(row['result_path']) if row.get('result_path') else None
        report_path = None
        if path and path.exists():
            raw = read(path)
            if row.get('reused'):
                effective_config(raw)
                raw['config']['agent']['kwargs']['version'] = '1.50.1'
                report_path = FIXED / 'report-inputs' / (key.replace('/', '-') + '.json')
                write(report_path, raw)
                write(report_path.with_suffix('.provenance.json'), {'original': str(path), 'sha256': digest(path), 'normalization': protocol['config_normalization']})
            else:
                report_path = path
        tasks.append({'task_id': 'omnimatbench/' + key.replace('/', '-'),
                      'result_path': str(report_path) if report_path else None,
                      'validation_issues': row.get('validation_issues', [])})
    manifest = {'sampling_plan': protocol['selection'] + '; protocol sha256=' + digest(FIXED / 'protocol.json'),
                'expected_config': protocol['effective_config'], 'expected_agent_info': protocol['expected_agent_info'], 'tasks': tasks}
    write(FIXED / 'summary-manifest.json', manifest)
    result = summarize_accuracy.summarize(manifest, FIXED)
    write(FIXED / 'accuracy-report.json', result)
    return result


def run100():
    protocol = read(FIXED / 'protocol.json')
    ids = protocol['selected_ids']
    assert ids == read(OUT / 'protocol.json')['selected_ids'][:100] == read(FIXED / 'frozen-ids.json')
    assert len(ids) == len(set(ids)) == 100
    assert digest(OUT / 'protocol.json') == protocol['parent_protocol_sha256']
    assert digest(OUT / 'progress.json') == protocol['parent_progress_sha256']
    assert digest(ROOT / 'scripts/run_modal.py') == protocol['runner_sha256']
    assert digest(Path(__file__)) == protocol['orchestrator_sha256']
    assert read(FIXED / 'preflight.json')['oracle_count'] == 100
    assert read(FIXED / 'preflight.json')['protocol_sha256'] == digest(FIXED / 'protocol.json')
    for key in ids:
        assert hashes(task(key)) == protocol['task_hashes'][key]
        assert (FIXED / 'oracle' / key.replace('/', '-') / 'result.json').exists()
    if (FIXED / 'progress.json').exists():
        raise ValueError('Existing run; no implicit restart/retries')
    env = run_modal.host_environment(Path.home() / '.env', CONFIG['agent'])
    assert all(env.get(k) for k in run_modal.CREDENTIALS)
    env['MODAL_ENVIRONMENT'] = CONFIG['modal_environment']
    env['LLM_BASE_URL'] = CONFIG['base_url']
    results = []
    for key, reuse in protocol['reused'].items():
        path = Path(reuse['result_path'])
        assert digest(path) == reuse['sha256']
        results.append(record(key, path, read(FIXED / 'oracle' / key.replace('/', '-') / 'result.json'), protocol['effective_config'], True))
    remaining = iter(key for key in ids if key not in protocol['reused'])
    paused = False
    pending = {}
    (FIXED / 'orchestrator.pid').write_text(str(os.getpid()) + '\n')

    def save(state):
        progress = {'state': state, 'pid': os.getpid(), 'planned': 100, 'finished': len(results),
                    'in_flight': list(pending.values()), 'updated_at': datetime.now(timezone.utc).isoformat()}
        write(FIXED / 'progress.json', {**progress, 'records': results})
        write(FIXED / 'results.json', results)
        write(HANDOFF, {**progress, 'measurement_complete': state == 'complete_trials',
                       'protocol': str(FIXED / 'protocol.json'), 'progress': str(FIXED / 'progress.json'),
                       'log': str(FIXED / 'orchestrator.log'), 'report': str(FIXED / 'accuracy-report.json'),
                       'reused': list(protocol['reused']), 'old_artifacts_unchanged': True,
                       'no_accuracy_claim_until_complete': True})

    save('running')
    with ThreadPoolExecutor(max_workers=4) as pool:
        def submit():
            key = next(remaining, None)
            if key:
                pending[pool.submit(run_one, key, protocol, env)] = key
        for _ in range(4):
            submit()
        save('running')
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                key = pending.pop(future)
                try:
                    item = future.result()
                except Exception as exc:
                    item = {'id': key, 'status': 'unknown', 'reward': None, 'infrastructure_failure': True,
                            'validation_issues': ['orchestrator_' + type(exc).__name__]}
                results.append(item)
                paused |= item['infrastructure_failure']
                write(FIXED / 'records' / (key.replace('/', '-') + '.json'), item)
                print(json.dumps({'finished': len(results), 'id': key, 'status': item['status'], 'paused': paused}), flush=True)
            if not paused:
                for _ in done:
                    submit()
            save('draining_infrastructure_pause' if paused else 'running')
    save('paused_infrastructure' if paused else 'complete_trials')
    final = report(protocol, results)
    handoff = read(HANDOFF)
    handoff['final_report'] = final
    write(HANDOFF, handoff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze100', 'run100'])
    args = parser.parse_args()
    freeze100() if args.mode == 'freeze100' else run100()
