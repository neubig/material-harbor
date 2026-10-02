"""Frozen text-only MATRIX measurement, reconstructed official-style scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import urllib.request

import run_modal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'matrix/runs/fixed100-v2'
HANDOFF = ROOT.parent / 'matrix-measurement-handoff.json'


def load_module(name):
    spec = importlib.util.spec_from_file_location('matrix_' + name, ROOT / 'matrix' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_module('build')
verify = load_module('verify')
CONFIG = {'agent': 'openhands-sdk', 'version': '1.50.1', 'model': 'openai/deepseek-v4-flash',
          'max_iterations': 20, 'timeout': 600, 'temperature': 0, 'load_skills': False,
          'concurrency': 4, 'modal_environment': 'paper2rlenv', 'judge': verify.MODEL,
          'judge_temperature': 0, 'judge_reasoning_effort': 'none', 'judge_max_completion_tokens': 2048,
          'network': 'public (changed from archived unrun no-network pilot)',
          'architecture': 'native Harbor separate verifier, verifier-only environment templating',
          'fidelity': verify.FIDELITY}


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def progress(state, **extra):
    write(HANDOFF, {'status': state, 'pid': os.getpid(), 'updated_at': datetime.now(timezone.utc).isoformat(),
                    'config': CONFIG, 'output': str(OUT), **extra})


def freeze():
    build.fetch_source()
    manifest = build.build(OUT / 'tasks')
    hashes = {str(p.relative_to(OUT)): digest(p) for p in (OUT / 'tasks').rglob('*') if p.is_file()}
    write(OUT / 'protocol.json', {'config': CONFIG, 'manifest': manifest, 'task_hashes': hashes,
          'frozen_at': datetime.now(timezone.utc).isoformat(), 'rubrics': verify.RUBRICS,
          'failure_policy': 'All100 attempted once; no outcome stopping; missing/invalid answer=0; infrastructure unknown, no drops',
          'bootstrap': {'resamples': 100000, 'seed': 20260915, 'confidence': .95},
          'scope': '100 deterministic text items of220, not full470 multimodal benchmark',
          'code_hashes': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT/'matrix/build.py', ROOT/'matrix/verify.py']}})
    progress('frozen_before_outcomes', planned=100)


def validate():
    protocol = read(OUT / 'protocol.json')
    for name, value in protocol['task_hashes'].items():
        if digest(OUT / name) != value:
            raise ValueError('Frozen task changed')
    for name, value in protocol['code_hashes'].items():
        if digest(ROOT / name) != value:
            raise ValueError('Frozen implementation changed')
    return protocol


def judge_env():
    os.environ['MATRIX_JUDGE_URL'] = verify.ENDPOINT
    os.environ['MATRIX_JUDGE_MODEL'] = verify.MODEL
    os.environ['MATRIX_JUDGE_API_KEY'] = os.environ['LLM_API_KEY']


def sanity():
    protocol = validate()
    judge_env()
    request = urllib.request.Request(verify.ENDPOINT.replace('/chat/completions', '/models'),
              headers={'Authorization': 'Bearer ' + os.environ['LLM_API_KEY']})
    models = [r['id'] for r in json.load(urllib.request.urlopen(request, timeout=60))['data']]
    if verify.MODEL not in models:
        raise ValueError('Published GPT-5.1 unavailable; no substitution')
    directory = OUT / 'sanity'
    directory.mkdir(exist_ok=False)
    results = []
    wrong = 'All materials are made of cheese. Conservation of energy is false and atoms do not exist.'
    def one(item):
        gold = read(OUT / 'tasks' / ('matrix-' + item['qid']) / 'tests/gold.json')
        result = {'qid': item['qid'], 'kind': item['kind'], 'rubric': verify.rubric(item['kind'])}
        for name, answer in [('reference', gold['answer']), ('deliberately_wrong', wrong)]:
            try:
                result[name] = verify.judge(gold, answer)
            except Exception as exc:
                result[name] = {'score': None, 'error_type': type(exc).__name__}
        write(directory / (item['qid'] + '.json'), result)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(one, item) for item in protocol['manifest']['tasks']]):
            results.append(future.result())
            progress('judge_sanity_running', completed=len(results), planned=100)
    good = all(r['reference']['score'] is not None and r['reference']['score'] >= .75
               and r['deliberately_wrong']['score'] == 0 for r in results)
    write(OUT / 'sanity.json', {'passed': good, 'records': results,
          'interpretation': 'Reference-answer and synthetic wrong-answer sanity only, NOT scientific FP/FN validation'})
    progress('judge_sanity_complete', passed=good)


def run_one(item, smoke=False):
    qid = item['qid']
    directory = OUT / ('smoke' if smoke else 'model') / qid
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    task = OUT / 'tasks' / ('matrix-' + qid)
    env = run_modal.host_environment(Path.home() / '.env', 'openhands-sdk')
    if not all(env.get(k) for k in run_modal.CREDENTIALS):
        raise ValueError('Missing credentials')
    env['MODAL_ENVIRONMENT'] = 'paper2rlenv'
    env['LLM_BASE_URL'] = 'https://llm-proxy.app.all-hands.dev/v1'
    env['MATRIX_JUDGE_API_KEY'] = env['LLM_API_KEY']
    args = argparse.Namespace(output=directory, agent='openhands-sdk', model=CONFIG['model'],
           timeout=600, sandbox_timeout=1800, max_iterations=20, app_name='material-harbor')
    name = 'matrix-' + qid[:12] + ('-smoke' if smoke else '-fixed100')
    cmd = run_modal.command(args, task, name) + ['--agent-kwarg', 'version=1.50.1',
          '--verifier-env', 'MATRIX_JUDGE_API_KEY=${MATRIX_JUDGE_API_KEY}',
          '--verifier-env', 'MATRIX_JUDGE_MODEL=' + verify.MODEL,
          '--verifier-env', 'MATRIX_JUDGE_URL=' + verify.ENDPOINT]
    path = directory / 'trials' / name / 'result.json'
    summary = run_modal.run_trial(cmd, env, directory, path)
    raw = read(path) if path.exists() else {}
    grade_path = path.parent / 'verifier/result.json'
    grade = read(grade_path) if grade_path.exists() else {}
    exception = (raw.get('exception_info') or {}).get('exception_type')
    valid = summary['status'] == 'completed' and raw.get('agent_info', {}).get('version') == '1.50.1'
    reward = grade.get('reward') if valid else None
    record = {'qid': qid, 'kind': item['kind'], 'score': reward, 'verifier': grade,
              'exception_type': exception, 'summary_status': summary['status'],
              'result_path': str(path), 'agent_info': raw.get('agent_info'),
              'rubric': verify.rubric(item['kind'])}
    write(directory / 'record.json', record)
    return record


def statistics(records, planned=100):
    values = [r['score'] for r in records if r['score'] is not None]
    unknown = planned - len(values)
    result = {'denominator': planned, 'scored': len(values), 'unknown': unknown,
              'mean_rubric_score': None, 'bootstrap95': None,
              'full_denominator_bounds': [sum(values)/planned, (sum(values)+unknown)/planned],
              'resamples': 100000, 'seed': 20260915,
              'uncertainty_scope': 'IID item bootstrap; not judge systematic error or full multimodal generalization'}
    if unknown == 0:
        rng = random.Random(20260915)
        means = sorted(sum(rng.choices(values, k=planned))/planned for _ in range(100000))
        result.update(mean_rubric_score=sum(values)/planned, bootstrap95=[means[2499], means[97499]])
    return result


def run(smoke=False):
    protocol = validate()
    if not read(OUT / 'sanity.json')['passed']:
        raise ValueError('Sanity failed; requires review')
    items = protocol['manifest']['tasks']
    if smoke:
        record = run_one(items[0], True)
        write(OUT / 'smoke.json', record)
        progress('smoke_complete', record=record)
        return
    if read(OUT / 'smoke.json')['score'] is None:
        raise ValueError('End-to-end smoke failed; requires review')
    (OUT / 'run-started').open('x').close()
    results = []
    progress('running_fixed100', completed=0, planned=100)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_one, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {**item, 'score': None, 'error_type': type(exc).__name__}
            results.append(record)
            write(OUT / 'results.json', results)
            progress('running_fixed100', completed=len(results), planned=100)
    report = statistics(results)
    write(OUT / 'report.json', report)
    progress('complete_fixed100', completed=100, report=report,
             sanity='Synthetic/reference checks, not scientific FP/FN', raw_artifacts='Local ignored files only')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'sanity', 'smoke', 'run'])
    mode = parser.parse_args().mode
    os.umask(0o077)
    try:
        if mode == 'freeze':
            freeze()
        elif mode == 'sanity':
            sanity()
        else:
            run(mode == 'smoke')
    except Exception as exc:
        progress('blocked_' + mode, error_type=type(exc).__name__)
        raise SystemExit(2)
