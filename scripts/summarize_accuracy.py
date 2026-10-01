"""Report a frozen, fixed-size set of Harbor binary trials (no model calls).

Manifest: {sampling_plan: str, expected_config: Harbor trial config,
expected_agent_info: Harbor agent_info, tasks: [{task_id: Harbor task_name,
result_path: str|null, paper_id: str (required for --scope paper),
validation_issues: optional list[str] of oracle/native/infrastructure failures}]}.
Harbor reward alone cannot detect swallowed native verifier failures: supply
validation_issues from the independent native/oracle checks when applicable.
Paths are relative to the manifest. Freeze this manifest before collecting results;
its hash records identity, not proof of prospective freezing. Never stop on outcomes.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


RESAMPLES = 100000
SEED = 20260915
# Only task identity and output bookkeeping differ across otherwise identical trials.
CONFIG_EXCLUSIONS = {"task", "trial_name", "trials_dir", "job_id"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def config_signature(config):
    return {key: value for key, value in config.items() if key not in CONFIG_EXCLUSIONS}


def classify(ci):
    low, high = ci
    if low > .10 and high < .70:
        return 'within'
    if high < .10 or low > .70:
        return 'outside'
    return 'inconclusive'


def wilson_ci(successes, n):
    z = NormalDist().inv_cdf(.975)
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def bootstrap_ci(rewards, papers=None, seed=SEED, resamples=RESAMPLES):
    rng = np.random.default_rng(seed)
    n = len(rewards)
    if papers is None:
        # An IID resampled binary sum is exactly Binomial(n, empirical p).
        samples = rng.binomial(n, sum(rewards)/n, size=resamples) / n
    else:
        groups = {}
        for reward, paper in zip(rewards, papers, strict=True):
            group = groups.setdefault(paper, [0, 0])
            group[0] += reward
            group[1] += 1
        totals = np.asarray(list(groups.values()))
        k = len(groups)
        samples = np.empty(resamples)
        batch_size = max(1, min(4096, 1000000 // k))
        for start in range(0, resamples, batch_size):
            stop = min(start + batch_size, resamples)
            counts = rng.multinomial(k, np.full(k, 1/k), size=stop-start)
            samples[start:stop] = (counts @ totals[:, 0]) / (counts @ totals[:, 1])
    return np.quantile(samples, [.025, .975], method='linear').tolist()


def summarize(manifest, base_dir, scope='iid'):
    if scope not in ('iid', 'paper'):
        raise ValueError('scope must be iid or paper')
    tasks = manifest.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Nonempty explicit frozen tasks list is required')
    for field in ('sampling_plan', 'expected_config', 'expected_agent_info'):
        if not manifest.get(field):
            raise ValueError(f'Manifest requires {field}')
    expected = config_signature(manifest['expected_config'])
    expected_info = manifest['expected_agent_info']
    if not expected.get('agent') or not expected_info.get('name'):
        raise ValueError('Expected agent configuration and identity are required')
    ids, paths, trial_ids = set(), set(), set()
    rows, rewards, papers = [], [], []
    for task in tasks:
        task_id = task.get('task_id')
        if not isinstance(task_id, str) or not task_id:
            raise ValueError('Every expected task requires a nonempty task_id')
        if task_id in ids:
            raise ValueError(f'Duplicate task ID: {task_id}')
        ids.add(task_id)
        if scope == 'paper' and (not isinstance(task.get('paper_id'), str) or not task['paper_id']):
            raise ValueError(f'Missing paper_id: {task_id}')
        row = {'task_id': task_id, 'paper_id': task.get('paper_id'),
               'result_path': task.get('result_path'),
               'issues': [f'external_validation:{issue}' for issue in task.get('validation_issues', [])],
               'reward': None}
        rows.append(row)
        if not task.get('result_path'):
            row['issues'].append('missing_result_path')
            continue
        path = (Path(base_dir) / task['result_path']).resolve()
        if path in paths:
            raise ValueError(f'Duplicate result path: {path}')
        paths.add(path)
        try:
            raw = path.read_bytes()
            row['result_sha256'] = hashlib.sha256(raw).hexdigest()
            result = json.loads(raw)
        except (OSError, ValueError) as exc:
            row['issues'].append(f'unreadable_result:{type(exc).__name__}')
            continue
        if not isinstance(result, dict):
            row['issues'].append('invalid_result_object')
            continue
        trial_id = result.get('id')
        if not isinstance(trial_id, str) or not trial_id:
            row['issues'].append('missing_trial_id')
        elif trial_id in trial_ids:
            raise ValueError(f'Duplicate trial ID: {trial_id}')
        else:
            trial_ids.add(trial_id)
        row['trial_id'] = trial_id
        row['task_checksum'] = result.get('task_checksum')
        if result.get('task_name') != task_id:
            row['issues'].append('task_identity_mismatch')
        if task.get('task_checksum') and task['task_checksum'] != result.get('task_checksum'):
            row['issues'].append('task_checksum_mismatch')
        config = result.get('config')
        if not isinstance(config, dict):
            row['issues'].append('missing_config')
        else:
            actual = config_signature(config)
            row['config_sha256'] = digest(actual)
            row['full_config_sha256'] = digest(config)
            for key in sorted(set(expected) | set(actual)):
                if key not in expected or key not in actual or expected[key] != actual[key]:
                    row['issues'].append(f'config_mismatch:{key}')
        if result.get('agent_info') != expected_info:
            row['issues'].append('agent_or_model_identity_mismatch')
        if result.get('exception_info') is not None:
            row['issues'].append('trial_exception')
            row['exception_info'] = result['exception_info']
        if not result.get('finished_at'):
            row['issues'].append('unfinished_trial')
        verifier = result.get('verifier_result')
        reward_map = verifier.get('rewards') if isinstance(verifier, dict) else None
        reward = reward_map.get('reward') if isinstance(reward_map, dict) else None
        if type(reward) not in (int, float) or reward not in (0, 1):
            row['issues'].append('missing_or_nonbinary_reward')
        else:
            row['reward'] = int(reward)
        if not row['issues']:
            rewards.append(row['reward'])
            papers.append(task.get('paper_id'))
    n, observed = len(tasks), len(rewards)
    successes = sum(rewards)
    complete = observed == n
    warnings = ['Fixed sample size only; repeated outcome-dependent stopping invalidates nominal coverage.',
                'Empirical bootstrap reflects sampled tasks/papers, not model-run randomness or unrepresented populations.']
    ci = bootstrap_ci(rewards, papers if scope == 'paper' else None) if complete else None
    wilson = wilson_ci(successes, n) if complete and scope == 'iid' else None
    decision = classify(ci) if complete else 'incomplete'
    bootstrap_decision = decision
    if complete and scope == 'iid' and classify(wilson) != decision:
        decision = 'inconclusive'
    if complete and successes in (0, n):
        warnings.append('All-zero/all-one empirical percentile CI is degenerate and cannot exclude unseen outcomes; IID classification additionally requires Wilson agreement.')
        if scope == 'paper':
            decision = 'inconclusive'
            warnings.append('No IID Wilson guard is valid for correlated papers; degenerate cluster results remain inconclusive.')
    if scope == 'paper':
        warnings.append('Cluster resampling uses total successes / total tasks, not unweighted cluster means; few papers can yield unreliable coverage.')
        if complete and len(set(papers)) < 2:
            decision = 'inconclusive'
    return {
        'classification': decision, 'bootstrap_classification': bootstrap_decision,
        'point_estimate': successes/n if complete else None,
        'observed_only_point_estimate': successes/observed if observed else None,
        'unknown_outcome_bounds': [successes/n, (successes+n-observed)/n],
        'bootstrap_ci': ci, 'wilson_ci': wilson,
        'denominators': {'expected': n, 'observed_binary': observed, 'successes': successes,
                         'reward_failures': observed-successes, 'unresolved': n-observed},
        'failure_counts': dict(sorted(Counter(issue for row in rows for issue in row['issues']).items())),
        'expected_papers': len({task['paper_id'] for task in tasks}) if scope == 'paper' else None,
        'protocol': {'confidence': .95, 'method': 'empirical percentile', 'quantile_method': 'linear',
                     'resamples': RESAMPLES, 'seed': SEED, 'scope': scope, 'rng': 'PCG64',
                     'numpy_version': np.__version__, 'target_open_interval': [.10, .70],
                     'sampling_plan': manifest['sampling_plan'],
                     'classification_guard': 'Wilson agreement' if scope == 'iid' else 'nondegenerate, at least two papers'},
        'manifest_sha256': digest(manifest), 'expected_config_sha256': digest(expected),
        'expected_agent_info_sha256': digest(expected_info),
        'config_excluded_fields': sorted(CONFIG_EXCLUSIONS), 'warnings': warnings, 'trials': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--scope', choices=['iid', 'paper'], default='iid')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = summarize(json.loads(args.manifest.read_text()), args.manifest.parent, args.scope)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
