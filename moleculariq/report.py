"""Summarize retained trials; never substitute missing outputs or judgments."""
import hashlib
import importlib.metadata
import json
import math
import random
from collections import Counter
from pathlib import Path

from verifier import accepts

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def wilson(errors, n):
    if not n:
        return None
    z = 1.959963984540054
    p = errors / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [max(0, center-radius), min(1, center+radius)]


def summarize(name, uids):
    directory = ROOT / 'jobs' / name
    trials = [load(p) for p in sorted(directory.glob('*/result.json'))]
    by_uid = {r['trial_name'].split('__')[0]: r for r in trials}
    rows = []
    for uid in uids:
        r = by_uid.get(uid)
        reward = ((r or {}).get('verifier_result') or {}).get('rewards', {}).get('reward')
        rows.append({'uid': uid, 'trial_id': (r or {}).get('id'), 'trial_name': (r or {}).get('trial_name'),
                     'reward': reward, 'exception': (r or {}).get('exception_info'),
                     'agent_usage': (r or {}).get('agent_result'), 'agent_info': (r or {}).get('agent_info')})
    job = load(directory / 'result.json') if (directory / 'result.json').exists() else {}
    successes = sum(r['reward'] == 1 for r in rows)
    return {'job_id': job.get('id'), 'finished_at': job.get('finished_at'), 'planned': len(uids),
            'completed_trial_records': len(trials), 'successes': successes,
            'accuracy_all_selected_denominator': successes / len(uids),
            'accuracy_wilson95': wilson(successes, len(uids)),
            'missing_rewards': sum(r['reward'] is None for r in rows),
            'exceptions': dict(Counter(r['exception']['exception_type'] for r in rows if r['exception'])),
            'stats': job.get('stats'), 'trials': rows}


def calibration(uids):
    directory = ROOT / 'jobs/moleculariq-scientific100'
    rng = random.Random(20261010)
    rows = []
    for uid in uids:
        outputs = list(directory.glob(f'{uid}__*/artifacts/**/audit.json'))
        if len(outputs) != 1:
            rows.append({'uid': uid, 'status': 'missing_or_multiple_judgments', 'outputs': len(outputs)})
            continue
        try:
            raw = outputs[0].read_bytes()
            judgment = json.loads(raw)
            ref = load(ROOT / 'frozen/tasks' / uid / 'tests/reference.json')
            count = judgment['count']
            if (judgment['answerable'] is not True or type(count) is not int
                    or not 0 <= count <= ref['upper_bound'] or judgment['key'] != ref['key']
                    or not judgment.get('evidence')):
                raise ValueError('unresolved scientific judgment')
            wrong = rng.choice([v for v in range(ref['upper_bound'] + 1) if v != count])
            rows.append({'uid': uid, 'status': 'adjudicated', 'judgment': judgment,
                         'judgment_sha256': hashlib.sha256(raw).hexdigest(),
                         'valid_candidate': count, 'invalid_candidate': wrong,
                         'false_negative': not accepts(json.dumps({ref['key']: count}), ref),
                         'false_positive': accepts(json.dumps({ref['key']: wrong}), ref)})
        except (ValueError, KeyError, TypeError) as exc:
            rows.append({'uid': uid, 'status': 'invalid_judgment', 'error': str(exc)})
    judged = [r for r in rows if r['status'] == 'adjudicated']
    n = len(judged)
    fp = sum(r['false_positive'] for r in judged)
    fn = sum(r['false_negative'] for r in judged)
    return {'planned': len(uids), 'adjudicated': n, 'unresolved': len(uids)-n,
            'scope': 'Available-judgment conditional rates only; not representative full-random100 rates when any judgment is missing.',
            'FP': {'errors': fp, 'invalid_denominator': n, 'planned_denominator':len(uids), 'rate': fp/n if n else None, 'wilson95': wilson(fp,n)},
            'FN': {'errors': fn, 'valid_denominator': n, 'planned_denominator':len(uids), 'rate': fn/n if n else None, 'wilson95': wilson(fn,n)},
            'missingness_worst_case_error_rate_bounds': {'FP':[fp/len(uids),(fp+len(uids)-n)/len(uids)], 'FN':[fn/len(uids),(fn+len(uids)-n)/len(uids)]},
            'plan': load(ROOT / 'calibration-plan.json'), 'rows': rows}


def main():
    sample = load(ROOT / 'frozen/sample.json')
    uids = sample['uids']
    jobs = {kind: summarize('moleculariq-' + name + '100', uids)
            for kind, name in [('oracle','oracle'),('deepseek','deepseek'),('scientific','scientific')]}
    cal = calibration(uids)
    audit = load(ROOT.parents[1] / 'research-moleculariq-continuation/research-moleculariq-continuation.json')
    complete = all(j['finished_at'] for j in jobs.values())
    report = {'status': ('BLOCKED_PARTIAL_SCIENTIFIC_CALIBRATION' if cal['unresolved'] else 'EVALUATIONS_COMPLETE') if complete else 'RUNNING_NOT_YET_QUALIFIED',
              'scope': 'Faithful six-feature bounded-count subset with explicit serialization and version semantics, NOT full 5111-row benchmark.',
              'pins': {**audit['pins'], 'harbor_executed_main': 'd5ac1be17f575852eaf4fffc4072fd18481c209b',
                       'rdkit': '2025.09.4', 'sail_sdk': importlib.metadata.version('sail'),
                       'dataset_sha256': '00539423f440a6e02524a8a995a0ddb77c0f525505de53a8dd4105f60e78bfc6'},
              'contract': load(ROOT / 'frozen/contract.json'), 'sample': sample,
              'population': load(ROOT / 'frozen/population.json'),
              'upstream_context': {'released_rows':5111,'canonical_molecules':854,
                                   'version_reconciliation':audit['label_audit'],
                                   'licensing': 'Not evaluated as a blocker, per user instruction.'},
              'tests': {'log':'tests.log','result':(ROOT/'tests.log').read_text(),
                        'build_failures':load(ROOT/'frozen/build_failures.json'),
                        'claims':'Mechanical exhaustive tests are separate from independent scientific FP/FN.'},
              'jobs': jobs, 'scientific_calibration': cal,
              'blocker': {'type':'LLM_PROXY_BUDGET_EXCEEDED',
                          'audit_api_errors': jobs['scientific']['exceptions'],
                          'missing_judgments': cal['unresolved'],
                          'interpretation':'92 scientific judgments saved; 13 API errors, including 5 trials that saved usable judgments before a later API error. Missing 8 remain in the planned100 cohort. No replacements or retries.',
                          'next_step_requires_confirmation':'Restore proxy budget, then authorize same-ID retries of the eight missing audit judgments and Sail roundtrip calibration of frozen valid/invalid candidates.',
                          'sail_candidate_roundtrip':'Prepared implementation, not executed after budget blocker; local strict-verifier calibration on available judgments only.'},
              'sail': {'box_ids_file':'sail-boxes.json', 'infra_cost_usd':None,
                       'infra_cost_note':'Sail billing cost not exposed in Harbor trial results; unknown, not zero.',
                       'llm_cost_usd':{k:(v.get('stats') or {}).get('cost_usd') for k,v in jobs.items()}},
              'isolation': {'build_context':'environment/Dockerfile only, no targets/solutions/parquet',
                            'verifier':'separate fresh Sail environment, tests injected after agent; only answer artifact transferred',
                            'network':'agent phase allowlist only llm-proxy.app.all-hands.dev; public setup for SDK installation; no source keys in setup',
                            'audit':'separate tasks built exclusively from key-free frozen universe; no source solvers, targets or model outputs'},
              'limitations':['Subset qualification only; 97 molecules among sampled 100 rows, not statistically independent molecules.',
                             'GPT-5.6 is a model adjudicator, not a human; shares RDKit parser but not source solver.',
                             'Base python image and installation dependencies were not digest/lock pinned before run; Sail image IDs and observed SDK versions retained.',
                             'Uniform finite-domain invalid candidates do not estimate false acceptance among natural model errors.'],
              'C:J': {'C':'Yes: molecular graph reasoning', 'D':'5111 released rows / 854 molecules; eligible subset 179 rows / 170 molecules',
                      'E':'ChemDFM-R-14B, LlaSMol-Mistral-7B, ether0 evaluated in source paper; not run here',
                      'F':f"Harbor latest-main adapter; oracle {jobs['oracle']['successes']}/100, missing {jobs['oracle']['missing_rewards']}",
                      'G':f"DeepSeek-v4.1-flash t1 max50/prompt40: {jobs['deepseek']['successes']}/100, missing {jobs['deepseek']['missing_rewards']}; subset only",
                      'H':cal['FP'], 'I':cal['FN'],
                      'J':'Pending complete runs/calibration; no full-benchmark qualification' if not complete or cal['unresolved'] else 'Completed subset evaluation; inspect retained errors and confidence intervals; no full-benchmark qualification'},
              'scope_preservation':'Only moleculariq/ and tests/test_moleculariq.py intentionally written; workbook untouched; no commit/push.',
              'disclosure':'Generated by an AI agent (OpenHands) on behalf of the user.'}
    report['artifact_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/'adapter.py',ROOT/'verifier.py',ROOT/'frozen/universe.json',ROOT/'frozen/sample.json',
                  ROOT/'calibration-plan.json',ROOT/'sail-boxes.json']}
    (ROOT/'research-moleculariq-harborization.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'], {k:(v['successes'],v['missing_rewards']) for k,v in jobs.items()}, 'audits',cal['adjudicated'])


if __name__ == '__main__':
    main()
