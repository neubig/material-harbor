"""Prepare primary runs and a key-free independent scientific audit."""
import json
from pathlib import Path

from adapter import ROOT, dump, DOCKER, CONFIG


def config(name, dataset, agent):
    return {'job_name': name, 'jobs_dir': str(ROOT / 'jobs'), 'n_concurrent_trials': 10,
            'retry': {'max_retries': 0}, 'environment': {'type': 'sail'},
            'agents': [agent], 'datasets': [{'path': str(dataset)}]}


def model(name, turns):
    return {'name': 'openhands-sdk', 'model_name': 'openai/' + name,
            'kwargs': {'load_skills': False, 'max_iterations': turns, 'temperature': 1},
            'env': {'LLM_API_KEY': '${LLM_API_KEY}', 'LLM_BASE_URL': 'https://llm-proxy.app.all-hands.dev'}}


def main():
    frozen = ROOT / 'frozen'
    sample = json.loads((frozen / 'sample.json').read_text())
    public = {r['uid']: r for r in json.loads((frozen / 'universe.json').read_text())}
    audit = ROOT / 'audit-tasks'
    for uid in sample['uids']:
        task = audit / uid
        for directory in ('environment', 'tests', 'solution'):
            (task / directory).mkdir(parents=True, exist_ok=True)
        # No source targets, solutions or solver outcomes are read to construct audit tasks.
        row = public[uid]
        (task / 'environment/Dockerfile').write_text(DOCKER)
        (task / 'task.toml').write_text(CONFIG.format(name='audit-' + uid).replace('/app/answer.json', '/app/audit.json'))
        contract = json.loads((frozen / 'contract.json').read_text())
        feature = row['features'].removeprefix('single_count_')
        prompt = f'''# Independent key-free scientific adjudication
{row['question']}
Presented SMILES: {json.loads(row['metadata'])['smiles']}
Definition: {contract['features'][feature]}
{contract['semantics']}
Independently derive the unique count under this definition. Do not consult source datasets, answer keys, prior solver outputs or prior judgments. They are absent and network access is restricted. Check atom identities and graph connectivity; explain any ambiguity or failure. Use RDKit as a parser if helpful but do not use MolecularIQ source solvers.
Write /app/audit.json as {{"key":"{feature}_count","count":integer or null,"answerable":boolean,"evidence":"scientific derivation and cross-check","confidence":"high|medium|low"}}.
You have a working budget of 40 turns.
'''
        (task / 'instruction.md').write_text(prompt)
        (task / 'tests/test.sh').write_text('''#!/bin/sh
python3 -I - <<'PYCODE'
import json
from pathlib import Path
reward = 0
try:
    x=json.loads(Path('/app/audit.json').read_text())
    reward=int(type(x) is dict and set(x)=={'key','count','answerable','evidence','confidence'} and type(x['answerable']) is bool and (x['count'] is None or type(x['count']) is int) and isinstance(x['evidence'],str) and bool(x['evidence']))
except (OSError,ValueError,TypeError):
    pass
Path('/logs/verifier').mkdir(parents=True,exist_ok=True)
Path('/logs/verifier/reward.txt').write_text(str(reward))
PYCODE
''')
        (task / 'solution/solve.sh').write_text('#!/bin/sh\nexit 1\n')
    dump(ROOT / 'oracle.json', config('moleculariq-oracle100', frozen / 'tasks', {'name': 'oracle'}))
    dump(ROOT / 'deepseek.json', config('moleculariq-deepseek100', frozen / 'tasks', model('deepseek-v4.1-flash', 50)))
    dump(ROOT / 'scientific.json', config('moleculariq-scientific100', audit, model('gpt-5.6', 50)))
    dump(ROOT / 'calibration-plan.json', {
        'sample': 'same preregistered SRS100 of complete eligible row universe',
        'auditor': 'GPT-5.6, temperature 1, max50/prompt40, key-free independent tasks; not human adjudication',
        'candidates': 'After judgments are persisted, one auditor-derived valid integer and one uniformly random alternative from 0..N excluding auditor count per row, seed 20261010. No source-value-based candidate filtering.',
        'rates': 'FN rejected / independently valid; FP accepted / independently invalid; Wilson 95% intervals. Missing/ambiguous judgments remain in cohort and reported separately, never replaced.',
        'limitations': 'Model adjudication shares RDKit parser semantics but not source solver; conditional on declared versioned contract. Synthetic uniform invalids are not the distribution of agent mistakes.'})


if __name__ == '__main__':
    main()
