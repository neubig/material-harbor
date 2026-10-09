import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset
from main import DATASETS, REVISIONS

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'tests/research-bioreason-verification'
SEED = 202610091850


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def main():
    if (OUT / 'manifest.json').exists():
        raise SystemExit('Frozen manifest exists; refusing resampling')
    universe, rows = [], {}
    for setting, repo in DATASETS.items():
        data = load_dataset(repo, split='test', revision=REVISIONS[setting])
        for index, row in enumerate(data):
            identity = f'{setting}:{index}'
            universe.append({'id': identity, 'setting': setting, 'index': index, 'revision': REVISIONS[setting], 'row_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()})
            rows[identity] = row
    write(OUT / 'universe.json', universe)
    selected = random.Random(SEED).sample(universe, 50)
    manifest = {'seed': SEED, 'algorithm': 'Python random.Random(seed).sample(universe, 50), without replacement', 'universe_n': len(universe), 'eligibility': 'All released test rows, coding then non-snv, source index ascending. No label, adapter exclusion, outcome, or reviewer filtering.', 'universe_sha256': hashlib.sha256((OUT / 'universe.json').read_bytes()).hexdigest(), 'sample': [{'blind_id': f'blind-{i:03d}', **item} for i, item in enumerate(selected)], 'retry_policy': 'At most one same-task infrastructure retry; no outcome retries or replacements.', 'conditional_solver': 'Run DeepSeek on the same random cohort only if scientific full-answer validity is established; do not form a reviewer-selected success subset.', 'rates': 'FP=accepted independently invalid answers/independently invalid answers; FN=rejected independently valid answers/independently valid answers. Unresolved evidence is neither valid nor invalid.', 'settings': {'model': 'openai/gpt-5.6', 'temperature': 1, 'max_iterations': 50, 'prompt_budget': 40}}
    write(OUT / 'manifest.json', manifest)
    for item in manifest['sample']:
        task = OUT / 'tasks' / item['blind_id']
        (task / 'environment/data').mkdir(parents=True)
        (task / 'tests').mkdir()
        row = rows[item['id']]
        case = {k: row[k] for k in ('question', 'reference_sequence')}
        case['variant_sequence'] = row['variant_sequence' if item['setting'] == 'coding' else 'mutated_sequence']
        write(task / 'environment/data/case.json', case)
        (task / 'environment/Dockerfile').write_text('FROM python:3.12-slim\nWORKDIR /app\nCOPY data /app/data\n')
        (task / 'instruction.md').write_text('''# Independent scientific adjudication
You have a working budget of 40 turns.
Assess the clinical variant task in /app/data/case.json independently. You have only the supplied question and DNA windows and Python standard library. Do not use external network resources, lookup benchmark source data or keys, tests, prior judgments, or solver outputs.
The task requires an exact full answer {"pathogenicity":"benign" or "pathogenic","diseases":[all associated disease names]}, with [] for benign. The original question may only ask pathogenicity, but the full disease scope is mandatory. Disease matching ignores capitalization, punctuation and underscores; it does not infer synonyms, and subtype words/numbers matter.
Determine what the supplied evidence scientifically supports. Distinguish a missing clinical evidence base from a provably wrong assertion; do not guess source labels, and do not equate a molecular effect or question phrasing with established clinical pathogenicity. Equally, do not assume evidence is insufficient without inspecting the actual case.
Write /app/audit.json with:
{"full_answer_supported":boolean,"pathogenicity_supported":boolean,"disease_set_supported":boolean,"assessment":"supported"|"insufficient_evidence"|"contradictory_evidence","valid_answers":[answer objects],"invalid_answers":[{"answer":answer object,"evidence":"why independently disproven"}],"evidence":"case-specific reasoning","missing_evidence":[strings],"confidence":"high"|"medium"|"low"}.
List independently defensible full answers and independently disproven full answers only when possible. Unsupported is NOT disproven. Empty lists are appropriate when validity cannot be adjudicated. This is scientific calibration, not an attempt to reproduce a hidden benchmark answer. Do not manufacture malformed answers as invalid clinical examples.
''')
        (task / 'task.toml').write_text('''schema_version = "1.0"
[task]
name = "bioreason-scientific-audit/'''+item['blind_id']+'''"
[verifier]
timeout_sec = 60.0
[agent]
timeout_sec = 1200.0
[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 4096
''')
        (task / 'tests/test.sh').write_text('''#!/bin/sh
mkdir -p /logs/verifier
python3 - <<'CHECK'
import json
from pathlib import Path
try:
    v=json.loads(Path('/app/audit.json').read_text())
    ok=all(type(v[k]) is bool for k in ('full_answer_supported','pathogenicity_supported','disease_set_supported')) and v['assessment'] in ('supported','insufficient_evidence','contradictory_evidence') and all(isinstance(v[k],list) for k in ('valid_answers','invalid_answers','missing_evidence')) and bool(v['evidence'])
except (OSError,ValueError,KeyError,TypeError):
    ok=False
Path('/logs/verifier/reward.txt').write_text(str(float(ok)))
CHECK
''')
        (task / 'tests/test.sh').chmod(0o755)
    config = {'job_name': 'bioreason-random50-gpt56-t1-i50', 'jobs_dir': str(OUT / 'jobs'), 'orchestrator': {'n_concurrent_trials': 10}, 'retry': {'max_retries': 1, 'include_exceptions': ['EnvironmentStartError', 'EnvironmentBuildError']}, 'environment': {'type': 'sail'}, 'agents': [{'name': 'openhands-sdk', 'model_name': 'openai/gpt-5.6', 'kwargs': {'load_skills': False, 'max_iterations': 50, 'temperature': 1}, 'env': {'LLM_API_KEY': '${LLM_API_KEY}', 'LLM_BASE_URL': 'https://llm-proxy.app.all-hands.dev'}}], 'datasets': [{'path': str(OUT / 'tasks')}], 'artifacts': ['/app/audit.json']}
    write(OUT / 'job-config.json', config)
    print('Frozen', len(universe), 'rows; sample', len(selected))


if __name__ == '__main__':
    main()
