import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'tests/research-bioreason-verification'


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    job = OUT / 'jobs/bioreason-random50-gpt56-t1-i50'
    result = json.loads((job / 'result.json').read_text())
    trials = {}
    for path in job.glob('*/result.json'):
        data = json.loads(path.read_text())
        blind_id = path.parent.name.split('__')[0]
        trials[blind_id] = {'trial_id': data['id'], 'trial_name': data['trial_name'], 'exception': (data.get('exception_info') or {}).get('exception_type'), 'usage': data.get('agent_result'), 'result_path': str(path.relative_to(OUT))}
    outcomes = []
    for item in manifest['sample']:
        trial = trials.get(item['blind_id'])
        outcomes.append({**item, 'status': trial['exception'] if trial else 'not_started_after_budget_stop', 'trial': trial, 'adjudication': None})
    boxes = json.loads((OUT / 'sail-boxes-response.json').read_text())['data']
    names = {t['trial_name'] for t in trials.values()}
    fields = ('sailbox_id', 'name', 'status', 'image_id', 'created_at', 'updated_at')
    boxes = [{k: b.get(k) for k in fields} for b in boxes if any(b.get('name', '').startswith(n + '__') for n in names)]
    (OUT / 'sail-boxes-response.json').write_text(json.dumps({'data': boxes}, indent=2))
    rates = {name: {'estimate': None, 'numerator': None, 'denominator': 0, 'ci95': None, 'status': 'unmeasured'} for name in ('scientific_false_positive', 'scientific_false_negative', 'deepseek_accuracy')}
    cells = {
        'C6': {'value': 'Yes (genomics)', 'status': 'green'},
        'D6': {'value': '2,106 public source test rows; 1,727 retained by existing adapter', 'status': 'green', 'note': 'Availability only, not clinical qualification. Random50 sampled from all 2,106 without source-key filtering.'},
        'E6': {'value': 'Yes (Evo2-1B; Nucleotide Transformer-500M)', 'status': 'green', 'note': 'Prior source evidence; not tools in the audit environment.'},
        'F6': {'value': 'Yes (existing adapter; Sail containers launched)', 'status': 'green', 'note': 'Existing oracle 2/2 is historical plumbing evidence, not scientific validation.'},
        'G6': {'value': 'Unmeasured (clinical validity unresolved; GPT-5.6 budget blocked)', 'status': 'yellow'},
        'H6': {'value': 'Unmeasured (independent invalid-answer denominator 0)', 'status': 'yellow'},
        'I6': {'value': 'Unmeasured (independent valid-answer denominator 0)', 'status': 'yellow'},
        'J6': {'value': 'Pending (yellow): provider budget blocks scientific adjudication', 'status': 'yellow'},
    }
    report = {
        'status': 'blocked_requires_budget_restore_and_user_confirmation',
        'recommendation': 'yellow',
        'reason': 'HTTP 429 budget_exceeded from LLM proxy. Infrastructure failure is not measured scientific invalidity or verifier failure.',
        'sample': manifest,
        'source_revisions': {x['setting']: x['revision'] for x in manifest['sample']},
        'harbor': {'version': '0.24.0', 'source_revision': '4b94505a91c5ddcb70b5740ac95718ddee13e5a0', 'installed_from': 'local harbor-main', 'openhands_sdk_version_observed': '1.54.0'},
        'jobs': [{'harbor_job_id': result['id'], 'name': job.name, 'config': str((OUT / 'job-config.json').relative_to(ROOT.parent)), 'stats_raw': result['stats'], 'sail_boxes': boxes, 'stopped_with': 'SIGINT to exact Harbor PID 901972 after repeated budget_exceeded', 'retries': 0}],
        'cost': {'llm_usd': None, 'sail_usd': None, 'total_usd': None, 'input_tokens': None, 'output_tokens': None, 'cache_tokens': None, 'note': 'Usage and billing not reported by failed/cancelled trials; unknown is not zero. Proxy cumulative team budget figures are not this job cost.'},
        'random_adjudication_outcomes': {'sample_n': 50, 'recorded_scientific_assessments': 0, 'missing_assessments': 50, 'categories': dict(Counter(x['status'] for x in outcomes)), 'full_answer_supported_n': None, 'insufficient_clinical_evidence_n': None, 'contradictory_evidence_n': None, 'ci95': None, 'note': 'No scientific outcome distribution can be estimated from zero returned assessments. All sampled rows remain in the denominator.'},
        'rates': rates,
        'C:J': cells,
        'outcomes': outcomes,
        'blinding': {'uploaded_case_fields': ['question', 'reference_sequence', 'variant_sequence'], 'source_keys_uploaded': False, 'solver_outputs_uploaded': False, 'prior_judgments_uploaded': False, 'reconciliation_performed': False, 'note': 'Each independent Harbor trial receives only its own key-free case; no oracle solution in audit tasks. Network use prohibited in prompt, not claimed technically blocked.'},
        'prior_evidence': {'path': 'research-bioreason-continuation', 'use': 'Context and source revision discovery only; previous DeepSeek temperature0 tool-free assessments are not GPT-5.6 Harbor adjudications and were not merged into current rates.', 'adapter_exclusions': 'Existing 1,727-row retained population uses source-answer exclusions; current full-source draw avoids this filtering.'},
        'deepseek': {'run': False, 'reason': 'No randomly sampled scientifically validated cohort has been established; no reviewer-selected subset or historical label-only score substituted.'},
        'tests': {'passed': 2, 'log': str((OUT / 'tests.log').relative_to(ROOT.parent)), 'scope': 'Actual frozen random draw and actual key-free task environments/settings; no mocks.'},
        'next_step': 'Restore LLM proxy budget, obtain confirmation to resume these exact 50 cases with the same settings and no replacement. Reconcile only after blind assessments are frozen; run DeepSeek only if cohort validity established. Missing clinical evidence must remain unresolved rather than false-positive/false-negative.',
        'scope': {'writes': ['material-harbor/bioreason-vep/', 'material-harbor/tests/research-bioreason-verification/'], 'workbook_edited': False, 'committed': False, 'pushed': False},
        'artifact_hashes': {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT / 'manifest.json', OUT / 'universe.json', OUT / 'job-config.json']},
    }
    path = ROOT / 'research-bioreason-verification.json'
    path.write_text(json.dumps(report, indent=2))
    print(path)
    print(report['random_adjudication_outcomes'])
    print('Sail boxes:', Counter(b['status'] for b in boxes))


if __name__ == '__main__':
    main()
