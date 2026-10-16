"""Assemble qualification evidence without inventing missing run results."""
import datetime
import json
from pathlib import Path

from adapter import ROOT, digest


def read(path, default=None):
    path = ROOT / path
    return json.loads(path.read_text()) if path.exists() else default


def report():
    audit = read('scientific-audit-comparison.json')
    summaries = {p.name: json.loads(p.read_text()) for p in (ROOT / 'runtime').glob('*-summary.json')}
    data = {
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'IMPLEMENTED_SCIENTIFICALLY_AUDITED_SAIL_BLOCKED_NOT_FULLY_QUALIFIED',
        'scope': 'Balanced finite Matbench Discovery adaptation, not published leaderboard or full-dataset evaluation',
        'protocol': read('protocol.json'), 'sample': read('sample-manifest.json'),
        'source_revision': '6ffd8070e0a7133a795a9bd4b42ec8dd9e543a46',
        'pins': {'harbor_sail': read('runtime/upstream.json'), 'checkpoint': read('science/checkpoint.json'),
                 'scientific_requirements_sha256': digest(ROOT/'science/requirements.lock') if (ROOT/'science/requirements.lock').exists() else None},
        'tests': {'log': 'tests.log', 'output': (ROOT/'tests.log').read_text() if (ROOT/'tests.log').exists() else None},
        'scientific_audit': audit,
        'container_pins': read('container-pins.json'),
        'verifier_container_smoke': read('verifier-container-smoke.json'),
        'container_validation': read('container-validation.json'),
        'task_revision': {'current': 2, 'history': 'task-revisions.json', 'reason': 'SDK venv relocation fix; frozen cohort unchanged; no revision2 Sail run'},
        'science_interface': read('science/interface.json'),
        'runtime_installed': read('runtime/installed.json'),
        'model_availability': read('runtime/model-availability.json'),
        'runtime_status': read('runtime/status.json'), 'science_status': read('science/status.json'),
        'sail': {'jobs': summaries, 'harbor_job_id': '04238341-d6a4-4b3e-a78e-6a5e3f47d6b7', 'observed_sailbox_ids': [], 'agent_run_count': 0, 'llm_cost_usd': 0, 'llm_cost_scope': 'No benchmark model calls; excludes orchestration agents.', 'all100_denominator': {'verified': 0, 'recorded_upload_errors': 4, 'missing_or_interrupted': 96}, 'accuracy': None, 'events_file': 'runtime/sail-events.jsonl', 'cost_usd': None,
                 'cost_note': 'Unknown until billing evidence; not zero. Per-agent token and cost records in job summaries.'},
        'errors': read('errors.json', []),
        'blocked_next_step': 'After user confirmation/service recovery, complete remaining runtime validation, refresh security review for task revision2, rerun same cohort with at most one infrastructure retry, then run model100. No replacement tasks.',
        'C:J': {
            'C': {'value': 'Yes: materials crystal stability prediction', 'status': 'green'},
            'D': {'value': '215488 eligible public unique-prototype structures;100 frozen balanced sample', 'status':'green'},
            'E': {'value': 'Pinned CHGNet0.3.0 CPU relaxation tested; already-corrected energies minus independently rebuilt MP DFT hull. Container compatibility rebuild pending.', 'status':'yellow'},
            'F': {'value': '100 Harbor tasks parse; separate strict verifier; oracle Sail upload503 blocks end-to-end qualification', 'status':'yellow'},
            'G': {'value': None, 'status':'yellow', 'note':'DeepSeek-v4.1-flash availability verified; temp1 max50/prompt40 prepared but not launched after repeated oracle upload503'},
            'H': {'value': audit['fp_rate'] if audit else None, 'status':'green' if audit and audit['qualified'] else 'yellow', 'note':'Scientific key-free paired invalid boolean FP; not natural agent-output distribution'},
            'I': {'value': audit['fn_rate'] if audit else None, 'status':'green' if audit and audit['qualified'] else 'yellow', 'note':'Scientific key-free paired valid boolean FN; not natural agent-output distribution'},
            'J': {'value':'Pending: scientific audit passes, Sail infrastructure blocks requested qualification; no intrinsic scientific impossibility proved', 'status':'yellow'}},
        'caveats': ['No workbook changes, commits, or pushes.', 'Licensing uncertainty excluded as requested.',
                    'Both constant baselines50%;10-70% band does not demonstrate scientific skill.',
                    'Unique prototypes are not independent samples; no independence-adjusted generalization claim.',
                    'Public structures can be memorized; network controls do not prove uncontaminated training.',
                    'Missing infrastructure outputs remain in100-task denominator; no outcome-dependent substitutions.',
                    'No intrinsic scientific blocker proven; incomplete validation is not impossibility.']}
    (ROOT/'research-matbench-harborization.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__ == '__main__':
    report()
