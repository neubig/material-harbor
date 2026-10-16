"""Compare previously frozen key-free scientific judgments with hidden grading."""
import json
import math
import tempfile
from pathlib import Path

from adapter import ROOT, digest
from verify import evaluate


def wilson(errors, total):
    if not total:
        return None
    z = 1.959963984540054
    p = errors / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [0.0 if errors == 0 else max(0, center-half), 1.0 if errors == total else min(1, center+half)]


def compare():
    audit_path = ROOT / 'science/audit.json'
    audit = json.loads(audit_path.read_text())
    if audit['state'] != 'frozen_before_gold_comparison':
        raise ValueError('Audit judgments not frozen')
    if digest(audit_path) != (ROOT / 'science/audit.sha256').read_text().split()[0]:
        raise ValueError('Audit changed after freeze')
    gold = json.loads((ROOT / 'private/gold.json').read_text())
    judgments = {r['task_id']: r for r in audit['records']}
    if len(judgments) != len(audit['records']) or set(judgments) != set(gold):
        raise ValueError('Incomplete or duplicate audit; no substitution allowed')
    rows, fp, fn = [], 0, 0
    with tempfile.TemporaryDirectory(dir=ROOT / 'private') as temporary:
        output = Path(temporary)
        for task_id, record in judgments.items():
            stable = record['stable']
            if type(stable) is not bool or not math.isfinite(record['signed_distance_ev_per_atom']):
                raise ValueError('Invalid scientific judgment')
            (output / 'answer.json').write_text(json.dumps({'stable': stable}))
            valid = evaluate(output, gold[task_id]['stable'])
            (output / 'answer.json').write_text(json.dumps({'stable': not stable}))
            invalid = evaluate(output, gold[task_id]['stable'])
            fn += 1-valid['reward']
            fp += invalid['reward']
            rows.append({'task_id': task_id, 'sign_agrees': stable == gold[task_id]['stable'],
                         'distance_delta_ev_per_atom': record['signed_distance_ev_per_atom']-gold[task_id]['distance'],
                         'valid_accepted': bool(valid['reward']), 'invalid_accepted': bool(invalid['reward'])})
    result = {'scope':'100 randomly sampled balanced structures, independent key-free DFT reconstruction; paired correct/opposite boolean answers, not natural agent-output error distribution',
              'audit_sha256':digest(audit_path), 'valid_denominator':len(rows), 'invalid_denominator':len(rows),
              'false_positives':fp, 'false_negatives':fn, 'fp_rate':fp/len(rows), 'fn_rate':fn/len(rows),
              'fp_wilson95':wilson(fp,len(rows)), 'fn_wilson95':wilson(fn,len(rows)),
              'ci_caveat':'Conditional descriptive binomial interval; chemistry/lineage correlation not accounted for.',
              'scientific_sign_disagreements':sum(not r['sign_agrees'] for r in rows),
              'max_absolute_distance_delta':max(abs(r['distance_delta_ev_per_atom']) for r in rows),
              'qualified':fp==fn==0, 'records':rows}
    (ROOT / 'scientific-audit-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    result = compare()
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
