"""Send independently adjudicated valid/invalid candidates through Sail verifier."""
import json
import shutil
from pathlib import Path

from prepare_runs import config
from report import calibration, ROOT

sample = json.loads((ROOT/'frozen/sample.json').read_text())
job = json.loads((ROOT/'jobs/moleculariq-scientific100/result.json').read_text())
if not job['finished_at']:
    raise RuntimeError('Persist all scientific judgments before candidate generation')
results = calibration(sample['uids'])
(ROOT/'calibration-frozen.json').write_text(json.dumps(results, indent=2)+'\n')
for kind in ('valid', 'invalid'):
    dest = ROOT / ('calibration-' + kind)
    dest.mkdir(exist_ok=False)
    for row in results['rows']:
        if row['status'] != 'adjudicated':
            continue
        uid = row['uid']
        task = dest / uid
        shutil.copytree(ROOT/'frozen/tasks'/uid, task)
        key = row['judgment']['key']
        answer = json.dumps({key: row[kind + '_candidate']})
        (task/'solution/solve.sh').write_text("#!/bin/sh\ncat > /app/answer.json <<'ANSWER'\n"+answer+'\nANSWER\n')
    run = config('moleculariq-calibration-' + kind, dest, {'name':'oracle'})
    (ROOT/(kind+'.json')).write_text(json.dumps(run,indent=2)+'\n')
print('Frozen', results['adjudicated'], 'valid and invalid candidate pairs;', results['unresolved'], 'unresolved rows retained')
