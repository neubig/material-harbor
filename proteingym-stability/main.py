"""Build the pre-outcome quality successor from pinned ProteinGym data."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = None
OUT = None
CONTEXT = ('Predict the higher thermodynamic unfolding stability in the Tsuboyama 2023 cDNA-display '
           'proteolysis assay, pH 7.4, 298 K. Trypsin and chymotrypsin concentration series are fitted '
           'jointly with a folded/unfolded kinetic model correcting sequence-dependent unfolded-state '
           'protease susceptibility. Higher unfolding deltaG (kcal/mol) means more stable. Within this '
           'WT domain, ProteinGym ddG_ML_float is mutant minus reference unfolding deltaG: higher is '
           'also more stable, unlike the opposite folding-ddG convention. Predict the recorded assay '
           'endpoint, not universal fitness or pathogenicity. Mutation positions are 1-based in the '
           'supplied WT domain. Pairs have large observed margins and disjoint model-derived 95% '
           'intervals; these quantify sequencing-count uncertainty, not all systematic uncertainty.')
ATTRIBUTION = ('ProteinGym v1.3, https://zenodo.org/records/15293562, MIT dataset license; Notin et al., '
               'ProteinGym. Tsuboyama et al. (2023), Mega-scale experimental analysis of protein folding '
               'stability in biology and design, https://doi.org/10.1038/s41586-023-06328-6. '
               'Adaptation: quality-filtered binary comparison, unchanged observed labels.\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rows(z, name):
    return list(csv.DictReader(io.TextIOWrapper(z.open(name))))


def select():
    selected, excluded = [], []
    with zipfile.ZipFile(SOURCE / 'substitutions_raw_DMS.zip') as raw, zipfile.ZipFile(SOURCE / 'DMS_ProteinGym_substitutions.zip') as processed:
        for task in json.loads((ROOT / 'curation-manifest.json').read_text()):
            p = task['public']; assay = p['DMS_id']
            if 'Tsuboyama_2023' not in assay:
                continue
            rr = rows(raw, 'substitutions_raw_DMS/' + assay + '.csv')
            pair = [[r for r in rr if r['mut_type'] == p['variant_' + s]] for s in 'AB']
            if any(len(r) != 1 for r in pair):
                excluded.append({'assay': assay, 'reason': 'missing_raw_join', 'counts': list(map(len, pair))}); continue
            pair = [r[0] for r in pair]
            if any(not math.isclose(float(r['ddG_ML_float']), task['gold']['score_' + s], rel_tol=0, abs_tol=1e-8) for s, r in zip('AB', pair)):
                excluded.append({'assay': assay, 'reason': 'raw_processed_discrepancy', 'raw': [r['ddG_ML_float'] for r in pair], 'processed': [task['gold']['score_'+s] for s in 'AB']}); continue
            pr = rows(processed, 'DMS_ProteinGym_substitutions/' + assay + '.csv')
            wt = p['wild_type_sequence']
            for side, r in zip('AB', pair):
                mutation = p['variant_' + side]
                old, pos, new = re.fullmatch(r'([A-Z])(\d+)([A-Z])', mutation).groups(); pos = int(pos)-1
                assert wt[pos] == old and old != new
                sequence = wt[:pos]+new+wt[pos+1:]
                hits = [v for v in pr if v['mutant'] == mutation]
                assert len(hits) == 1 and sequence == r['aa_seq'] == hits[0]['mutated_sequence']
                assert math.isclose(float(hits[0]['DMS_score']), task['gold']['score_'+side], abs_tol=1e-8)
                lo, dg, hi = [float(r[k]) for k in ('deltaG_95CI_low','deltaG','deltaG_95CI_high')]
                assert all(math.isfinite(x) for x in (lo,dg,hi)) and lo <= dg <= hi
                assert not any(c in r['dG_ML']+r['ddG_ML'] for c in '<>') and -1 <= dg <= 5
            expected = 'A' if float(pair[0]['deltaG']) > float(pair[1]['deltaG']) else 'B'
            high, low = pair if expected == 'A' else pair[::-1]
            assert float(low['deltaG_95CI_high']) < float(high['deltaG_95CI_low'])
            assert expected == task['gold']['answer']
            assert all(float(high['deltaG_'+e]) > float(low['deltaG_'+e]) for e in ('t','c'))
            offsets = [float(r['deltaG'])-float(r['ddG_ML_float']) for r in pair]
            assert math.isclose(*offsets, abs_tol=1e-8)
            assert pair[0]['WT_cluster'] == pair[1]['WT_cluster']
            task.update(raw_pair=pair, source_cluster=pair[0]['WT_cluster'])
            selected.append(task)
    assert len(selected) == 57 and len(excluded) == 7
    assert len({r['public']['UniProt_ID'] for r in selected}) == 57
    assert len({r['public']['wild_type_sequence'] for r in selected}) == 57
    return selected, excluded


def identity(a, b):
    # Global +2/-1/-2 alignment, ties favor more identical residue pairs.
    previous = [(-2*j,0,0) for j in range(len(b)+1)]
    for i,x in enumerate(a,1):
        current = [(-2*i,0,0)]
        for j,y in enumerate(b,1):
            d = previous[j-1]
            current.append(max((d[0]+(2 if x==y else -1),d[1]+(x==y),d[2]+1),
                               (previous[j][0]-2,previous[j][1],previous[j][2]),
                               (current[j-1][0]-2,current[j-1][1],current[j-1][2])))
        previous = current
    _, matches, aligned = previous[-1]
    return matches/aligned, aligned/max(len(a),len(b))


def cluster(records):
    parents = list(range(len(records))); comparisons = []
    def find(i):
        while parents[i] != i:
            i = parents[i]
        return i
    for i,a in enumerate(records):
        for j,b in enumerate(records[:i]):
            ident,cov = identity(a['public']['wild_type_sequence'],b['public']['wild_type_sequence'])
            comparisons.append({'a':i,'b':j,'identity':ident,'coverage':cov})
            if ident >= .3 and cov >= .8:
                parents[find(i)] = find(j)
    for i,r in enumerate(records):
        r['homology_cluster'] = find(i)
    return {'method':'global +2/-1/-2 alignment; >=30% identity over aligned residue pairs, >=80% longer-sequence coverage; transitive components; sensitivity grouping, not established families', 'comparisons':comparisons, 'n_clusters':len({find(i) for i in range(len(records))})}


def generate(record):
    task = OUT / 'tasks' / record['public']['task_id']
    for d in ('environment/data','tests','solution'):
        (task/d).mkdir(parents=True)
    (task/'environment/data/case.json').write_text(json.dumps({**record['public'],'assay_context':CONTEXT},indent=2))
    (task/'environment/data/ATTRIBUTION.txt').write_text(ATTRIBUTION)
    shutil.copyfile(ROOT/'LICENSE',task/'environment/data/LICENSE')
    (task/'environment/Dockerfile').write_text('FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea\nRUN apt-get update && apt-get install -y --no-install-recommends curl coreutils git && rm -rf /var/lib/apt/lists/*\nRUN pip install --no-cache-dir biopython==1.85 && python -m venv --system-site-packages /opt/openhands-sdk-venv && /opt/openhands-sdk-venv/bin/pip install --no-cache-dir openhands-sdk==1.47.0 openhands-tools==1.47.0 fastapi\nWORKDIR /app\nCOPY data /app/data\n')
    (task/'instruction.md').write_text('Read /app/data/case.json and predict which mutant A or B is more stable in the specified assay. Write exactly {"answer":"A"} or {"answer":"B"} to /logs/artifacts/answer.json, no extra keys or prose. Normal shell tools, Python and Biopython are available. Foundation model use is optional, not forbidden. Network is limited to the inference endpoint; do not seek public benchmark labels.\n')
    (task/'task.toml').write_text('schema_version = "1.0"\n[metadata]\ndataset = "ProteinGym v1.3 Tsuboyama quality successor"\nlicense = "MIT"\n[agent]\ntimeout_sec = 600.0\n[verifier]\ntimeout_sec = 60.0\nenvironment_mode = "separate"\n[verifier.environment]\nnetwork_mode = "no-network"\ncpus = 1\nmemory_mb = 512\n[environment]\nbuild_timeout_sec = 1200.0\ncpus = 1\nmemory_mb = 2048\nstorage_mb = 8192\nnetwork_mode = "allowlist"\nallowed_hosts = ["llm-proxy.app.all-hands.dev"]\n')
    (task/'tests/Dockerfile').write_text('FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea\nCOPY . /tests\n')
    shutil.copyfile(ROOT/'verify.py',task/'tests/verify.py')
    (task/'tests/gold.json').write_text(json.dumps(record['gold']))
    (task/'tests/test.sh').write_text('#!/bin/sh\nmkdir -p /logs/verifier\npython -I /tests/verify.py /logs/artifacts/answer.json /tests/gold.json /logs/verifier/reward.txt\n')
    (task/'solution/solve.sh').write_text('#!/bin/sh\nmkdir -p /logs/artifacts\nprintf \'%s\\n\' \''+json.dumps({'answer':record['gold']['answer']})+"' > /logs/artifacts/answer.json\n")
    for f in ('tests/test.sh','solution/solve.sh'):
        (task/f).chmod(0o755)


def main():
    global SOURCE, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    SOURCE, OUT = args.source_dir.resolve(), args.output_dir.resolve()
    expected = json.loads((ROOT / 'source-sha256.json').read_text())
    for name, digest in expected.items():
        if sha(((ROOT if name == 'curation-manifest.json' else SOURCE) / name).read_bytes()) != digest:
            raise ValueError(f'Pinned source checksum mismatch: {name}')
    if OUT.exists():
        raise ValueError('Refusing to overwrite frozen output')
    records,excluded = select(); clusters = cluster(records)
    OUT.mkdir(parents=True)
    for r in records:
        generate(r)
    for name,value in [('manifest.json',records),('exclusions.json',excluded),('sequence-similarity.json',clusters)]:
        (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
    hashes = {str(p.relative_to(OUT)):sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file()}
    (OUT/'frozen-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({'n':len(records),'labels':dict(collections.Counter(r['gold']['answer'] for r in records)),'clusters':clusters['n_clusters'],'manifest_sha256':sha((OUT/'manifest.json').read_bytes())}))


if __name__ == '__main__':
    main()
