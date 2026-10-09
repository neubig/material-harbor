"""Frozen balanced adaptation of the public Matbench Discovery unique subset."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABEL = 'e_above_hull_mp2020_corrected_ppd_mp'
PINS = {
    'wbm-summary.csv.gz': 'adbdd8b24086d4888195d8894c77f6d5fb29ce33bbb5d7d3898b4d79e964dc54',
    'wbm-init-structs.jsonl.gz': '98d545172c1ea9060f03f40cace6f8173a4ac06f1ee875ccd963765211519b58',
}


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def sanitize(structure: dict) -> dict:
    matrix = structure['lattice']['matrix']
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError('lattice shape')
    if not all(type(x) in (int, float) and math.isfinite(x) for row in matrix for x in row):
        raise ValueError('nonfinite lattice')
    a, b, c = matrix
    det = sum(a[i] * (b[(i+1)%3]*c[(i+2)%3]-b[(i+2)%3]*c[(i+1)%3]) for i in range(3))
    if abs(det) < 1e-10:
        raise ValueError('singular lattice')
    sites = []
    for site in structure['sites']:
        species, abc = site['species'], site['abc']
        if len(species) != 1 or species[0]['occu'] != 1:
            raise ValueError('not ordered unit occupancy')
        element = species[0]['element']
        if not isinstance(element, str) or not element.isalpha():
            raise ValueError('invalid element')
        if len(abc) != 3 or not all(type(x) in (int, float) and math.isfinite(x) for x in abc):
            raise ValueError('invalid coordinates')
        sites.append({'species': [{'element': element, 'occu': 1}], 'abc': abc, 'properties': {}})
    if not sites:
        raise ValueError('empty structure')
    return {'@module': 'pymatgen.core.structure', '@class': 'Structure', 'charge': 0,
            'lattice': {'matrix': matrix, 'pbc': [True, True, True]}, 'sites': sites}


def prepare(source: Path, destination: Path = ROOT / 'private') -> dict:
    if (destination / 'sample.json').exists():
        raise FileExistsError('Frozen sample exists; refusing to resample')
    for name, expected in PINS.items():
        if digest(source / name) != expected:
            raise ValueError(f'Source hash mismatch: {name}')
    destination.mkdir(parents=True, exist_ok=True)
    rows, excluded, summary_seen = {}, [], set()
    with gzip.open(source / 'wbm-summary.csv.gz', 'rt') as handle:
        for row in csv.DictReader(handle):
            key = row['material_id']
            if key in summary_seen:
                raise ValueError(f'Duplicate summary ID {key}')
            summary_seen.add(key)
            if row['unique_prototype'] != 'True':
                excluded.append({'material_id': key, 'reason': 'not_unique_prototype'})
                continue
            value = float(row[LABEL])
            if not math.isfinite(value):
                excluded.append({'material_id': key, 'reason': 'nonfinite_label'})
                continue
            rows[key] = value
    eligible, seen = [], set()
    with gzip.open(source / 'wbm-init-structs.jsonl.gz', 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            key = record['material_id']
            if key in seen:
                raise ValueError(f'Duplicate initial ID {key}')
            seen.add(key)
            if key not in rows:
                continue
            try:
                sanitize(record['initial_structure'])
            except (KeyError, TypeError, ValueError) as exc:
                excluded.append({'material_id': key, 'reason': f'invalid_structure: {exc}'})
                continue
            eligible.append(key)
    excluded.extend({'material_id': key, 'reason': 'missing_initial'} for key in rows.keys() - seen)
    eligible.sort()
    strata = {label: [key for key in eligible if (rows[key] <= 0) == label] for label in (True, False)}
    rng = random.Random(20261009)
    selected = rng.sample(strata[True], 50) + rng.sample(strata[False], 50)
    # Opaque IDs and shuffled order prevent class disclosure from stratum order.
    rng.shuffle(selected)
    sample = [{'task_id': f'mbd-{i:03d}', 'material_id': key} for i, key in enumerate(selected)]
    mapping = {row['material_id']: row['task_id'] for row in sample}
    (destination / 'inputs').mkdir()
    with gzip.open(source / 'wbm-init-structs.jsonl.gz', 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            if record['material_id'] in mapping:
                task_id = mapping[record['material_id']]
                (destination / 'inputs' / f'{task_id}.json').write_text(json.dumps(sanitize(record['initial_structure'])))
    gold = {mapping[key]: {'stable': rows[key] <= 0, 'distance': rows[key]} for key in selected}
    for filename, obj in [('sample.json', sample), ('gold.json', gold), ('eligible.json', eligible), ('exclusions.json', excluded)]:
        (destination / filename).write_text(json.dumps(obj, indent=2) + '\n')
    manifest = {'n_eligible': len(eligible), 'n_stable': len(strata[True]), 'n_unstable': len(strata[False]),
                'n_sampled': len(sample), 'seed': 20261009, 'shuffle_after_stratum_sampling': True,
                'protocol_sha256': digest(ROOT / 'protocol.json'), 'source_sha256': PINS,
                'sample_sha256': digest(destination / 'sample.json'), 'eligible_sha256': digest(destination / 'eligible.json'),
                'near_hull_abs_le_0_001_sample': sum(abs(rows[k]) <= .001 for k in selected),
                'exclusion_count': len(excluded)}
    (ROOT / 'sample-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source), indent=2))
