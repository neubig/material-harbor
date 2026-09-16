import argparse
import collections
import hashlib
import json
import random
import shutil
from pathlib import Path

from validation import FAMILIES, reverse_complement, template, validate

ROOT = Path(__file__).resolve().parent
REVISION = '998a8e0a40cf116c80e1b0e7a805ebb5fb9fa838'
REFUSAL = 'Insufficient information to answer the question'
CONVENTIONS = {
    'ORF': "Use the supplied forward strand (5′ to 3′), all three forward reading frames, and the standard genetic code. Consider ATG starts only, each separately including nested starts, and end at the first in-frame TAA/TAG/TGA. Exclude incomplete ORFs. The stop is not an amino acid for length comparisons; 'greater than' is strict. Protein-sequence options include the terminal '*' stop marker. Do not search the reverse-complement strand.",
    'PCR': "Use an idealized linear DNA template and exact complementary binding, with all primer sequences written 5′ to 3′. Primers face inward on opposite strands. Amplicon lengths include both primers. Evaluate sequence compatibility only, without mismatch, thermodynamic, or experimental-efficiency modeling.",
    'Prop': "GC percentage is 100 times (G + C) divided by the sequence length, rounded to the nearest integer.",
}
ATTRIBUTION = ('Source: FutureHouse, LAB-Bench: Measuring Capabilities of Language Models for Biology Research. '
               'https://github.com/Future-House/LAB-Bench/tree/' + REVISION + '\n'
               'Source data and this adaptation: CC-BY-SA-4.0, https://creativecommons.org/licenses/by-sa/4.0/. '
               'Changes: deterministic option shuffle, explicit computational conventions, strict JSON file output, '
               'and independently validated subset selection. Original question and option text preserved.\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def choices(row):
    options = [row['ideal'], REFUSAL, *row['distractors']]
    permutation = list(range(len(options)))
    random.Random('lab-bench-seqqa-mcq-v1:' + row['id']).shuffle(permutation)
    return {chr(65 + i): options[j] for i, j in enumerate(permutation)}, chr(65 + permutation.index(0))


def load_candidates(source):
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    eligible, excluded, audit = [], [], []
    expected_files = {name for name in provenance['files_sha256'] if name.startswith('SeqQA/') and name.endswith('-public.jsonl')}
    actual_files = {'SeqQA/' + p.name for p in (source / 'SeqQA').glob('*-public.jsonl')}
    if actual_files != expected_files:
        raise ValueError('Pinned SeqQA source file set is incomplete or unexpected')
    for file in sorted((source / 'SeqQA').glob('*-public.jsonl')):
        relative = 'SeqQA/' + file.name
        if digest(file.read_bytes()) != provenance['files_sha256'][relative]:
            raise ValueError(f'Source hash mismatch: {file}')
        family = file.name.removesuffix('-v1-public.jsonl')
        for line in file.read_text().splitlines():
            row = json.loads(line)
            if family not in FAMILIES:
                excluded.append({'id': row['id'], 'family': family, 'reason': 'outside_predeclared_six_family_scope'})
                continue
            result = validate(row, family)
            audit.append({'id': row['id'], 'family': family, **result})
            if not result['valid']:
                excluded.append({'id': row['id'], 'family': family, 'reason': result['reason']})
                continue
            sequence = template(row, family)
            eligible.append({'id': row['id'], 'family': family, 'row': row,
                             'sequence': sequence,
                             'sequence_sha256': digest(min(sequence, reverse_complement(sequence)).encode())})
    if len(audit) != 240 or len({r['id'] for r in eligible}) != len(eligible):
        raise ValueError('Unexpected dataset size or duplicate IDs')
    return eligible, excluded, audit


def group_candidates(candidates):
    # Shared templates, reverse complements, or contained templates link records transitively.
    parents = list(range(len(candidates)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, a in enumerate(candidates):
        for j in range(i):
            b = candidates[j]
            short, long = sorted((a['sequence'], b['sequence']), key=len)
            if short in long or reverse_complement(short) in long:
                parents[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for i, row in enumerate(candidates):
        groups[find(i)].append(row)
    for members in groups.values():
        group = digest('\n'.join(sorted(r['id'] for r in members)).encode())
        for row in members:
            row['group_id'] = group
    return list(groups.values())


def select_representatives(candidates, size=80, seed=20260915):
    groups = group_candidates(candidates)
    representatives = [min(members, key=lambda r: digest(f'{seed}:representative:{r["id"]}'.encode()))
                       for members in groups]
    per_family = {family: sorted((r for r in representatives if r['family'] == family), key=lambda r: r['id'])
                  for family in FAMILIES}
    total = len(representatives)
    quotas = {family: size * len(rows) // total for family, rows in per_family.items()}
    remainder = size - sum(quotas.values())
    order = sorted(FAMILIES, key=lambda family: (-(size * len(per_family[family]) % total), family))
    for family in order[:remainder]:
        quotas[family] += 1
    rng = random.Random(seed)
    selected = []
    for family in FAMILIES:
        selected.extend(rng.sample(per_family[family], quotas[family]))
    if len(selected) != size or len({r['group_id'] for r in selected}) != size:
        raise ValueError('Not enough distinct sequence groups')
    return sorted(selected, key=lambda r: (r['family'], r['id'])), groups, quotas


def generate_task(record, output):
    row, family = record['row'], record['family']
    task = output / ('seqqa-' + row['id'])
    if task.exists():
        raise ValueError(f'Task already exists: {task}')
    for directory in ('environment/data', 'tests', 'solution'):
        (task / directory).mkdir(parents=True)
    options, answer = choices(row)
    case = {'question': row['question'], 'options': options,
            'conventions': CONVENTIONS[family.split('-')[0]]}
    (task / 'environment/data/case.json').write_text(json.dumps(case, ensure_ascii=False))
    (task / 'environment/data/ATTRIBUTION.txt').write_text(ATTRIBUTION + '\n' + row['canary'])
    shutil.copyfile(ROOT / 'LICENSE', task / 'environment/data/LICENSE')
    instruction = ("Answer the original multiple-choice biology question in /app/data/case.json. "
                   "The case contains the unchanged question and shuffled options, plus explicit computational conventions. "
                   "Choose one option; for a multi-component option all components must be correct. "
                   "Write /logs/artifacts/answer.json containing exactly {\"answer\": \"<option letter>\"}, using one uppercase letter "
                   "from the listed options. No other keys or prose in that file. "
                   "Python and Biopython are available; local computation and tools are permitted. "
                   "Use only the supplied case and local computation, not external benchmark answers. "
                   "Network access is limited to the model inference endpoint.\n")
    (task / 'instruction.md').write_text(instruction)
    gold = {'answer': answer, 'letters': list(options), 'id': row['id'], 'family': family,
            'source_revision': REVISION, 'source_row': row, 'group_id': record['group_id']}
    (task / 'tests/gold.json').write_text(json.dumps(gold))
    (task / 'solution/solve.sh').write_text("#!/bin/sh\nset -eu\nmkdir -p /logs/artifacts\ncat > /logs/artifacts/answer.json <<'ANSWER'\n" + json.dumps({'answer': answer}) + "\nANSWER\n")
    for relative in ('environment/Dockerfile', 'tests/test.sh', 'tests/verify.py', 'tests/Dockerfile', 'task.toml'):
        shutil.copyfile(ROOT / 'task-template' / relative, task / relative)
    for relative in ('tests/test.sh', 'solution/solve.sh'):
        (task / relative).chmod(0o755)
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-size', type=int, default=80)
    parser.add_argument('--seed', type=int, default=20260915)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output must be empty to preserve frozen manifests')
    candidates, exclusions, audit = load_candidates(args.source)
    sample, groups, quotas = select_representatives(candidates, args.sample_size, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for record in sample:
        generate_task(record, args.output / 'tasks')
    fields = ('id', 'family', 'group_id', 'sequence_sha256')
    manifest = {'source_revision': REVISION, 'seed': args.seed, 'eligible_count': len(candidates),
                'sequence_groups': len(groups), 'sample_size': len(sample), 'quotas': quotas,
                'grouping': 'transitive exact/RC/containment of full question templates; no outcome inputs',
                'representative_rule': 'minimum SHA256(seed:representative:id) per group',
                'sampling': 'proportional largest-remainder family allocation; sorted IDs; random.Random(seed).sample',
                'groups': [{'group_id': group[0]['group_id'], 'members': [r['id'] for r in group]} for group in groups],
                'selected': [{k: record[k] for k in fields} for record in sample]}
    for name, data in [('manifest.json', manifest), ('exclusions.json', exclusions), ('validation.json', audit)]:
        (args.output / name).write_text(json.dumps(data, indent=2) + '\n')
    hashes = {str(p.relative_to(args.output)): digest(p.read_bytes())
              for p in sorted(args.output.rglob('*')) if p.is_file()}
    (args.output / 'frozen-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('groups', 'selected')}, indent=2))


if __name__ == '__main__':
    main()
