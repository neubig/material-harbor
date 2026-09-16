"""Exact source-answer-independent validation of the six SeqQA families."""
import itertools
import re

FAMILIES = ('ORF-seq-AAseq', 'ORF-seq-numlen', 'PCR-len-primers',
            'PCR-primers-len', 'PCR-seq-primers', 'Prop-seq-gcpercent')
CODONS = dict(zip(map(''.join, itertools.product('TCAG', repeat=3)),
                  'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))


def reverse_complement(sequence):
    return sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def template(row, family):
    sequences = re.findall('[ACGT]{15,}', row['question'])
    return sequences[-1] if family.startswith('PCR') else sequences[0]


def proteins(sequence):
    result = []
    for start in range(len(sequence) - 2):
        if sequence[start:start + 3] != 'ATG':
            continue
        protein = []
        for offset in range(start, len(sequence) - 2, 3):
            aa = CODONS[sequence[offset:offset + 3]]
            protein.append(aa)
            if aa == '*':
                result.append(''.join(protein))
                break
    return result


def amplicons(sequence, pair):
    a, b = map(str.strip, pair.split(','))
    result = set()
    for forward, reverse in ((a, b), (b, a)):
        left = [m.start() for m in re.finditer('(?=' + forward + ')', sequence)]
        right = [m.start() for m in re.finditer('(?=' + reverse_complement(reverse) + ')', sequence)]
        for i in left:
            for j in right:
                if i <= j and i + len(forward) <= j + len(reverse):
                    result.add(sequence[i:j + len(reverse)])
    return result


def validate(row, family):
    sequence = template(row, family)
    options = [row['ideal'], *row['distractors']]
    if len(options) != len(set(options)):
        return {'valid': False, 'reason': 'duplicate_options'}
    if family == 'ORF-seq-AAseq':
        values = proteins(sequence)
        longest = [p for p in values if len(p) == max(map(len, values), default=0)]
        if len(longest) != 1:
            return {'valid': False, 'reason': 'tied_or_missing_longest_complete_forward_ORF'}
        computed = longest[0]
        matches = [i for i, option in enumerate(options) if option == computed]
    elif family == 'ORF-seq-numlen':
        threshold = int(re.search(r'greater than (\d+)', row['question'])[1])
        computed = str(sum(len(p) - 1 > threshold for p in proteins(sequence)))
        matches = [i for i, option in enumerate(options) if option == computed]
    elif family == 'Prop-seq-gcpercent':
        n = len(sequence)
        computed = str((200 * (sequence.count('G') + sequence.count('C')) + n) // (2 * n))
        matches = [i for i, option in enumerate(options) if option == computed]
    elif family == 'PCR-primers-len':
        primers = re.findall('[ACGT]{15,}', row['question'])[:2]
        values = sorted({len(p) for p in amplicons(sequence, ','.join(primers))})
        if len(values) != 1:
            return {'valid': False, 'reason': 'ambiguous_or_missing_amplicon_length'}
        computed = str(values[0])
        matches = [i for i, option in enumerate(options) if option == computed]
    elif family in ('PCR-len-primers', 'PCR-seq-primers'):
        target = (int(re.search(r'(\d+) bp', row['question'])[1]) if family == 'PCR-len-primers'
                  else re.findall('[ACGT]{15,}', row['question'])[0])
        matches = [i for i, option in enumerate(options)
                   if any((len(p) if isinstance(target, int) else p) == target
                          for p in amplicons(sequence, option))]
        computed = matches
    else:
        raise ValueError(f'Unsupported family: {family}')
    return {'valid': matches == [0], 'reason': None if matches == [0] else 'gold_mismatch_or_distractor_collision',
            'computed': computed, 'correct_option_indices': matches}
