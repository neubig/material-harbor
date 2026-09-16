"""Audit all pinned original test rows without exporting bulk sequences."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

SOURCES = {
    "coding": (
        "wanglab/variant_effect_coding",
        "7684ab820d728fb5362328757079a1390a3f43bb",
        1233,
    ),
    "non-snv": (
        "wanglab/variant_effect_non_snv",
        "aeba75d031b8cb507b3962ab486aa22700afa31c",
        873,
    ),
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def compare(ref, alt):
    distance = sum(a != b for a, b in zip(ref, alt))
    result = {
        "reference_length": len(ref),
        "alternate_length": len(alt),
        "hamming_distance": distance if len(ref) == len(alt) else None,
    }
    if ref == alt:
        return dict(result, kind="identical")
    p = next(
        (i for i, (a, b) in enumerate(zip(ref, alt)) if a != b), min(len(ref), len(alt))
    )
    result.update(
        first_difference_0based=p,
        reference_excerpt=ref[max(0, p - 12) : p + 28],
        alternate_excerpt=alt[max(0, p - 12) : p + 28],
    )
    result["kind"] = (
        "single_substitution"
        if len(ref) == len(alt) and distance == 1
        else "other_difference"
    )
    if result["kind"] == "other_difference":
        for size in range(1, 257):
            for kind, a, b in [
                ("insertion", ref[p:], alt[p + size :]),
                ("deletion", ref[p + size :], alt[p:]),
            ]:
                overlap = min(len(a), len(b))
                if overlap >= 64 and a[:overlap] == b[:overlap]:
                    return dict(
                        result,
                        kind="consistent_with_" + kind + "_and_boundary_change",
                        candidate_indel_length=size,
                        exact_downstream_overlap=overlap,
                    )
    return result


def context(q):
    chrom = re.search(r"chromosome\s+([0-9]+|X|Y)\b", q, re.IGNORECASE)
    pos = re.search(r"(?:position|location)\s+(\d+)", q, re.IGNORECASE)
    genes = re.findall(r"\b([A-Z][A-Za-z0-9.-]*(?:, [A-Z][A-Za-z0-9.-]*)*)\s*\(", q)
    genes = [
        g
        for g in genes
        if g not in {"DNA", "RNA", "Disease", "Clinical", "Chromosome", "Gene"}
    ]
    if not genes:
        genes = re.findall(
            r"\b(?:[Gg]ene|impacting|affecting|within|in)\s+([A-Z][A-Za-z0-9.-]*)\b", q
        )
    genes = [g for g in genes if g not in {"Chromosome", "Gene", "Disease", "Clinical"}]
    return (
        chrom.group(1) if chrom else None,
        pos.group(1) if pos else None,
        genes[0] if genes else None,
    )


def audit(setting, repo, revision, expected):
    data = load_dataset(repo, split="test", revision=revision)
    if len(data) != expected:
        raise ValueError(f"Unexpected row count: {len(data)}")
    key = "variant_sequence" if setting == "coding" else "mutated_sequence"
    counts, forms, chromosomes, genes, comparisons, lengths = (
        Counter() for _ in range(6)
    )
    flags, examples, inputs, pairs = (defaultdict(list) for _ in range(4))
    labels, answers = {}, {}
    gene_presence_labels = Counter()
    alphabet = Counter()
    h = hashlib.sha256()
    for i, row in enumerate(data):
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n")
        q, answer, ref, alt = (
            row["question"],
            row["answer"],
            row["reference_sequence"],
            row[key],
        )
        for field, value in row.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                flags["missing_or_empty_" + field].append(i)
        label, sep, disease = answer.partition(";")
        label, disease = label.strip().lower(), disease.strip()
        labels[i], answers[i] = label, answer
        counts[label] += 1
        missing = False
        if label not in {"benign", "pathogenic"}:
            flags["invalid_label"].append(i)
        if not sep:
            form = "bare_" + label
            missing = label == "pathogenic"
        elif setting == "coding":
            form = (
                "pathogenic_semicolon_text"
                if label == "pathogenic" and disease
                else "invalid"
            )
            missing = disease.lower() in {"not provided", "not specified", ""}
        else:
            try:
                parsed = ast.literal_eval(disease)
                valid = isinstance(parsed, list) and all(
                    isinstance(x, str) and x.strip() for x in parsed
                )
            except (SyntaxError, ValueError):
                valid = False
            form = (
                "pathogenic_semicolon_python_list"
                if valid and label == "pathogenic"
                else "invalid"
            )
            missing = valid and not parsed
        forms[form] += 1
        if form == "invalid":
            flags["invalid_answer_format"].append(i)
        if missing:
            flags["missing_pathogenic_disease_label"].append(i)
            counts["missing_disease_" + (disease.lower() or "bare_pathogenic")] += 1
        if "cleaned_pathogenicity" in row and row["cleaned_pathogenicity"] != label:
            flags["cleaned_pathogenicity_disagrees"].append(i)
        chrom, pos, gene = context(q)
        gene_presence_labels[
            ("gene_present/" if gene else "gene_absent_or_unparsed/") + label
        ] += 1
        chromosomes[chrom or "UNPARSED"] += 1
        genes[gene or "UNPARSED"] += 1
        for name, value in [
            ("chromosome", chrom),
            ("variant_position", pos),
            ("gene", gene),
        ]:
            if value is None:
                flags["missing_or_unparsed_" + name].append(i)
        for name, pattern in [
            ("assembly", r"\b(?:GRCh\d+|hg\d+|genome assembly|reference assembly)\b"),
            ("transcript", r"\b(?:[NX][MR]_\d+|ENST\d+|transcript)\b"),
        ]:
            if not re.search(pattern, q, re.IGNORECASE):
                flags["no_explicit_" + name + "_in_question"].append(i)
        if not re.search(
            r"benign|pathogenic|disease|clinical|biological|medical|harmful",
            q,
            re.IGNORECASE,
        ):
            flags["unrecognized_question_intent"].append(i)
        alphabet.update(ref + alt)
        bad = sorted(set(ref + alt) - set("ACGT"))
        if bad:
            flags["non_ACGT_sequence"].append(i)
        if set(ref + alt) - set("ACGTN"):
            flags["invalid_DNA_alphabet_outside_ACGTN"].append(i)
        if "N" in ref or "N" in alt:
            flags["ambiguous_N_bases"].append(i)
            if "N" not in ref and "N" not in alt.rstrip("N"):
                flags["alternate_trailing_N_only"].append(i)
        comp = compare(ref, alt)
        lengths[f"{len(ref)}/{len(alt)}"] += 1
        comparisons[comp["kind"]] += 1
        if comp["kind"] == "identical":
            flags["identical_sequences"].append(i)
        inputs[digest([q, ref, alt])].append(i)
        pairs[digest([ref, alt])].append(i)
        categories = [comp["kind"]]
        if missing:
            categories.append(
                "missing_disease_" + (disease.lower() or "bare_pathogenic")
            )
        if bad:
            categories.append("non_ACGT_sequence")
        if form == "invalid":
            categories.append("invalid_answer_format")
        for category in categories:
            if len(examples[category]) < 2:
                examples[category].append(
                    {
                        "test_index_0based": i,
                        "source_id": row.get("ID", row.get("__index_level_0__")),
                        "question": q,
                        "answer": answer,
                        "comparison": comp,
                    }
                )

    def duplicates(groups):
        dup = [v for v in groups.values() if len(v) > 1]
        return {
            "groups": len(dup),
            "rows": sum(map(len, dup)),
            "conflicting_full_answer_groups": [
                v for v in dup if len({answers[i] for i in v}) > 1
            ],
            "conflicting_binary_label_groups": [
                v for v in dup if len({labels[i] for i in v}) > 1
            ],
            "example_groups": dup[:5],
            "conflict_examples": [
                {"indices": v, "answers": [answers[i] for i in v]}
                for v in dup
                if len({answers[i] for i in v}) > 1
            ][:2],
        }

    return {
        "source": {
            "repository": repo,
            "revision": revision,
            "split": "test",
            "url": f"https://huggingface.co/datasets/{repo}/tree/{revision}",
            "card": f"https://huggingface.co/datasets/{repo}/blob/{revision}/README.md",
            "ordered_rows_sha256": h.hexdigest(),
            "fields": data.column_names,
        },
        "rows_audited": len(data),
        "answer_counts": dict(counts),
        "answer_formats": dict(forms),
        "sequence_length_pairs": dict(lengths),
        "sequence_comparisons": dict(comparisons),
        "flags": {
            k: {
                "count": len(v),
                "test_indices_0based": v if len(v) < len(data) else "all",
            }
            for k, v in sorted(flags.items())
        },
        "zero_count_checks": [
            k
            for k in [
                "invalid_label",
                "invalid_answer_format",
                "identical_sequences",
                "non_ACGT_sequence",
                "invalid_DNA_alphabet_outside_ACGTN",
                "cleaned_pathogenicity_disagrees",
            ]
            if not flags[k]
        ],
        "DNA_character_counts": dict(alphabet),
        "gene_presence_by_label": dict(gene_presence_labels),
        "chromosome_counts": dict(chromosomes.most_common()),
        "gene_context_concentration": {
            "unique_contexts": len(genes),
            "top_20": dict(genes.most_common(20)),
            "note": "Extracted contexts; comma-separated multi-gene contexts stay grouped. UNPARSED is explicit.",
        },
        "duplicates_identical_question_and_sequences": duplicates(inputs),
        "duplicates_sequence_pair_ignoring_question": duplicates(pairs),
        "representative_examples": dict(examples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("quality-report.json")
    )
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "scope": "All 2106 original test rows at pinned revisions; no clinical expert validation or external remapping.",
        "reproduce": "experiments/generation-venv/bin/python material-harbor/bioreason-vep/quality_audit.py",
        "methods": {
            "duplicates": "Exact question+ordered sequences; separately ordered sequence pairs ignoring question. Full-answer conflicts use raw strings, binary conflicts separately.",
            "sequences": "Exact equality, Hamming distance, first differing zero-based window offset. Search shifts 1..256 for exact downstream overlap >=64 bp through shorter tail. Candidate shifts are not normalized alleles, global alignment or functional consequences.",
            "context": "Source schema inventory and regex question extraction. No assembly inferred from provenance/coordinates. Flags absent from report have zero occurrences.",
            "missing_labels": "Bare pathogenic, empty disease lists, coding not provided/not specified mean missing disease supervision, not benign or no disease.",
        },
        "datasets": {s: audit(s, *spec) for s, spec in SOURCES.items()},
        "interpretation": {
            "objective_integrity": "Parsing, valid DNA and distinct sequences do not validate clinical labels or biological answerability.",
            "biological_answerability": "Window differences are observable, but pathogenicity and disease are not logically determined by DNA difference plus gene name. Assembly, transcript/strand/coding frame, normalized allele and clinical evidence are needed for defensible interpretation. Source-label prediction is possible; unique clinical answers are not established.",
            "source_field_remediation": {
                "coding": "Only ID, question, answer, reference_sequence and variant_sequence exist. ID is a task ID, not genomic accession. No additional position/assembly/transcript/disease field repairs missing context. Present diseases are already in answer.",
                "non-snv": "Question supplies position. Extra cleaned_pathogenicity is a binary target, not disease/context; __index_level_0__ is a row index without an exposed join table. Neither repairs disease/assembly/transcript. Present disease lists are already in answer.",
                "external_reconstruction": "Cards cite GPN-MSA/ClinVar/gnomAD for coding and ClinVar 2024-02-28 for non-SNV. Upstream mapping or genome alignment might recover context but requires validated assembly/strand/allele matching; not performed here.",
            },
            "observed_answerability_hazards": "Coding exact input duplicates can have different disease targets despite agreeing on binary class: disease exact-match is not uniquely specified. Coding missing gene context is perfectly associated with benign labels in this split, allowing a question-only shortcut; this is not biological reasoning. Non-SNV N bases are ambiguity symbols, not invalid DNA; terminal padding can obscure boundary comparisons. A malformed source disease list needs explicit handling, not silently fabricated repair.",
            "recommendation": "Do not endorse as clinically validated or uniformly answerable. Preserve binary targets; explicitly mark missing disease supervision and exclude from disease scoring without inventing diseases. Structured source-aligned answers alone do not repair biological context. Report concentration and duplicates separately.",
        },
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}: 2106 rows audited")


if __name__ == "__main__":
    main()
