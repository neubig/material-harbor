import ast


def parse_source_answer(raw: str) -> dict:
    label, _, remainder = raw.partition(";")
    label = label.strip().casefold()
    if label not in {"benign", "pathogenic"}:
        raise ValueError(f"Unsupported source label: {raw!r}")
    remainder = remainder.strip()
    if label == "benign":
        if remainder:
            raise ValueError(f"Unexpected benign disease annotation: {raw!r}")
        diseases = []
    elif not remainder or remainder.casefold() in {"not provided", "not specified"}:
        diseases = None
    elif remainder.startswith("["):
        try:
            diseases = ast.literal_eval(remainder)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"Invalid disease list: {raw!r}") from error
        if (
            not isinstance(diseases, list)
            or not diseases
            or any(
                not isinstance(value, str) or not value.strip() for value in diseases
            )
        ):
            raise ValueError(f"Invalid disease list: {raw!r}")
    else:
        diseases = [remainder]
    return {"pathogenicity": label, "diseases": diseases}


def excluded_rows(rows, variant_key):
    exclusions = {}
    groups = {}
    for index, row in enumerate(rows):
        try:
            answer = parse_source_answer(row["answer"])
        except ValueError:
            exclusions[index] = "malformed source answer"
            continue
        if answer["diseases"] is None:
            exclusions[index] = "missing disease annotation"
            continue
        key = (row["reference_sequence"], row[variant_key])
        signature = (answer["pathogenicity"], tuple(sorted(answer["diseases"])))
        groups.setdefault(key, []).append((index, signature))
    for group in groups.values():
        if len({signature for _, signature in group}) > 1:
            for index, _ in group:
                exclusions[index] = (
                    "same sequence pair has conflicting full source answers"
                )
        else:
            for index, _ in group[1:]:
                exclusions[index] = "duplicate sequence pair and full source answer"
    return exclusions
