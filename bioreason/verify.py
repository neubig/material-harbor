import json
import re
from pathlib import Path

GO_RE = re.compile(r"^GO:\d{7}$")
ASPECT_ROOTS = {"bp": "GO:0008150", "mf": "GO:0003674", "cc": "GO:0005575"}


def load_ontology(path):
    parents = {}
    aliases = {}
    obsolete = set()
    current = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if line == "[Term]":
            current = None
        elif line.startswith("id: GO:"):
            current = line.split("id: ", 1)[1]
            parents.setdefault(current, set())
        elif current and line.startswith("alt_id: GO:"):
            aliases[line.split("alt_id: ", 1)[1]] = current
        elif current and line.startswith("is_obsolete: true"):
            obsolete.add(current)
        elif current and line.startswith("is_a: GO:"):
            parents[current].add(line.split()[1])
        elif current and line.startswith("relationship: part_of GO:"):
            parents[current].add(line.split()[2])
    return parents, aliases, obsolete


def normalize_terms(terms, parents, aliases, obsolete):
    normalized = set()
    stack = []
    for term in terms:
        if not isinstance(term, str) or not GO_RE.fullmatch(term):
            raise ValueError("Every entry must be a canonical GO:NNNNNNN string")
        term = aliases.get(term, term)
        if term in obsolete or term not in parents:
            raise ValueError(f"Unknown or obsolete GO term: {term}")
        stack.append(term)
    while stack:
        term = stack.pop()
        if term in normalized:
            continue
        normalized.add(term)
        stack.extend(parents[term])
    return normalized


def grade(answer, gold, ontology, ontology_data=None):
    parents, aliases, obsolete = ontology_data or load_ontology(ontology)
    try:
        value = json.loads(answer) if isinstance(answer, str) else answer
        if not isinstance(value, dict) or set(value) != {"go_ids"} or not isinstance(value["go_ids"], list):
            raise ValueError('Expected exactly {"go_ids": [...]}')
        predicted = normalize_terms(value["go_ids"], parents, aliases, obsolete)
        if not predicted:
            raise ValueError("Empty prediction")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"reward": 0, "status": "invalid_answer", "reason": str(exc)}
    expected_by_aspect = {aspect: set(terms) for aspect, terms in gold["aspects"].items() if terms}
    matches = {}
    for aspect, expected in expected_by_aspect.items():
        root = ASPECT_ROOTS[aspect]
        predicted_aspect = {term for term in predicted if root in normalize_terms([term], parents, aliases, obsolete)}
        matches[aspect] = predicted_aspect == expected
    return {"reward": int(bool(matches) and all(matches.values())), "status": "scored",
            "aspect_matches": matches, "predicted_count": len(predicted)}


def main():
    answer_path = Path("/logs/artifacts/answer.json")
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    try:
        answer = answer_path.read_text() if answer_path.exists() else ""
        result = grade(answer, json.loads(Path("/tests/gold.json").read_text()), "/tests/go-basic.obo")
    except Exception as exc:
        result = {"reward": None, "status": "infrastructure_error", "error_type": type(exc).__name__}
    result["protocol"] = "bioreason-go-complete-v1"
    (logs / "result.json").write_text(json.dumps(result, indent=2))
    if result["reward"] is None:
        raise SystemExit(2)
    (logs / "reward.txt").write_text(str(result["reward"]))


if __name__ == "__main__":
    main()
