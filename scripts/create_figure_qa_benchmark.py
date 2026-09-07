#!/usr/bin/env python3
"""Create Materials Figure QA-style examples from selected arXiv papers."""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ATOM = "http://www.w3.org/2005/Atom"
ARXIV_API = "https://export.arxiv.org/api/query"
FIGURE = re.compile(r"\\begin\s*\{figure\*?\}(.*?)\\end\s*\{figure\*?\}", re.S)
LABEL = re.compile(r"\\label\s*\{([^}]+)\}")
CAPTION = re.compile(r"\\caption(?:\[[^]]*\])?\s*\{")
IMAGE = re.compile(r"\\(?:includegraphics(?:\[[^]]*\])?|input)\s*\{([^}]+)\}")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".pdf")


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    categories: list[str]
    published: str
    pdf_url: str


def normalize_id(value: str) -> str:
    value = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", value.strip())
    value = value.removesuffix(".pdf").removesuffix("/")
    return re.sub(r"v\d+$", "", value)


def clean_tex(text: str, limit: int = 12_000) -> str:
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    text = re.sub(r"\\(?:cite|citep|citet|footnote)\s*\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:textbf|textit|emph|mathrm|mathbf|operatorname)\s*", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^]]*\])?\s*", "", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()[:limit]


def balanced_argument(text: str, opening: int) -> str:
    depth = 1
    index = opening + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
        index += 1
    return ""


def parse_metadata_feed(data: bytes) -> dict[str, Paper]:
    papers = {}
    root = ET.fromstring(data)
    for entry in root.findall(f"{{{ATOM}}}entry"):
        paper_id = normalize_id(entry.findtext(f"{{{ATOM}}}id", "").rsplit("/", 1)[-1])
        links = entry.findall(f"{{{ATOM}}}link")
        papers[paper_id] = Paper(
            paper_id,
            re.sub(r"\s+", " ", entry.findtext(f"{{{ATOM}}}title", "")).strip(),
            re.sub(r"\s+", " ", entry.findtext(f"{{{ATOM}}}summary", "")).strip(),
            [node.attrib["term"] for node in entry.findall(f"{{{ATOM}}}category")],
            entry.findtext(f"{{{ATOM}}}published", ""),
            next((node.attrib["href"] for node in links if node.attrib.get("title") == "pdf"),
                 f"https://arxiv.org/pdf/{paper_id}"),
        )
    return papers


def fetch_arxiv(query: dict[str, str | int], timeout: int) -> dict[str, Paper]:
    request = urllib.request.Request(
        f"{ARXIV_API}?{urllib.parse.urlencode(query)}",
        headers={"User-Agent": "materials-figure-qa-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse_metadata_feed(response.read())


def fetch_metadata(ids: list[str], timeout: int) -> dict[str, Paper]:
    papers = fetch_arxiv({"id_list": ",".join(ids), "max_results": len(ids)}, timeout)
    missing = [paper_id for paper_id in ids if paper_id not in papers]
    if missing:
        raise RuntimeError(f"No metadata returned for: {', '.join(missing)}")
    return papers


def fetch_domain_metadata(
    domains: list[str], papers_per_domain: int, timeout: int, delay: float = 0
) -> dict[str, Paper]:
    papers = {}
    for index, domain in enumerate(domains):
        if not re.fullmatch(r"[A-Za-z0-9.-]+", domain):
            raise ValueError(f"Invalid arXiv domain: {domain}")
        if index:
            time.sleep(max(0, delay))
        papers.update(fetch_arxiv({
            "search_query": f"cat:{domain}",
            "start": 0,
            "max_results": papers_per_domain,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }, timeout))
    return papers


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe path in source archive: {member.name}")
        source.extractall(destination, filter="data")


def download_source(paper_id: str, corpus: Path, timeout: int, delay: float) -> Path:
    root = corpus / paper_id.replace("/", "_")
    extracted = root / "tex"
    if extracted.exists() and any(extracted.rglob("*.tex")):
        return extracted
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "source.tar"
    time.sleep(max(0, delay))
    request = urllib.request.Request(
        f"https://export.arxiv.org/e-print/{urllib.parse.quote(paper_id)}",
        headers={"User-Agent": "materials-figure-qa-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    safe_extract(archive, extracted)
    if not any(extracted.rglob("*.tex")):
        raise RuntimeError(f"No LaTeX source found for {paper_id}")
    return extracted


def resolve_image(raw: str, tex_file: Path, source: Path) -> Path | None:
    names = [Path(raw)] if Path(raw).suffix else [Path(raw + suffix) for suffix in IMAGE_EXTENSIONS]
    source = source.resolve()
    for name in names:
        for parent in (tex_file.parent, source):
            candidate = (parent / name).resolve()
            if candidate.is_file() and source in candidate.parents:
                return candidate
    return None


def figure_discussion(label: str, tex_documents: list[str]) -> str:
    marker = re.compile(rf"\\(?:ref|autoref|cref|Cref)\s*\{{{re.escape(label)}\}}")
    passages = []
    for document in tex_documents:
        prose = FIGURE.sub(" ", document)
        for match in marker.finditer(prose):
            start = max(0, match.start() - 2200)
            end = min(len(prose), match.end() + 2200)
            passage = clean_tex(prose[start:end], 4000)
            if passage and passage not in passages:
                passages.append(passage)
    return " ".join(passages)[:7000]


def extract_candidates(paper: Paper, source: Path) -> list[dict]:
    files = list(source.rglob("*.tex"))
    documents = [path.read_text(encoding="utf-8", errors="ignore") for path in files]
    candidates = []
    for tex_file, document in zip(files, documents):
        for match in FIGURE.finditer(document):
            body = match.group(1)
            labels = [found.group(1) for found in LABEL.finditer(body) if "fig" in found.group(1).lower()]
            caption_match = CAPTION.search(body)
            caption = clean_tex(
                balanced_argument(body, body.find("{", caption_match.start())), 4000
            ) if caption_match else ""
            images = [resolve_image(found.group(1).strip(), tex_file, source) for found in IMAGE.finditer(body)]
            images = list(dict.fromkeys(image for image in images if image))
            if not labels or not caption or not images:
                continue
            for label in labels:
                discussion = figure_discussion(label, documents)
                if not discussion:
                    continue
                candidates.append({
                    "candidate_id": f"{paper.arxiv_id}:{label}:{tex_file.name}",
                    "paper_id": paper.arxiv_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "published": paper.published,
                    "pdf_url": paper.pdf_url,
                    "figure_label": label,
                    "caption": caption,
                    "discussion": discussion,
                    "context_status": "found",
                    "figure_source": clean_tex(body),
                    "source_tex": str(tex_file),
                    "original_image_path": str(images[0]),
                })
    return candidates


def render_image(source: Path, destination: Path) -> Path | None:
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    output = destination.with_suffix(".png")
    if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        with Image.open(source) as image:
            image.convert("RGB").save(output)
    elif source.suffix.lower() == ".pdf":
        try:
            import pymupdf
        except ImportError as error:
            raise RuntimeError("Install PDF rendering support with: pip install PyMuPDF") from error
        document = pymupdf.open(source)
        try:
            document[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(output)
        finally:
            document.close()
    else:
        return None

    with Image.open(output) as image:
        width, height = image.size
        if max(width / height, height / width) > 4:
            output.unlink()
            return None
        if max(width, height) > 2048:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            image.convert("RGB").save(output, optimize=True)
    return output


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    snippet = text if text.startswith("{") else text[text.find("{") : text.rfind("}") + 1]
    try:
        return json.loads(snippet, strict=False)
    except json.JSONDecodeError:
        # Some VLMs emit unescaped LaTeX commands inside JSON strings.
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', snippet)
        return json.loads(repaired, strict=False)


def call_vlm(
    key: str, base_url: str, model: str, prompt: str, image: Path, timeout: int, *,
    judge: bool = False, parse_response: bool = True,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url(image)}},
        ]}],
        "max_tokens": 1800 if judge else 6000,
    }
    if judge:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                choices = json.load(response).get("choices", [])
            if not choices:
                raise RuntimeError("VLM returned no choices")
            content = choices[0]["message"].get("content", "")
            return parse_json(content) if parse_response else {"content": content.strip()}
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:1000]
            failure = RuntimeError(f"VLM HTTP {error.code}: {detail}")
        except (OSError, ValueError, RuntimeError) as error:
            failure = error
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise failure


def author_prompt(item: dict) -> str:
    return f"""You are an expert materials scientist curating a difficult multimodal QA benchmark.

Paper: {item['title']}
Abstract: {item['abstract'][:1800]}
Figure label: {item['figure_label']}
Caption: {item['caption'][:2500]}
Surrounding discussion: {item['discussion'][:5000]}

Inspect the supplied figure. Create ONE challenging, self-contained question requiring both visual inspection and materials-science domain knowledge. It must not be answerable from the caption or discussion alone. Focus on a trend, comparison, structure-property relation, mechanism, phase/defect/interface interpretation, or experimental limitation. Do not invent values, labels, panels, or entities. The answer must identify the necessary visual evidence.

Return ONLY JSON with keys: question, answer, reasoning, difficulty (hard or very_hard), quality_flags (array). Keep each string under 900 characters."""


def review_prompt(item: dict) -> str:
    return f"""You are a strict expert evaluator of a materials-science multimodal QA item. Inspect the figure yourself.

Question: {item['question']}
Reference answer: {item['answer']}
Reasoning: {item['reasoning']}
Caption: {item['caption']}
Surrounding discussion: {item['discussion'][:4500]}

Pass only if the question is unambiguous and self-contained, the image is necessary, the answer is visibly supported and scientifically valid, every cited label/value/panel exists, the item is not answerable from caption/discussion alone, and no outside paper-specific knowledge is needed. Return ONLY JSON with keys: pass (boolean), question_valid (boolean), answer_valid (boolean), image_required (boolean), rationale (under 500 chars), quality_flags (array)."""


CALIBRATION_MODELS = (
    "gpt-5.5", "claude-opus-4-6", "gemini-2.5-pro",
    "claude-sonnet-4-6", "gemini-3.5-flash",
)


def answer_prompt(item: dict) -> str:
    return f"""Answer this difficult materials-science multimodal question using the supplied figure and domain knowledge. If it is under-specified or unreadable, say so. Give a concise answer with the key evidence and reasoning.

Question: {item['question']}"""


def grade_prompt(item: dict, model_answer: str) -> str:
    return f"""Strictly judge this materials-science multimodal answer. Inspect the figure; do not blindly trust the reference.

Question: {item['question']}
Reference answer: {item['answer']}
Proposed answer: {model_answer}
Caption: {item['caption']}
Discussion: {item['discussion'][:3500]}

Return ONLY JSON with keys: score (0, 1, 2, or 3), reference_valid (boolean), question_valid (boolean), rationale (under 500 chars), error_type (none, visual_misread, domain_error, unsupported_claim, question_ambiguous, reference_error, or unreadable_image)."""


def calibrate(
    item: dict, image: Path, key: str, base_url: str, models: tuple[str, ...], timeout: int
) -> dict:
    results = []
    for model in models:
        answer = call_vlm(
            key, base_url, model, answer_prompt(item), image, timeout, parse_response=False
        ).get("content")
        if not answer:
            raise RuntimeError(f"Calibration model {model} returned no answer")
        verdict = call_vlm(
            key, base_url, "gpt-5.5", grade_prompt(item, answer), image, timeout, judge=True
        )
        results.append({"model": model, "answer": answer, "verdict": verdict})
    scores = [int(result["verdict"].get("score", 0)) for result in results]
    return {
        "models": list(models), "results": results,
        "valid": all(result["verdict"].get("question_valid") is True and result["verdict"].get("reference_valid") is True for result in results),
        "mean_score": sum(scores) / len(scores),
        "fully_correct": sum(score == 3 for score in scores),
    }


def select_diverse(items: list[dict], limit: int) -> list[dict]:
    by_paper: dict[str, list[dict]] = {}
    for item in sorted(items, key=lambda row: (row.get("published", ""), row["candidate_id"]), reverse=True):
        by_paper.setdefault(item["paper_id"], []).append(item)
    selected = []
    while by_paper and len(selected) < limit:
        for paper_id in list(by_paper):
            selected.append(by_paper[paper_id].pop(0))
            if not by_paper[paper_id]:
                del by_paper[paper_id]
            if len(selected) == limit:
                break
    return selected


def write_splits(items: list[dict], output_dir: Path, validation_fraction: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_ids = sorted({item["paper_id"] for item in items})
    validation_count = max(1, round(len(paper_ids) * validation_fraction)) if len(paper_ids) > 1 else 1
    validation_papers = set(paper_ids[:validation_count])
    for split in ("validation", "test"):
        rows = [item for item in items if (item["paper_id"] in validation_papers) == (split == "validation")]
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def ids_in(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.add(json.loads(line)["candidate_id"])
        except (json.JSONDecodeError, KeyError):
            pass
    return result


def values_from_file(path: Path | None) -> list[str]:
    if not path:
        return []
    return [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def resolve_papers(args: argparse.Namespace) -> tuple[list[str], dict[str, Paper]]:
    ids = [normalize_id(value) for value in args.paper + values_from_file(args.paper_file)]
    domains = list(dict.fromkeys(args.domain + values_from_file(args.domain_file)))
    ids = list(dict.fromkeys(ids))
    if not ids and not domains:
        raise ValueError("Provide --paper, --paper-file, --domain, or --domain-file")
    metadata = fetch_metadata(ids, args.timeout) if ids else {}
    if domains:
        metadata.update(fetch_domain_metadata(
            domains, args.papers_per_domain, args.timeout, args.delay
        ))
    return list(metadata), metadata


def build(args: argparse.Namespace) -> int:
    ids, metadata = resolve_papers(args)
    if not ids:
        raise RuntimeError("No papers returned for the requested arXiv domains")
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    if not key or not base_url:
        raise RuntimeError("Set LLM_API_KEY (or OPENAI_API_KEY) and LLM_BASE_URL")

    approved_path = args.output.with_name(args.output.stem + ".approved.jsonl")
    candidates_path = args.output.with_name(args.output.stem + ".candidates.jsonl")
    rejected_path = args.output.with_name(args.output.stem + ".rejected.jsonl")
    if args.overwrite:
        for path in (args.output, approved_path, candidates_path, rejected_path):
            path.unlink(missing_ok=True)
    completed = ids_in(approved_path) | ids_in(rejected_path)
    questions = set()
    if approved_path.exists():
        questions = {
            json.loads(line)["question"].strip().lower()
            for line in approved_path.read_text(encoding="utf-8").splitlines()
        }

    processed = 0
    for paper_id in ids:
        source = download_source(paper_id, args.work_dir / "corpus", args.timeout, args.delay)
        for item in extract_candidates(metadata[paper_id], source):
            if item["candidate_id"] in completed:
                continue
            if args.max_candidates is not None and processed >= args.max_candidates:
                break
            processed += 1
            filename = re.sub(r"[^A-Za-z0-9_.-]", "_", item["candidate_id"])
            image = render_image(Path(item["original_image_path"]), args.work_dir / "rendered" / filename)
            if not image:
                continue
            item["image_path"] = str(image)
            append_jsonl(candidates_path, item)
            try:
                qa = call_vlm(key, base_url, args.author_model, author_prompt(item), image, args.timeout)
                if not all(qa.get(field) for field in ("question", "answer", "reasoning", "difficulty")):
                    raise RuntimeError("Author omitted required fields")
                authored = {**item, **qa, "author_model": args.author_model}
                review = call_vlm(
                    key, base_url, args.judge_model, review_prompt(authored), image, args.timeout,
                    judge=True,
                )
                authored.update(review=review, judge_model=args.judge_model)
                question_key = authored["question"].strip().lower()
                review_fields = ("pass", "question_valid", "answer_valid", "image_required")
                accepted = all(review.get(field) is True for field in review_fields)
                rejection_reasons = []
                if not accepted:
                    rejection_reasons.append("strict_review")
                if question_key in questions:
                    accepted = False
                    rejection_reasons.append("duplicate_question")
                if accepted and not args.skip_calibration:
                    calibration = calibrate(
                        authored, image, key, base_url, CALIBRATION_MODELS, args.timeout
                    )
                    authored["calibration"] = calibration
                    if not calibration["valid"]:
                        accepted = False
                        rejection_reasons.append("invalid_question_or_reference")
                    if calibration["fully_correct"] > args.max_fully_correct:
                        accepted = False
                        rejection_reasons.append("insufficient_difficulty")
                authored["rejection_reasons"] = rejection_reasons
                destination = approved_path if accepted else rejected_path
                append_jsonl(destination, authored)
                completed.add(item["candidate_id"])
                if accepted:
                    questions.add(question_key)
                    print(f"approved {len(ids_in(approved_path))}: {item['candidate_id']}", flush=True)
                else:
                    print(f"rejected {item['candidate_id']}: {review.get('rationale', '')}", flush=True)
            except Exception as error:
                print(f"skip {item['candidate_id']}: {type(error).__name__}: {error}", file=sys.stderr)
            time.sleep(max(0, args.delay))
        if args.max_candidates is not None and processed >= args.max_candidates:
            break

    approved = [json.loads(line) for line in approved_path.open()] if approved_path.exists() else []
    selected = select_diverse(approved, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_splits(selected, args.splits_dir, args.validation_fraction)
    return len(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="append", default=[], help="arXiv ID or URL; repeatable")
    parser.add_argument("--paper-file", type=Path, help="one arXiv ID or URL per line")
    parser.add_argument("--domain", action="append", default=[], help="arXiv category such as cond-mat.mtrl-sci; repeatable")
    parser.add_argument("--domain-file", type=Path, help="one arXiv category per line")
    parser.add_argument("--papers-per-domain", type=int, default=100, help="newest papers fetched per arXiv category")
    parser.add_argument("--output", type=Path, default=Path("data/materials-figure-qa.jsonl"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/materials-figure-qa"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/work"))
    parser.add_argument("--limit", type=int, default=300, help="examples after diverse sampling")
    parser.add_argument("--max-candidates", type=int, help="bound authoring attempts for smoke tests")
    parser.add_argument("--author-model", default="gemini-3.5-flash")
    parser.add_argument("--judge-model", default="gpt-5.5")
    parser.add_argument("--skip-calibration", action="store_true", help="skip costly five-model calibration")
    parser.add_argument("--max-fully-correct", type=int, default=4, choices=range(6), help="reject items all calibration models solve")
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.papers_per_domain < 1:
        raise ValueError("--papers-per-domain must be positive")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    accepted = build(args)
    print(f"Dataset contains {accepted} accepted examples at {args.output}")
    return 0 if accepted >= args.limit else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
