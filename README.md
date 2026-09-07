# Scientific benchmarks for Harbor

This repository adapts scientific benchmark datasets to the [Harbor](https://github.com/harbor-framework/harbor) task format.

## Datasets

### MatQnA

The existing MatQnA adapter is under `matqna`. It downloads the MIT-licensed [`richardhzgg/matQnA`](https://huggingface.co/datasets/richardhzgg/matQnA) Parquet source and generates multimodal materials-characterization tasks.

```bash
uv run --with pandas --with pyarrow python -m matqna.main \
  --output-dir datasets/matqna --limit 10
```

Use `--all` for the full source or `--task-ids 0 1 2` for selected rows.

### SciAgentGYM

The SciAgentGYM adapter is under `src/benchmarks/scieagentgym`. It downloads the public [`CMarsRover/SciAgentGYM`](https://github.com/CMarsRover/SciAgentGYM) repository, converts the 83 multi-question benchmark cases into Harbor tasks, and preserves each case's question, metadata, expected tool concepts, and answer in the generated task.

```bash
uv run python -m src.benchmarks.scieagentgym \
  --output-dir datasets/scieagentgym --limit 10
```

By default, only the 83 multi-step cases are generated. Add `--include-single` to include the 48 single-question cases as well. Use `--all` for the complete selected source or `--task-ids 0 1 2` for selected cases. Use `--source /path/to/SciAgentGYM` to avoid downloading the source.

Generated task directories contain `task.toml`, `instruction.md`, an isolated Docker environment, the source case at `/app/data/case.json`, and a deterministic verifier. The included oracle solution is useful for Harbor smoke tests; agent evaluations should solve the problem and write `/app/answer.txt`.

## Run an agent evaluation

Install Harbor and ensure Docker is running. Generate a local subset first, then run one task with an agent and model:

```bash
cd /path/to/harbor

uv run harbor trial start \
  -p /path/to/material-harbor/datasets/matqna/matqna-000004 \
  -a openhands-sdk \
  -m <provider>/<model> \
  -e docker \
  --ae LLM_API_KEY="$LLM_API_KEY" \
  --agent-kwarg max_iterations=20 \
  --agent-timeout 900 \
  --trials-dir /tmp/matqna-trials
```

The exact API environment variables depend on the selected model provider. `LLM_API_KEY` and any provider-specific base URL can be passed with repeated `--ae KEY=VALUE` flags. The `openai/` prefix is required when routing an OpenAI-compatible endpoint through LiteLLM.

Run the oracle smoke test to validate task infrastructure without an LLM:

```bash
uv run harbor trial start \
  -p /path/to/material-harbor/datasets/matqna/matqna-000004 \
  -a oracle -e docker \
  --trials-dir /tmp/matqna-oracle
```

To run a generated local dataset, use Harbor's dataset command with a concurrency setting appropriate for your Docker host:

```bash
uv run harbor run \
  -p /path/to/material-harbor/datasets/matqna \
  -a openhands-sdk \
  -m <provider>/<model> \
  -e docker \
  --n-concurrent 4 \
  --ae LLM_API_KEY="$LLM_API_KEY"
```

Trial results, rewards, logs, and trajectories are written below the directory supplied by `--trials-dir` (or Harbor's default trials directory). The original MatQnA adapter retains its existing objective/subjective scoring behavior; the Materials Figure QA adapter below uses a strict multimodal VLM judge.

## Materials Figure QA adapter

This repository also contains `src/materials_figure_qa`, an adapter for the filtered Materials Figure QA dataset at [gneubig/materials-figure-qa](https://huggingface.co/datasets/gneubig/materials-figure-qa). It generates Harbor tasks from the `validation` and `test` Parquet splits:

```bash
uv run --with datasets --with pillow python -m src.materials_figure_qa.main \
  --output-dir datasets/materials-figure-qa --split both
```

Run an agent task with Harbor as usual:

```bash
uv run harbor trial start \
  -p /path/to/material-harbor/datasets/materials-figure-qa/materials-figure-qa-validation-000000 \
  -a openhands-sdk -m openai/gpt-5.5 -e docker \
  --ae LLM_API_KEY="$LLM_API_KEY" \
  --trials-dir /tmp/materials-figure-qa-trials
```

The verifier uses a **strict multimodal VLM judge**, not lexical matching or non-empty-answer checks. It sends the figure, question, reference answer, and agent answer to an OpenAI-compatible endpoint. Configure the judge with:

```bash
--ae VLM_JUDGE_API_KEY="$LLM_API_KEY" \
--ae VLM_JUDGE_BASE_URL="https://llm-proxy.app.all-hands.dev/v1" \
--ae VLM_JUDGE_MODEL="gpt-5.5"
```

The verifier gives reward `1` only when the judge finds the answer substantively correct and supported by the figure; otherwise it gives `0`. It writes the judge rationale to `/logs/verifier/details.json`.

## Publishing

The canonical Harbor workflow is `harbor dataset init`, `harbor add --scan`, `harbor sync`, then `harbor publish --public`; see [Harbor dataset publishing](https://harborframework.github.io/harbor/docs/datasets/publishing). A Git repository can also be run directly with Harbor using a `registry.json` manifest. For Hugging Face, publish the generated task tree or a packaged archive with Git LFS and include a data card, source citation, license, and generation command.

## Citation

```bibtex
@misc{weng2025matqna, title={MatQnA: A Benchmark Dataset for Multi-modal Large Language Models in Materials Characterization and Analysis}, year={2025}, eprint={2509.11335}, archivePrefix={arXiv}}
```

## Materials Figure QA

The Materials Figure QA adapter is under `materials-figure-qa`. It generates tasks from the filtered `gneubig/materials-figure-qa` dataset and uses strict multimodal VLM grading.

```bash
uv run python materials-figure-qa/main.py --output-dir datasets/materials-figure-qa --split both
```

## Creating new figure-QA benchmarks

`scripts/create_figure_qa_benchmark.py` generalizes the Materials Figure QA
creation pipeline to user-supplied arXiv papers or categories:

```bash
export LLM_API_KEY=...
export LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
uv run python scripts/create_figure_qa_benchmark.py \
  --paper 2608.19185 \
  --paper 2608.19178 \
  --limit 300
```

To discover papers from one or more arXiv categories, repeat `--domain` or use
`--domain-file`. The newest 100 papers per category are fetched by default;
use `--papers-per-domain` to change that bound:

```bash
uv run python scripts/create_figure_qa_benchmark.py \
  --domain cond-mat.mtrl-sci \
  --domain cond-mat.soft \
  --papers-per-domain 50 \
  --limit 300
```

Explicit papers and categories can be combined, and duplicate papers are
processed only once. The run is resumable and retains candidate, approved,
rejected, calibration, and provenance data. Use `--overwrite` for a clean rebuild.
Five-model calibration is enabled by default; `--skip-calibration` is intended
only for cheap smoke tests.

### Quality controls

The builder includes every quality-control stage recovered from the original
Materials Figure QA creation process:

- requires a resolvable LaTeX figure, caption, and paper discussion that
  references the figure;
- requires the author VLM to use both visual evidence and materials-science
  knowledge, forbidding invented labels, panels, values, and entities;
- uses a separate strict VLM review for ambiguity, question validity, reference
  validity, visual necessity, visible support, and caption-only answerability;
- rejects duplicate questions;
- rejects figures with an aspect ratio greater than 4:1;
- downsamples figures to a maximum dimension of 2048 pixels;
- runs the five original calibration models and semantically grades each answer
  on the original 0–3 scale with error classification;
- rejects candidates that any semantic grader marks as an invalid question or
  reference, and by default rejects questions all five models solve fully;
- samples round-robin across papers for source diversity; and
- emits validation and test splits that are paper-disjoint to prevent leakage.

The generated `*.approved.jsonl`, `*.rejected.jsonl`, and
`*.candidates.jsonl` files provide an audit trail. The final JSONL and split
files include the full structured review and calibration records.
