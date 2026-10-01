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

### BioReason variant effect prediction

The BioReason adapter covers the two difficult variant-effect classification settings from [BioReason](https://arxiv.org/abs/2505.23579): coding variants and coding non-SNVs. It converts the public [`wanglab/variant_effect_coding`](https://huggingface.co/datasets/wanglab/variant_effect_coding) and [`wanglab/variant_effect_non_snv`](https://huggingface.co/datasets/wanglab/variant_effect_non_snv) test splits into structured pathogenicity-and-disease tasks. Original questions and full raw answers are retained; source revisions are pinned.

```bash
uv run python bioreason-vep/main.py \
  --output-dir datasets/bioreason-vep --setting both --limit 10 \
  --max-iterations 20
```

Use `--all` to inspect all 1,233 coding and 873 non-SNV source test examples. The quality filter currently retains **890 coding and 837 non-SNV tasks**. Missing disease annotations, malformed answers, and sequence pairs with conflicting complete answers are excluded and recorded in `excluded-<setting>.json`. Use `--setting coding` or `--setting non-snv` for one benchmark, or `--task-ids 4 676` for selected rows. `--limit` applies to source rows before filtering, so fewer tasks may be emitted. `--max-iterations` defaults to 20 and states the available agent-iteration budget in each generated prompt. Each task asks the agent to inspect `/app/data/case.json` and write `/app/answer.json`:

```json
{"pathogenicity": "pathogenic", "diseases": ["Disease name"]}
```

For benign variants, `diseases` must be `[]`. The disease list must match the complete source answer. The deterministic verifier normalizes case, punctuation, whitespace and underscores, but does not infer medical synonyms or merge disease subtypes. This is a reproducible **source-label exact-set metric**, not expert semantic adjudication.

Following [Harbor reward conventions](https://www.harborframework.com/docs/tasks), `/logs/verifier/reward.json` contains numeric `format`, `pathogenicity`, `diseases`, and `reward` metrics. **`reward` is 1 only if all three criteria pass**; partial criterion success never makes the binary reward true. A matching `reward.txt` supports scalar-only consumers. `details.json` records grading diagnostics. Use a fresh output directory when migrating from the old label-only adapter (or explicitly `--overwrite` the same selection).

### BioReason data quality and answerability

The reproducible full-source audit is in `bioreason-vep/quality_audit.py` and `bioreason-vep/quality-report.json`:

```bash
uv run python bioreason-vep/quality_audit.py --output bioreason-vep/quality-report.json
```

The audit covers all 2,106 source test rows. Coding has 41 missing disease annotations and 302 rows excluded for conflicting full answers on identical sequence pairs; non-SNV has 35 missing disease annotations and one malformed list. No disease labels are invented to repair these rows. The coding source contains exact question-and-sequence duplicates with different disease answers, and all coding test cases are on chromosome 8. Importantly, all 557 benign coding questions lack parsed gene context whereas pathogenic questions have gene context: this is a substantial shortcut risk, not evidence of biological reasoning.

**The remaining tasks are not certified as uniformly answerable or clinically valid.** All sequence pairs differ, but DNA windows and gene names do not uniquely establish clinical pathogenicity/disease. Genome assembly, transcript/strand and clinical evidence are generally absent. Filtering repairs objective grading contradictions, not missing biological evidence. Exact disease-name scoring can also reject legitimate synonyms. Clinical expert review and validated allele/context reconstruction are required before scientific endorsement.

### Historical label-only results (not the new full-answer metric)

The old DeepSeek-v4-flash/OpenHands-SDK Harbor runs reported 316/1,233 coding (25.63%) and 422/873 non-SNV (48.34%) with only two iterations; 885/2,106 answers were missing ([PR #3](https://github.com/neubig/material-harbor/pull/3)).

A subsequent 20-iteration, 600-second pilot on the first 50 non-SNV rows scored **28/50 = 56% label-only accuracy**, 60% balanced accuracy, versus a 60% majority-class raw baseline. Benign recall was 16/20; pathogenic recall 12/30. Sixteen agent timeouts were retained and their saved answers graded (7/16 correct). There were no final setup failures after retrying Docker-network failures/cancellations; successful results and true agent timeouts were not rerun. Recorded final-trial model cost was $1.27807624, excluding any unrecorded interrupted-attempt cost. The sample was not randomized. See `bioreason-vep/evaluation-results.json`.

**Neither historical result evaluates disease answers, the filtered subset, or the new all-criteria success metric.** New oracle tests validate plumbing only; no new-model accuracy is claimed.

Run the benchmark through Harbor with its standard OpenHands SDK agent:

```bash
harbor run \
  -p datasets/bioreason-vep \
  -a openhands-sdk \
  -m openai/deepseek-v4-flash \
  -e docker \
  --ae LLM_API_KEY="$LLM_API_KEY" \
  --ae LLM_BASE_URL="https://llm-proxy.app.all-hands.dev" \
  --agent-kwarg load_skills=false \
  --agent-kwarg max_iterations=20 \
  --agent-kwarg temperature=0 \
  --n-concurrent 8 --max-retries 2 -y
```

When overriding the iteration budget, pass the same value to the generator's `--max-iterations` option and Harbor's `--agent-kwarg max_iterations=...` option so the prompt matches the enforced limit.

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

## Fixed-manifest Modal runner

`scripts/run_modal.py` ports science-agent's built-in Harbor Modal backend setup,
using its Harbor pin `c178c20710c362ef806c5d5d18852f95b21ca34b`, without HPC
packages or credential-bearing command logging. Requires **Python 3.12+**:

```bash
uv sync --extra modal --python 3.12
```

Supply any local Harbor task and an explicit JSON manifest of reviewed files,
relative to that task, e.g.:

```json
["task.toml", "instruction.md", "environment/Dockerfile",
 "environment/data/case.json", "solution/solve.sh", "tests/test.sh",
 "tests/verify.py", "tests/data/info.json"]
```

Save the manifest outside the task. Review its files, Dockerfile, and task
configuration before upload. The manifest is a source boundary, not a detector
of secrets embedded in ordinary files or a sandbox for malicious task definitions.
Only listed files are staged. Hidden/credential-like filenames, symlinks, traversal,
and files outside standard task source directories are rejected. Never construct
a manifest from a recursive listing of your workspace/home.

```bash
uv run --extra modal python scripts/run_modal.py \
  --task /path/to/task --manifest /path/to/manifest.json \
  --output /tmp/modal-oracle --dry-run

uv run --extra modal python scripts/run_modal.py \
  --task /path/to/task --manifest /path/to/manifest.json \
  --output /tmp/modal-oracle --env-file "$HOME/.env" --agent oracle

uv run --extra modal python scripts/run_modal.py \
  --task /path/to/task --manifest /path/to/manifest.json \
  --output /tmp/modal-deepseek --env-file "$HOME/.env" \
  --agent openhands-sdk --model openai/deepseek-v4-flash \
  --base-url https://llm-proxy.app.all-hands.dev/v1 \
  --max-iterations 20 --timeout 180 --sandbox-timeout 900
```

Each real run needs a **fresh output directory** and defaults to one trial without retries.
After smoke validation and budget approval, `--repetitions N` runs independent
sequential trials against the same staged source snapshot. SHA-256 hashes are saved
in `source-hashes.json`; each trial has its own name, logs, command, and result.
`batch-summary.json` retains raw rewards and failures, not an accuracy estimate.
Infrastructure failures stop the batch immediately, preserving attempted trials;
zero-reward completed trials do not stop the batch. No retries or resume are implicit.
Outputs include the staged task, manifest, command, redacted console log, and
Harbor's `trials/smoke/result.json`, verifier logs, and agent trajectory.
Infrastructure exceptions return nonzero even when Harbor reports completion;
a completed trial with zero reward is not a runner failure. Keep outputs private
and outside version control. The sandbox lifetime/agent limits are not a monetary
budget or a total build-time limit. Harbor requests deletion on completion;
confirm cleanup in Modal after interruption.

Dotenv is parsed locally, never executed or copied. Only `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` are selected from the credential file, and both are required
(no ambient token/profile fallback). `LLM_API_KEY` is read only from the exported
environment; a dotenv LLM key is ignored.
The default agent is Harbor's native **OpenHands SDK** (`openhands-sdk`), with
skills disabled, temperature 0, and 20 iterations by default. `~/.env` is the
default credential file. The oracle receives no LLM key. Harbor forwards
`LLM_API_KEY` and `LLM_BASE_URL` to the installed SDK process in the sandbox;
the key is not passed in CLI arguments or embedded in the staged task. This
does not isolate credentials from agent-executed code. Modal tokens stay on the
host. Model calls send prompt/task data to your endpoint.
Tasks requiring verifier credentials or nonstandard source files need a separately
reviewed integration; arbitrary environment variables, mounts, skills, and agent
options are not forwarded. Your Modal identity must have write access to its
selected environment. `--modal-environment NAME` explicitly selects an authorized
environment via `MODAL_ENVIRONMENT`; there is no automatic fallback. Do not switch
environments or change permissions without approval. A successful environment
listing alone does not establish write permission. Resolve permission errors
with your workspace administrator.

```bash
uv run --extra modal python -m unittest discover -s tests -v
```

An oracle validates infrastructure only. A one-task model smoke is neither
benchmark accuracy nor scientific qualification.

## Reproducible local setup and offline tests

Adapter tooling supports Python 3.11+; Harbor/Modal requires Python 3.12+.
Install the locked dependencies from this directory:

```bash
uv sync --locked --extra test --extra analysis --python 3.12
uv run --locked --extra test python -m unittest discover -s tests -p 'test_run_modal.py' -v
uv run --locked --extra test python -m unittest discover -s tests -p 'test_run_omni_cohort.py' -v
uv run --locked --extra test python -m unittest discover -s tests -p 'test_summarize_accuracy.py' -v
uv run --locked --extra test python -m unittest discover -s tests -p 'test_omnimatbench.py' -v
```

These tests run offline after dependency installation. Omni native verifier tests
need no downloaded data; cohort integration tests explicitly skip until the fetch
below is run. Harbor schema validation skips when Harbor is not installed.
The `analysis` extra declares NumPy for `scripts/summarize_accuracy.py` and the
cohort runner. MATCHA image decoding dependencies are available with `--extra matcha`.
The historical cohort-runner tests still require local measurement fixtures; do not
copy private logs into a checkout to satisfy them. Keep generated tasks, source downloads, images, logs, trajectories,
measurement artifacts, handoffs, caches, and credentials out of Git. Use reviewed
file paths rather than `git add .`; ignore rules are not a secret scanner.

## OmniMatBench CAL text-only adapter

The adapter uses [Wanhao Liu and contributors' OmniMatBench](https://github.com/wanhaoliu/OmniMatBench)
at revision `f933f03c733bb378ce5dd85b96452d9d5683c17e`. The two unmodified native
comparator/path-helper modules are redistributed under the upstream MIT license
in `omnimatbench/source/LICENSE`. The generator includes that license alongside
the copied native code. Source data and supplementary upstream files are fetched
explicitly, never during tests or import:

```bash
uv run --locked python -m omnimatbench.fetch_source
uv run --locked python -m omnimatbench.main --output-dir datasets/omnimatbench-pilot --count 10 --seed 20260814
uv run --locked --extra test python -m unittest discover -s tests -p 'test_omnimatbench.py' -v
```

The fetch uses the pinned raw GitHub revision and checks every file against
`omnimatbench/source/sha256.json`; existing matching files are reused and mismatched
downloads are rejected before writing. Generation revalidates all hashes, refuses
existing output directories, and records the source hashes, selection and exclusions.
Use `--count 360` for the complete eligible cohort: 360 text-only questions out of
502 CAL records (142 image-bearing records excluded), **not all of OmniMatBench**.

The agent sees only the original question and answer format in `/app/case.json`.
It must write a JSON list to `/app/answer.json`. Every flattened answer slot must
pass the native exact comparator. Its rounding follows the reference string's
decimal places; this is not physical-unit conversion, symbolic equivalence or
scientific tolerance. Scientific notation has known quirks. One retained reference,
`cal/11/018`, triggers native `decimal.InvalidOperation` even when given its own
gold answer; count this as unknown/native failure, not a valid incorrect answer.
The tests preserve this behavior rather than silently changing the benchmark.
Public reference answers and unrestricted network access pose contamination risks.

Run generated tasks using the fixed-manifest runner above. Include
`tests/native/LICENSE` as well as both native Python files, verifier, gold, task
metadata, instructions, environment files and oracle script in the reviewed upload
manifest. Gold and oracle files must stay outside the environment build context.

### Measurement scope and limitations

The recorded fixed-100 DeepSeek-v4-flash/OpenHands SDK pilot had **14 correct,
82 wrong, and 4 agent timeouts**, with timeouts retained as unknown. The raw
success fraction is 14/100; it is **not full-benchmark accuracy**. Missingness-aware
95% bootstrap sensitivity spans approximately **8%–26%**, so qualification against
a strict 10%–70% band remains **inconclusive**. This measurement concerns only the
frozen 100-task selection, with 20 iterations and a 600-second agent timeout.
Native scoring failures, incomplete trials and infrastructure failures must not be
dropped from denominators. No logs or trajectories are published with this code.

`scripts/summarize_accuracy.py --help` describes the explicit frozen-manifest
analysis interface. `scripts/run_omni_cohort.py` is a measurement-specific
orchestrator, **not a fresh-checkout one-command benchmark**: its `freeze100` and
`run100` modes require the original local protocol/task/measurement artifacts,
fixed environment configuration and handoff. Those artifacts are deliberately not
published. For new measurements, generate a new selection and use `run_modal.py`
with an explicitly reviewed manifest and authorized Modal environment; do not
implicitly resume or overwrite the historical run.

## Additional adapters (not yet measured)

- CSMBench: `python csmbench/main.py` generates a caption-matching visual MCQ pilot with pinned metadata and image checks. Real-source tests require the pilot plus the pinned upstream `utils.py` audit fixture.
- MATCHA: install the `matcha` and `analysis` extras, then run `python matcha/adapter.py --selection pilot10`, `--selection full`, or `--selection sample100`. The full local audit compared all 1,500 generated images to native upstream decoding and cropping. These are mechanical checks, not scientific label validation.
- MATRIX: `python matrix/build.py --fetch --output datasets/matrix-run/tasks` retrieves checksum-pinned source and generates text tasks. Its binary reference judge is explicitly experimental, not the official five-level GPT-5.1 protocol; verifier-only credentials and compatible network setup must be supplied before measurement.

Visual model measurements remain blocked on validated SDK image transport. A completed CSMBench model smoke is not a valid visual accuracy estimate: tool-image serialization was not established. Do not launch cohorts or infer vision accuracy merely because an image path is accessible to the agent.
