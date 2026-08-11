# MatQnA for Harbor

This repository adapts [MatQnA](https://arxiv.org/abs/2509.11335), a multimodal materials-characterization QA benchmark, to [Harbor](https://github.com/harbor-framework/harbor).

The source is [`richardhzgg/matQnA`](https://huggingface.co/datasets/richardhzgg/matQnA), licensed MIT. It contains 4,968 questions, 270 embedded images, and ten characterization categories. The source Parquet is not committed; the adapter downloads it and materializes tasks reproducibly.

## Generate

```bash
uv run --with pandas --with pyarrow python -m src.matqna.main --output-dir datasets/matqna --limit 10
# use --all for the complete benchmark, or --task-ids 0 1 2
```

Generated tasks contain the decoded image, instruction, oracle solution, and verifier.

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

Trial results, rewards, logs, and trajectories are written below the directory supplied by `--trials-dir` (or Harbor's default trials directory). Objective tasks receive exact-choice scoring. Subjective-task scoring is currently provisional: the verifier checks only that the agent writes a non-empty answer, so subjective results should not be treated as benchmark-quality scores until an expert or LLM-judge rubric is added.

## Publishing

The canonical Harbor workflow is `harbor dataset init`, `harbor add --scan`, `harbor sync`, then `harbor publish --public`; see [Harbor dataset publishing](https://harborframework.github.io/harbor/docs/datasets/publishing). A Git repository can also be run directly with Harbor using a `registry.json` manifest. For Hugging Face, publish the generated task tree or a packaged archive with Git LFS and include a data card, source citation, license, and generation command.

## Citation

```bibtex
@misc{weng2025matqna, title={MatQnA: A Benchmark Dataset for Multi-modal Large Language Models in Materials Characterization and Analysis}, year={2025}, eprint={2509.11335}, archivePrefix={arXiv}}
```
