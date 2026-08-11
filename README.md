# MatQnA for Harbor

This repository adapts [MatQnA](https://arxiv.org/abs/2509.11335), a multimodal materials-characterization QA benchmark, to [Harbor](https://github.com/harbor-framework/harbor).

The source is [`richardhzgg/matQnA`](https://huggingface.co/datasets/richardhzgg/matQnA), licensed MIT. It contains 4,968 questions, 270 embedded images, and ten characterization categories. The source Parquet is not committed; the adapter downloads it and materializes tasks reproducibly.

## Generate

```bash
uv run --with pandas --with pyarrow python -m src.matqna.main --output-dir datasets/matqna --limit 10
# use --all for the complete benchmark, or --task-ids 0 1 2
```

Generated tasks contain the decoded image, instruction, oracle solution, and verifier.

## Publishing

The canonical Harbor workflow is `harbor dataset init`, `harbor add --scan`, `harbor sync`, then `harbor publish --public`; see [Harbor dataset publishing](https://harborframework.github.io/harbor/docs/datasets/publishing). A Git repository can also be run directly with Harbor using a `registry.json` manifest. For Hugging Face, publish the generated task tree or a packaged archive with Git LFS and include a data card, source citation, license, and generation command.

## Citation

```bibtex
@misc{weng2025matqna, title={MatQnA: A Benchmark Dataset for Multi-modal Large Language Models in Materials Characterization and Analysis}, year={2025}, eprint={2509.11335}, archivePrefix={arXiv}}
```
