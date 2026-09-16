# lab-bench-seqqa

FutureHouse LAB-Bench, pinned commit 998a8e0a40cf116c80e1b0e7a805ebb5fb9fa838. Six families, 238 validated records after two tied-ORF exclusions, 207 groups, fixed 80 representatives, seed 20260915. Original MCQ/refusal text is preserved.

## Reproduce

From repository root, use Python 3.13 (standard library only). Keep downloaded sources and generated output outside the repository. Public downloads require no credentials; large ProteinGym archives are not bundled.

```sh
python lab-bench-seqqa/fetch_sources.py --output /tmp/lab-bench-seqqa-sources
python lab-bench-seqqa/main.py --source /tmp/lab-bench-seqqa-sources --output /tmp/lab-bench-seqqa-frozen
SEQQA_SOURCE_DIR=/tmp/lab-bench-seqqa-sources python tests/test_labbench_seqqa.py -v
```

Use a new output path. Retrieval verifies pinned checksums, including existing files. Source-enabled tests regenerate in temporary directories and compare every task/curation artifact against expected-frozen-sha256.json (the evaluated reference). No research workspace or experiments directory is required. Without source environment variables, source-dependent SeqQA tests skip; ProteinGym still verifies all 57 task outputs from bundled curation.

## Evaluation boundaries

Gold stays in /tests and oracle solutions, never agent inputs. Use Harbor supporting separate-verifier artifact transfer; shared verification is not equivalent. The trusted no-network verifier rejects symlinks, nonregular files and oversized answers. Python/Biopython are available to agents. Evaluated prompts, gold, Dockerfiles and task settings regenerate byte-for-byte.

Harness: OpenHands SDK 1.47.0, openai/deepseek-v4-flash, temperature 0, skills false, max iterations 20, agent timeout 600 seconds, concurrency 4. Inference-host allowlisting is unchanged. Host DNS required an override during evaluation; resolve the allowed hostname for your environment rather than copying a stale IP, and document transport-only amendments. Host allowlisting does not restrict models: report observed harness scores and audit auxiliary calls, not enforced single-model accuracy. Tests make no model calls.

## License and attribution

CC-BY-SA-4.0 for source data and adaptation. Retain LICENSE, CITATION.cff, generated attribution/canary, and share adaptations under the same license.

Do not commit downloaded archives, generated tasks, credentials, raw jobs/logs or trajectories. Bundled gold-bearing curation is builder/verifier data only; never copy it into agent sandboxes.
