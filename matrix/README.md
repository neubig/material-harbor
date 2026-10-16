# MATRIX — diagnostic-only, execution blocked

This adapter preserves original questions/context and canonical images and requires full scientific explanations/captions. It does not redefine MATRIX as technique classification. Licensing uncertainty is not used as a blocker.

## Frozen population

Source: `radical-ai/MATRIX@80b39472f8a22c4e47ec40b6a9b78c7077af01eb`, 470 test QAs (220 text, 250 vision). The diagnostic population is the 249 vision QAs excluding the single referenced validation/test exact-image overlap, qid `149788efb66846628901ea27d6d48208`. Paper-level provenance and transformed-image disjointness remain unresolved. The random-100 cohort is copied byte-for-byte from the continuation research; Python `random.Random(20261008).sample(qid_sorted_universe, 100)`. No task substitutions or outcome-based exclusions are allowed.

## Contract and trust boundary

Solver image contains only the question-facing image, not the answer or judge. Output is a regular UTF-8 `/logs/artifacts/answer.txt`, nonempty, no NUL bytes, maximum 65536 bytes. Harbor 0.24.0 `environment_mode = "separate"` gives the verifier a fresh environment; only collected artifacts cross from the solver. Tests, original question/reference, canonical image and their checksums are injected into this separate verifier. The judge never reads the agent's mutable image or question. The verifier credential is resolved only for the verifier phase. `python -I` prevents artifact Python-module shadowing. Checksums detect input corruption; they are not an independent trust root if an operator edits the whole trusted task directory.

Only numeric scores 0, .25, .5, .75, 1 plus a nonempty rationale are accepted. Strings, booleans, nonfinite values, duplicate/extra keys and markdown wrappers fail. Missing/malformed submissions receive 0 with a distinct status; API/schema/integrity failures produce no reward and an infrastructure-error record. Artifact transfer behavior for hostile symlinks still requires end-to-end Sail validation; local rejection alone does not prove transfer preserves file type.

## Substitute rubric

The complete versioned prompt is `verify.py:RUBRIC`. GPT-5.6 inspects the canonical image and original context, treats the reference as fallible evidence, and grades full scientific correctness, coverage, visual support and qualified interpretation. The five levels range from fundamentally incorrect (0), limited correct observations (.25), meaningful partial explanation (.5), substantially correct with a material omission (.75), to complete defensible response (1). Strict adapted binary correctness, if reported, means score exactly 1; native MATRIX reports ordinal scores and its GPT-5.1 rubric was not released. No native equivalence is claimed.

**Current blocker:** the initially frozen request uses `temperature=0`; the live proxy rejects this for reasoning-active GPT-5.6. No successful calibration or accuracy result exists. Do not use this adapter for rewards or training. Proposed recovery, pending user approval: change judge/reviewer temperature to 1, version that configuration, preflight successfully, and perform at most one same-task infrastructure retry with all original failures retained. Never change generated inputs during an active job.

## Calibration

`calibrate.py` freezes the protocol before inference. Stage one sees only the key-free prompt/image packet, records answerability and generates two full responses. Stage two independently reviews anonymous shuffled responses without keys, origin, intended-validity flags or previous judgments. Intended corruption is not adjudicated invalidity. Valid/invalid denominators and Wilson 95% intervals must be reported separately, retaining ambiguous and failed tasks. These synthetic responses do not estimate deployed solver-error distribution, paired responses are dependent, and same-family model review is not independent human ground truth. Human/expert or cross-model validation is required before qualification.

All 100 first-stage requests failed with HTTP 400; no second-stage adjudication or scientific FP/FN was performed. Handwritten schema/security tests are not representative scientific calibration.

## Commands

From the workspace root:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s material-harbor/matrix -p test_matrix.py
PYTHONDONTWRITEBYTECODE=1 python material-harbor/matrix/main.py
```

Generation refuses to overwrite existing tasks. Current tasks, failed job evidence, and the report are retained under this directory. Execution configs in `execution/` use latest released Harbor 0.24.0 and Sail 0.13.0, with DeepSeek-v4.1-flash temperature 1, maximum 50 iterations and prompt budget 40 turns. Latest upstream main was separately recorded, not installed or claimed executed. Costs absent from provider output are unknown, not zero. See `research-matrix-harborization.json` for exact run status.
