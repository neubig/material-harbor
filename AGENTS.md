# Benchmark evaluation policy

- Report full-dataset accuracy only when every source task is evaluated.
- For estimated accuracy or verifier FP/FN, use a reproducible random sample of 50–100 examples; prefer 100 when cost permits.
- Freeze and describe the complete eligible source universe before sampling. Sample without replacement with a recorded seed and algorithm.
- Use simple random sampling by default. A stratified random sample is allowed only when the strata and allocation rule are declared before model outcomes; report weighting needed to recover the source distribution.
- Never select tasks using model success, apparent difficulty, answer key, reviewer agreement, or verifier behavior.
- Do not replace sampled tasks because they are ambiguous, malformed, timed out, or failed. Keep them in the denominator and report each failure category. Infrastructure-only retries must rerun the same task under a declared policy.
- Calculate representative verifier FP/FN from randomly sampled, independently adjudicated valid and invalid answers. Report valid and invalid denominators separately with confidence intervals. Keep hand-written adversarial tests separate from representative rates.
- Preserve source keys outside the agent environment. Auditors must not see source keys, solver outputs, or prior judgments before recording their assessment.
- Record the source revision, sample manifest, random seed, Harbor revision, agent/model settings, Sail job IDs, missing outputs, errors, retries, token usage, and cost.
- Qualification claims apply only to the exact task contract and population sampled. Clearly distinguish objective-only or other subset results from the full source benchmark.
