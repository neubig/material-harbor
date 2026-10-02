# OmniMatBench image-bearing QA — metric protocol

Run: `omnimatbench-qa-vision-n10` (Harbor 0.22.0, deepseek-v4.1-flash, SDK `dad4aa5`,
judge gpt-5.1). 10 attempted image-bearing QA tasks, 140 eligible in the release.

## Primary metric — fixed budget, all attempts

| | value |
|---|---|
| Attempted trials | 10 |
| Unanswered counted as | 0 |
| **Primary mean reward** | **0.670** (6.7/10) |

Running out of the agent's own iteration budget is an end-to-end task failure under a
fixed budget, exactly like a wrong answer, so those trials score 0.

## Secondary metric — selection-conditioned

Mean over only the 7 trials that produced an answer: 0.957 (6.7/7, worst 0.8).
This is conditioned on answering and is **not** the benchmark result.

## The 3 unanswered trials

| trial | status | cause |
|---|---|---|
| 06-009 | missing_answer | no exception; 40-iteration budget exhausted (31,660 output tokens) — task failure |
| 13-006 | missing_answer | no exception; 40-iteration budget exhausted (48,756 output tokens) — task failure |
| 16-011 | missing_answer | `AgentTimeoutError` after 1800 s, agent metrics null — genuine harness exception |

Only 16-011 is an infrastructure fault. Its missingness is recorded separately rather than
pooled away or used to excuse the other two.

## Caveats

- The reward is weighted key-point coverage from a GPT-5.1 judge, because the release ships
  key points but no runnable QA scorer. It is a **proxy**, not binary accuracy, and cannot be
  compared against an accuracy band.
- Image transport is proven independently: all 10 staged figures matched byte-for-byte in the
  169 recorded requests, so the figures genuinely reached the model.
