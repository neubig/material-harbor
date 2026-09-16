# proteingym-stability

ProteinGym v1.3, https://zenodo.org/records/15293562. Bundled curation-manifest.json contains 64 pre-outcome Tsuboyama pairs derived entirely from public metadata and measurements. One missing raw join and six raw/processed discrepancies are excluded, leaving 57 tasks. The old manifest.private.json filename meant withheld gold, not confidential data. No external manifest is needed. Raw uncertainty/sequence joins are revalidated. Transitive similarity grouping yields one component; it is not evidence of 57 independent families.

## Reproduce

From repository root, use Python 3.13 (standard library only). Keep downloaded sources and generated output outside the repository. Public downloads require no credentials; large ProteinGym archives are not bundled.

```sh
python proteingym-stability/fetch_sources.py --output /tmp/proteingym-stability-sources
python proteingym-stability/main.py --source-dir /tmp/proteingym-stability-sources --output-dir /tmp/proteingym-stability-frozen
PROTEINGYM_SOURCE_DIR=/tmp/proteingym-stability-sources python tests/test_proteingym_stability.py -v
```

Use a new output path. Retrieval verifies pinned checksums, including existing files. Source-enabled tests regenerate in temporary directories and compare every task/curation artifact against expected-frozen-sha256.json (the evaluated reference). No research workspace or experiments directory is required. Without the source environment variable, the source-regeneration test skips; the verifier test still checks all 57 task outputs from bundled curation.

## Evaluation boundaries

Gold stays in /tests and oracle solutions, never agent inputs. Use Harbor supporting separate-verifier artifact transfer; shared verification is not equivalent. The trusted no-network verifier rejects symlinks, nonregular files and oversized answers. Python/Biopython are available to agents. Evaluated prompts, gold, Dockerfiles and task settings regenerate byte-for-byte.

Harness: OpenHands SDK 1.47.0, openai/deepseek-v4-flash, temperature 0, skills false, max iterations 20, agent timeout 600 seconds, concurrency 2. Inference-host allowlisting is unchanged. Host DNS required an override during evaluation; resolve the allowed hostname for your environment rather than copying a stale IP, and document transport-only amendments. Host allowlisting does not restrict models: report observed harness scores and audit auxiliary calls, not enforced single-model accuracy. Tests make no model calls.

## License and attribution

MIT as declared by the pinned Zenodo record. LICENSE preserves ProteinGym copyright and applies to public-derived curation. Cite Notin et al., ProteinGym, and Tsuboyama et al. (2023), DOI 10.1038/s41586-023-06328-6. Provenance and checksums are bundled.

Do not commit downloaded archives, generated tasks, credentials, raw jobs/logs or trajectories. Bundled gold-bearing curation is builder/verifier data only; never copy it into agent sandboxes.
