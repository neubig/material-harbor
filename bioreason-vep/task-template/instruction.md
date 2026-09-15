# BioReason variant-effect prediction ({{ setting }})

Answer the original question in `/app/data/case.json` using its reference DNA sequence,
variant DNA sequence, and genomic context. Predict both pathogenicity and the associated
disease(s), retaining the full scope of the source dataset answer even when the question
only explicitly asks for pathogenicity.

Write `/app/answer.json` as one JSON object with exactly these keys:
- `"pathogenicity"`: `"benign"` or `"pathogenic"`.
- `"diseases"`: a list of disease names; use `[]` for benign variants.

For example: `{"pathogenicity": "pathogenic", "diseases": ["Disease name"]}`.
Include all associated diseases, with no extra diseases. Disease order, capitalization,
punctuation and underscores versus spaces are ignored; subtype numbers and words are
significant. Unlisted synonyms or abbreviations are not automatically accepted.
Classification and disease-set correctness are scored separately. Overall success
requires valid JSON and both components correct. Missing disease annotations in the
source are excluded from this benchmark, not treated as evidence of no disease.

You have at most {{ max_iterations }} agent iterations. Do not use external network
resources. Write a provisional JSON answer within your first two iterations and revise
it as needed. Ensure the final answer is on disk before finishing.
