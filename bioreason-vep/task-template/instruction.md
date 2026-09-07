# BioReason variant-effect prediction ({{ setting }})

Inspect `/app/data/case.json`. It contains a reference DNA sequence, a variant DNA sequence, and genomic context. Classify the variant's clinical effect.

You have at most {{ max_iterations }} steps (agent iterations) to complete this task. Do not use external network resources. Write a provisional classification to `/app/answer.txt` within your first two iterations, then revise it later if needed. Before finishing, ensure the file contains exactly `benign` or `pathogenic`.
