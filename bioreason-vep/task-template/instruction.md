# BioReason variant-effect prediction ({{ setting }})

Inspect `/app/data/case.json`. It contains a reference DNA sequence, a variant DNA sequence, and genomic context. Classify the variant's clinical effect.

Do not use external network resources or run sequence-analysis scripts. Use exactly two terminal commands: first read `/app/data/case.json`, then immediately write your best classification to `/app/answer.txt`. The file must contain exactly `benign` or `pathogenic`.
