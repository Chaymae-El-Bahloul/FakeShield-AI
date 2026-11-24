# Data

Place the Fakeddit files here (not tracked by Git):

```
data/raw/
├── multimodal_train.tsv
├── multimodal_validate.tsv
├── multimodal_test_public.tsv
└── images/            # <id>.jpg  (created by scripts/download_images.py)
```

Download the TSV files from https://github.com/entitize/Fakeddit, then run:

```bash
python scripts/download_images.py --tsv data/raw/multimodal_train.tsv --out data/raw/images --limit 20000
```

Always check the license and terms of use of each dataset before redistribution.
