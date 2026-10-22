# Business Entity Resolution — Amazon ML Challenge

A command-line machine-learning project that matches business records from Source 2 and Source 3 to records in Source 1.

## How it works

1. Cleans business names and addresses, including common abbreviations and some Hindi transliteration.
2. Finds a short list of possible matches in the same country.
3. Compares each pair using name, address, number, and text-similarity features.
4. Uses a LightGBM model to choose the best match when its score clears a tuned threshold.
5. Writes candidate pairs and final matches as TSV files.

The default pipeline retrieves up to 20 candidates, reranks them, and scores up to 8 per query. Training uses a held-out-region validation split and tunes the threshold for macro F0.5. The default training sample uses 3% of non-validation queries. Wider retrieval settings have been measured; retrieval 80 is currently the best validation result. See [EXPERIMENTS.md](EXPERIMENTS.md).

## Measured result

On one held-out training-data split, retrieval 80 with an eight-candidate shortlist reached **0.9520 macro F0.5** and retained **95.44%** of known matches in that shortlist. These are validation estimates, not test or leaderboard scores. The decision threshold was selected on the same holdout, so the estimate may be optimistic. See the experiment notes for the split and comparison details.

## Project files

```text
src/normalize.py       Text and address normalization
src/blocking.py        Candidate generation
src/features.py        Pairwise matching features
src/pipeline.py        Shared candidate and feature pipeline
src/evaluate.py        Macro F0.5 metric
src/train.py           Training and threshold selection
src/predict.py         Test prediction and TSV output
src/check_outputs.py   Local submission-format checker
requirements.txt       Python dependencies
EXPERIMENTS.md         Retrieval comparison and validation limits
docs/index.html        Static portfolio project page
.github/workflows/     GitHub Pages deployment workflow
```

## Data

The challenge data is not included. Keep it outside the repository and arrange it like this:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

You can also pass the parent data folder with `--data-dir`, so the dataset does not need to be inside the repository.

## Run from this repository folder

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Train and predict, replacing the data path if needed:

```bash
python src/train.py --data-dir dataset --model-dir models
python src/predict.py --data-dir dataset --model-dir models --out-dir output
python src/check_outputs.py --out-dir output --test-dir dataset/test
```

To reproduce the better measured validation setting, use a separate model folder:

```bash
python src/train.py --data-dir dataset --model-dir models/retrieve80 --retrieve-k 80 --model-k 8
python src/predict.py --data-dir dataset --model-dir models/retrieve80 --out-dir output --tmp-dir temp_predict
```

Training checkpoints are saved under `models/ckpt/` so an interrupted run can resume. Remove `models/` to start from scratch. The local checker checks the output structure; the official challenge validator is not included in this repository.

Training prepares and processes Source 1 by country and saves query features in resumable chunks. The full feature set can still need several gigabytes of RAM and disk; 16 GB RAM and about 20 GB free disk are recommended. Actual runtime depends on the machine and dataset.

## Output files

- `output/candidate_pairs.tsv` lists the shortlisted Source-2/3 records for each Source-1 row.
- `output/matching_results.tsv` lists the model-selected matches.

The project is currently a command-line pipeline; it does not include a web app or hosted prediction service. Test-set accuracy cannot be measured without test ground-truth labels.

## Resume project page

`docs/index.html` is a static overview of the project and measured validation results; it does not run predictions. The workflow in `.github/workflows/pages.yml` publishes that page through GitHub Pages after Pages is enabled with GitHub Actions as the repository's publishing source.

## License

No license has been selected yet. Check the challenge terms and choose a license before inviting reuse of this code.
