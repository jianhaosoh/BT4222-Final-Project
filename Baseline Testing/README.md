# Baseline testing

This folder contains the first click-ranking benchmark: random ordering, newest-first ranking, user-category affinity, and a trained logistic regression model.

## Open the results

Open `click_ranking_baseline.ipynb`. It loads the saved benchmark by default. Set `RUN_TRAINING = True` to train again.

The recorded run used 50,000 training impressions (554,696 candidate pairs) and 10,000 later validation impressions (118,731 pairs). Every candidate within each selected slate was retained, including older articles.

| Method                 | Mean impression AUC | nDCG@5 |
| ---------------------- | ------------------: | -----: |
| Random                 |               0.500 |  0.347 |
| Freshness              |               0.504 |  0.338 |
| User-category affinity |               0.559 |  0.397 |
| Logistic regression    |               0.707 |  0.540 |

## Files

- `click_ranking_baseline.ipynb`: inspect results and rerun the experiment.
- `train_baseline.py`: sampling, feature construction, training, evaluation, and output saving.
- `test_baseline.py`: five correctness checks.
- `requirements.txt`: Python dependencies.
- `results/`: fitted model, metrics, run report, dependency versions, and validation predictions.

## Run from the repository root

Use a Python environment with the dependencies installed:

```bash
python3 -m pip install -r "Baseline Testing/requirements.txt"
python3 "Baseline Testing/train_baseline.py" --root data
python3 -m unittest discover -s "Baseline Testing" -p 'test_*.py'
```

`--root` points to the dataset folder containing `articles.parquet`, `train/`, and `validation/`. The raw files are stored in this repository under `data/`. The existing virtual environment remains in the sibling `project` folder.

Without `--root`, the script checks the repository's `data/`, the repository root, then the sibling `project/` folder. Outputs default to this folder's `results/`, independent of the working directory. The notebook works when opened from either the repository root or this folder.

The saved reports retain original-run paths as provenance. Their recorded runtime excludes regenerating the saved uniform samples. Future runs write their current paths.

Binary results are ignored by this folder's `.gitignore`; small CSV and JSON reports remain available to commit. Original baseline files in the sibling project folder are preserved.

## Models and features

Compare exact expected random ranking, newest-first ranking, user-category
affinity, and logistic regression. The logistic model uses one-hot encoded
article category, article type, device, weekday and age bucket, plus scaled
numeric features: log article age, metadata quality flags, premium status,
history length, category affinity/counts, a previously-clicked flag, and cyclic
hour features. Shared context alone cannot change within-slate order in this
additive model. Personalization mainly comes from user-candidate affinity and
previous clicks. Interactions and richer models are later experiments.

All shown but unclicked candidates have click label 0. These are observed
non-clicks, not explicit dislikes. There is no negative candidate sampling,
class reweighting, use of in-view list position, or whole-catalog negative creation.

Each split uses its own supplied history file. History events must be strictly
earlier than that split's earliest sampled impression. This conservative fixed
snapshot ignores later behavior updates; it avoids future-history leakage.
Current-page and next-page read/scroll outcomes, global catalog popularity
totals, and session outcome labels are excluded.

Train timestamps must precede validation timestamps. Model settings are fixed
in advance; validation is not used to select them. Future tuning needs an inner
chronological development split within training. Article IDs and user IDs are
join keys, not model features. Missing metadata and anomalous publication times
are flagged, and their candidates are retained. For publication times after the
impression, numeric age is clipped to zero and the anomaly is reported.

## Evaluation

AUC excludes slates with only one label class. nDCG excludes slates with no
positives. Both average across impressions rather than pooling unrelated slates.
Ties receive expected credit, avoiding dependence on shuffled candidate order.
Older-article shares measure candidate slots, using expected credit at tied
top-K boundaries. Unknown publication ages are not classified as older than
24 hours; their counts appear separately in the report.

Click probabilities describe the supplied logged slates. If impressions are
click-filtered, these are conditional probabilities, not platform-wide CTR.
Offline ranking metrics do not establish retention or causal engagement gains.
