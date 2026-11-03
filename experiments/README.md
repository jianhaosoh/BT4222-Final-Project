# Experiment standardisation

Keep model experiments here so the group can compare results and exchange saved
predictions and models without rerunning training notebooks.

## Tracks and numbering

| Track | ID prefix | Naming examples |
| --- | --- | --- |
| Traditional models | `1` | `1.1_log`, `1.2_random_forest` |
| Neural networks | `2` | `2.1_lstm`, `2.2_<model_name>` |
| Pretrained models | `3` | `3.1_<model_name>` |

Each experiment directory sits directly under `experiments/` and is named
`<ID>_<short_model_name>`. The first number identifies the track; there is no
separate track directory. Agree on the next unused ID before starting. Keep an ID
stable; record hyperparameter trials as separate runs inside that experiment
rather than renumbering it. Create only the experiments you own.

```text
experiments/
  README.md
  2.1_lstm/                            # empty until work starts
```

When implementing an experiment, use this layout inside its directory:

```text
experiment.ipynb
requirements.txt
outputs/                               # local; ignored by Git
  <run_id>/
    predictions/
      train_click_probabilities.csv
      validation_click_probabilities.csv
      test_click_probabilities.csv
    rankings/
      train_top100_articles.csv
      validation_top100_articles.csv
      test_top100_articles.csv
    model/
      model.pkl                        # OR model.safetensors / model.h5
      preprocessing.pkl                # fitted transforms, if separate
      model_config.json                # architecture/config, when needed
    metrics.json
    run_config.json
    requirements.txt
```

Existing work in [Baseline Testing](../Baseline%20Testing/README.md) remains
available as a reference.

Use a run ID such as `20261103_1530_seed42`. Save code and dependency declarations
in Git. Save CSVs, fitted models, checkpoints and run bundles locally under
`outputs/`; they are ignored by Git. No model or prediction-file push is required.

## Shared inputs: pull from Google Drive

Use the group's **BT4222 Shared Final Folder** as the common data source. Download
the required files or access them through the mounted Drive in Colab before
running an experiment. Keep the shared train/validation/test split and candidate
rows unchanged across models.

Set the data root in one configuration cell. The existing Colab notebooks use:

```python
from pathlib import Path

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/BT4222 Shared Final Folder"
)
# Local downloaded copy, if applicable:
# DRIVE_ROOT = Path.home() / "Downloads/BT4222 Shared Final Folder"

EXPERIMENT_ID = "2.1"
SEED = 42
FEATURE_CASE = "raw"  # raw, log, scaled, log_scaled
TRAIN_MODEL = True   # False: load the saved model and run inference/evaluation
```

Mount Drive first in Colab and adjust `DRIVE_ROOT` to the location of your copy or
shortcut. Apply `SEED` to the random-number generators and model/framework
settings you use. Keep all other input paths relative to it:

| Input | Path under `DRIVE_ROOT` |
| --- | --- |
| Raw engineered features | `feature_engineering/{split}_features.parquet` |
| Log variants | `feature_engineering/log_variants/{split}_features.parquet` |
| Scaled features | `feature_engineering/scaled_log_variants/{split}_features.parquet` |
| Feature vocabulary | `feature_engineering/feature_vocabulary.json` |
| Train/validation article vectors | `embeddings/train_val/{text,image}_embeddings.parquet` |
| Test article vectors | `embeddings/test/{text,image}_embeddings.parquet` |

Use `train`, `validation`, and `test` consistently in file names and metadata;
`validation` is the split otherwise called “val”. See the
[preprocessing guide](../Preprocessing/README.md) for the four feature cases and
column selection. `scaled` and `log_scaled` use the same physical files with
different predictor selection.

Record the selected feature case, input file versions or hashes, row counts,
ordered feature columns, embedding source/dimensions and missing-value policy in
`run_config.json`. Fit additional scalers, imputers, vocabularies and probability
calibration using training data or an inner training split only, then reuse them
for validation/test. Preserve missing-vector flags and all candidate rows.

## Common notebook structure

Use these headings in this order in every experiment notebook:

1. **Experiment overview** — ID, owner, model, hypothesis and comparison baseline.
2. **Setup and run configuration** — imports, dependencies, seed, paths, feature
   case, hyperparameters, output directory and train/load switch.
3. **Load and check shared data** — input versions, row counts, keys, labels,
   feature order and missing values; choose any documented development sample.
4. **Prepare model inputs** — transformations fitted on train, embedding lookups,
   tensors/arrays and saved ID mappings. Use the same preprocessing at inference.
5. **Define and train / load model** — model definition and fitting, or load a
   saved model when `TRAIN_MODEL = False`. Record validation-based early stopping
   or model selection and retain the selected checkpoint.
6. **Predict and evaluate** — produce predictions for train, validation and test;
   compute comparable metrics wherever labels are available.
7. **Export predictions and model** — save the six required CSVs and the complete
   model/run bundle before finishing the notebook session.
8. **Reload check** — load the saved model and preprocessing, predict a small batch
   and compare with the original predictions within a numerical tolerance.
9. **Results and discussion** — metrics, comparison with the baseline, limitations
   and the main conclusion from this experiment.

Keep the load-only path runnable without training. Training predictions are
in-sample diagnostics unless explicitly recorded as out-of-fold predictions.
Use validation for agreed model selection; do not tune on test results. The
current official test files have **no click labels**: export test predictions,
leave their labels blank, and record test metrics as unavailable.

## Required prediction CSVs

Every completed run exports **three click-probability CSVs**, one for each split,
using the file names in the directory tree above. Preserve every candidate row in
the evaluated split, including observed non-clicks.

Required column order:

```csv
user_id,impression_id,candidate_article_id,click_probability,clicked
```

| Column | Contract |
| --- | --- |
| `user_id` | Original user ID; preserve its integer value. |
| `impression_id` | Original impression ID. |
| `candidate_article_id` | Original article ID from the candidate list. |
| `click_probability` | Finite probability in `[0, 1]`, not a logit or hard class label. |
| `clicked` | Observed `0`/`1` for labelled data; blank for the unlabelled test split. |

The candidate key is **`(user_id, impression_id, candidate_article_id)`**. Group
ranking evaluation by **`(user_id, impression_id)`**; test impression IDs alone
are not globally unique. Decode internal user/article index mappings before
export. Do not silently drop duplicate keys or candidates; resolve any source-key
problem consistently across experiments.

Use the positive-class probability from the model's probability output, sigmoid
or softmax as appropriate. A model producing only ranking scores must document
its probability conversion. Do not require probabilities within an impression to
sum to one: an impression may have multiple clicked articles.

Write with `index=False`, retain numerical precision and use a consistent row
order such as `user_id`, `impression_id`, `candidate_article_id`. Do not export a
pandas index column or replace unavailable test labels with zeros. Full experiment
exports should cover the agreed full splits; mark development-sample runs clearly
and record their sampling rule and counts.

## Required top-100 article CSVs

**Default: top 100 distinct articles per user within each split.** Use the same
ranking unit and aggregation rule for every model being compared. Update this
shared definition first if the group agrees on a different requirement.

Required column order:

```csv
user_id,rank,candidate_article_id,click_probability
```

For each split:

1. Start from that split's complete click-probability CSV, using the shared
   observed candidate pool.
2. Group repeated `(user_id, candidate_article_id)` pairs and take their mean
   `click_probability` across the user's impressions in that split.
3. Sort each user's distinct articles by descending mean probability, breaking
   ties by ascending article ID. Assign consecutive ranks starting at 1.
4. Export the first 100 articles for each user. If a user has fewer than 100
   distinct candidates, export all available articles; do not pad or duplicate.

Save the resulting three files under `rankings/`. The probability is the mean
used for ranking. These user-level lists and the per-impression predictions serve
different evaluation purposes: use the full candidate prediction files for
impression AUC and nDCG. Record any separately agreed eligibility or re-ranking
rule in the run configuration and apply it consistently across experiments.

## Save a model that someone else can load

Save at least one fitted model locally before ending a run:

| Model type | Accepted saved form | Include with it |
| --- | --- | --- |
| Traditional / scikit-learn | `.pkl` | Fitted preprocessing + estimator, ideally one pipeline; ordered input columns. |
| PyTorch neural/pretrained model | `.safetensors` | Model architecture/config, feature dimensions, vocabulary/ID mappings and any tokenizer files. |
| Keras / TensorFlow | `.h5` | Model configuration, fitted preprocessing and any custom-layer loading code. |

A weights file alone is insufficient when reconstructing the model requires an
architecture, vocabulary or tokenizer. Save fitted preprocessing as part of the
pipeline or as separate files in `model/`. For a pretrained model, record the
base model identifier/revision and include the trained head or adapter weights.
If the experiment only freezes a pretrained model, record that explicitly.

`run_config.json` must identify the experiment/run, owner, seed, Python/package
versions, code commit (and whether there were uncommitted changes), input files,
feature order/case, preprocessing, model settings, chosen checkpoint, prediction
counts and top-100 rule. Copy the actual dependency versions needed to load the
model into the run's `requirements.txt`.

Save `metrics.json` with labelled train/validation results, metric definitions,
evaluated row/impression counts and exclusions. Use the same evaluation protocol
across experiments; the existing baseline reports mean impression AUC and nDCG@5.
Compare baseline numbers only when they use the same split and candidate rows;
the existing saved baseline benchmark used its own sample.
Do not report a score for unlabelled test data.

## Exchange a run without retraining

Keep a complete local copy under the experiment's `outputs/<run_id>/` first.
Share that run folder, or a ZIP preserving its structure, using Telegram or the
shared Google Drive. A suggested Drive location is
`BT4222 Shared Final Folder/experiments/<experiment_id>/<run_id>/`.

Before handing it over, confirm:

- All six CSVs exist and contain original IDs, valid probabilities and the
  expected split coverage; test labels are blank when unavailable.
- The model, preprocessing, config/tokenizer files, metrics, run configuration
  and dependency versions are included.
- The reload check passes, and a teammate can evaluate the CSVs directly or run
  the load-only notebook path without fitting the model again.
- Code/notebook changes are ready for Git; generated outputs and model files
  remain local or on Drive rather than being added to GitHub.
