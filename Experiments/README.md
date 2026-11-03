# Experiment standardisation

Keep model experiments here so the group can compare results and exchange saved
predictions and models without rerunning training notebooks.

## One workflow, three responsibilities

| Stage | Responsibility |
| --- | --- |
| Shared preprocessing — `Preprocessing/` | Prepare the agreed splits, features and embeddings once; publish common inputs on Drive. |
| Model training — `Experiments/` | Train/load each model, use validation for focused development, explain its behaviour and export predictions and the saved model. |
| Shared reranking and evaluation | One shared notebook/location consumes every model's train, validation and test CSVs, compares models, develops any ensemble and reranker, and produces the final evaluation tables and analysis for the report. |

Keep cross-model comparisons and final evaluation together in that shared stage.
Individual experiment notebooks analyse validation only for development; they
still generate all train/validation/test outputs. Do not analyse test results or
tune against them in experiment notebooks. Training loss curves can be used to
diagnose optimisation, but training predictions are not evidence of generalisation.

## Business objective and explainability

The **Project Interim Report**, Section 1.1 (p. 3) and Section 1.2.3 (pp. 4–5),
defines the objective as **meaningful user engagement and reader retention**
through relevant recommendations that encourage continued reading and exploration.
Click probability provides the initial relevance ranking; session continuity,
freshness and content diversity are additional reranking objectives.

Start each experiment with a business-linked hypothesis: **what information does
this model use, why should that improve recommendations, and what validation
evidence would support the choice?** Prioritise a small number of justified
experiments and targeted adjustments over adding models or repeatedly searching
hyperparameters for a marginal score increase. Explain why a change is worth
keeping, including when a simpler model is sufficient.

Use the following links to guide the explanation, rather than requiring every
model to address every objective:

| Business link | Example explanation in a model notebook | Evidence brought together in shared evaluation |
| --- | --- | --- |
| Relevant recommendations | Does category/topic affinity, reading history or content similarity explain why an article scores highly? | Impression AUC and nDCG@K on the same candidate rows, alongside representative ranking examples. |
| Continued reading | If using historical reading depth or sequence information, explain how it could represent a reader's interests over time. | Session-continuation performance where suitable outcome labels exist; distinguish this from click prediction. |
| Fresh content | Does recency information help the model consider timely articles? Explain any relevance/freshness trade-off. | Article-age distribution of top-ranked items before and after reranking. |
| Exploration and variety | Do the highest-scoring articles repeatedly concentrate on the same topics? Explain the model's behaviour and implications for reranking. | Topic/category diversity before and after reranking, alongside relevance metrics. |

Include one or two focused pieces of **validation** evidence suited to the model:
for example, feature importance or permutation importance, a targeted feature
ablation, or an explained ranking for a reader. An LSTM experiment might compare
using ordered history with a simpler history representation if sequence modelling
is its main claim. State what the evidence shows, its limitations and the resulting
decision; an importance plot or attention weight alone is not a causal explanation.

Use only information available at impression time as model inputs. Historical
reading depth can be an input; current/subsequent read-time or scroll outcomes
cannot. The report identifies these as leakage risks (Section 1.2.1, pp. 3–4).
Offline click-ranking improvements support a relevance claim, but do not by
themselves establish increased retention. Leave that distinction explicit in the
handoff to the shared evaluation/report owner.

## Tracks and numbering

| Track | ID prefix | Naming examples |
| --- | --- | --- |
| Traditional models | `1` | `1.1_log`, `1.2_random_forest` |
| Neural networks | `2` | `2.1_lstm`, `2.2_<model_name>` |
| Pretrained models | `3` | `3.1_<model_name>` |

Each experiment directory sits directly under `Experiments/` and is named
`<ID>_<short_model_name>`. The first number identifies the track; there is no
separate track directory. Agree on the next unused ID before starting. Keep an ID
stable; record hyperparameter trials as separate runs inside that experiment
rather than renumbering it. Create only the experiments you own.

```text
Experiments/
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
TRAIN_MODEL = True   # False: load the saved model and export predictions
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

1. **Business hypothesis and experiment overview** — ID, owner, model, business
   objective addressed, why this model/information should help and the focused
   validation check that will inform the decision.
2. **Setup and run configuration** — imports, dependencies, seed, paths, feature
   case, hyperparameters, output directory and train/load switch.
3. **Load and check shared data** — input versions, row counts, keys, labels,
   feature order and missing values; choose any documented development sample.
4. **Prepare model inputs** — transformations fitted on train, embedding lookups,
   tensors/arrays and saved ID mappings. Use the same preprocessing at inference.
5. **Define and train / load model** — model definition and fitting, or load a
   saved model when `TRAIN_MODEL = False`. Record validation-based early stopping
   or model selection and retain the selected checkpoint.
6. **Validation diagnostics and explainability** — use the agreed validation
   metrics and one or two focused explanations to assess the hypothesis. Record
   any justified adjustments, limitations and the selected configuration.
7. **Predict and export all splits** — generate train, validation and test
   predictions; save the six required CSVs and the complete
   model/run bundle before finishing the notebook session.
8. **Reload check** — load the saved model and preprocessing, predict a small batch
   and compare with the original predictions within a numerical tolerance.
9. **Business interpretation and handoff** — summarise the validation finding,
   how it relates to the business objective, remaining limitations and the bundle
   location for the reranking/evaluation owner. Leave final cross-model comparison
   and test analysis to the shared evaluation stage.

Keep the load-only path runnable without training. Training predictions are
in-sample diagnostics unless explicitly recorded as out-of-fold predictions.
Use validation for agreed model selection; do not tune on test results. The
current official test files have **no click labels**: still export test predictions;
labels are joined separately for evaluation where available. The shared evaluation stage must also record test
metrics as unavailable unless genuine ground truth becomes available.

## Shared behaviour identifier

A behaviour row is one user's impression slate. In both export formats below,
use **`behaviour_id = f"{user_id}:{impression_id}"`**, built from the original
integer IDs. For example, user `12345` and impression `67890` become
`12345:67890`. Treat this column as a string. Use the same definition for every
model and split; do not use a dataframe row number or an encoded model index.

This preserves the shared grouping key `(user_id, impression_id)`: test
`impression_id` values alone are not unique across users. The split is identified
by the file name. The evaluation owner can recover both original IDs by splitting
`behaviour_id` at `:` and join labels/context from the shared input tables.

## Required prediction CSVs

Every completed run exports **three click-probability CSVs**, one for each split,
using the file names in the directory tree above. Each row is **one behaviour–article
pair**, including observed non-clicks. Required columns and one example data row:

```csv
behaviour_id,article_id,click_probability
12345:67890,9876543,0.7312
```

| Column | Contract |
| --- | --- |
| `behaviour_id` | Shared behaviour identifier defined above. Repeats for each candidate in the same impression. |
| `article_id` | Original `candidate_article_id` from that behaviour's `article_ids_inview`; preserve its integer value. |
| `click_probability` | Finite probability in `[0, 1]`, not a logit or hard class label. |

The candidate key is **`(behaviour_id, article_id)`**. Preserve every candidate
row in the agreed split. Decode internal user/article index mappings before export.
Do not silently drop duplicate keys or candidates; resolve any source-key problem
consistently across experiments. Labels are not part of this three-column export:
shared evaluation joins the observed labels using the candidate key and checks
complete coverage. Unavailable test labels must never be replaced with zeros.

Use the positive-class probability from the model's probability output, sigmoid
or softmax as appropriate. A model producing only ranking scores must document
its probability conversion. Do not require probabilities within an impression to
sum to one: an impression may have multiple clicked articles.

Write with `index=False`, retain numerical precision and use a consistent row
order such as `behaviour_id`, `article_id`. Do not export a pandas index column.
Full experiment exports should cover the agreed full splits; mark development-sample
runs clearly and record their sampling rule and counts.

## Required top-100 article CSVs

Export **one row per behaviour**, containing an ordered list of up to 100 distinct
candidate article IDs. Each list ranks only the articles shown in that behaviour's
`article_ids_inview`; do not combine candidates across the user's impressions.

Required columns and one example data row, for a behaviour with three candidates:

```csv
behaviour_id,article_ids
12345:67890,"[9876543, 8765432, 7654321]"
```

`article_ids` is a **JSON array stored in one quoted CSV field**, ordered from
highest to lowest predicted click probability. Its first element is rank 1.
The example has three IDs because that behaviour has only three candidates; a
behaviour with at least 100 distinct candidates must contain exactly 100 IDs.

For each split:

1. Start from that split's complete click-probability CSV.
2. Group by `behaviour_id` and sort candidates by descending `click_probability`,
   breaking ties by ascending original article ID.
3. Keep the first `min(100, number of distinct candidates)` article IDs in order.
   Do not pad, repeat articles or add articles outside the behaviour's slate.
4. Encode each ordered list with `json.dumps(article_ids)` and save the dataframe
   with `to_csv(..., index=False)` so CSV quoting is handled correctly. Read the
   list back with `json.loads`; do not use `eval`.

Save the resulting three files under `rankings/`. Every behaviour represented in
its split's probability file must have exactly one ranking row. These exports are
base-model outputs; implement reranking rules centrally and record their
configuration there. Keep the complete probability files even when lists are
truncated to 100: shared impression AUC and nDCG evaluation uses the full candidate
predictions, grouped by `behaviour_id`.

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

Save `metrics.json` with **validation development metrics**, metric definitions,
evaluated row/impression counts and exclusions. Use the agreed protocol; the
existing baseline reports mean impression AUC and nDCG@5. Final cross-model tables,
train/validation comparisons and reranking results belong in the shared evaluation
stage, using the same split and candidate rows for every model. The existing saved
baseline benchmark used its own sample and is not directly comparable by default.
Do not report a score for unlabelled test data.

The reranking/evaluation owner receives all three splits' complete predictions and
top-100 lists. Tune any reranking/ensemble choices on validation or an agreed inner
training split, then freeze them before final evaluation. In-sample train predictions
must not be treated as out-of-fold predictions for training a stacking model.
The final report should explain the relevance, continuation, freshness and diversity
trade-offs in one place, including cases where added complexity did not help.

## Exchange a run without retraining

Keep a complete local copy under the experiment's `outputs/<run_id>/` first.
Share that run folder, or a ZIP preserving its structure, using Telegram or the
shared Google Drive. A suggested Drive location is
`BT4222 Shared Final Folder/Experiments/<experiment_id>/<run_id>/`.

Before handing it over, confirm:

- All six CSVs exist with the schemas above, original IDs, valid probabilities
  and complete split coverage; each ranking list contains only that behaviour's
  candidates in score order.
- The model, preprocessing, config/tokenizer files, metrics, run configuration
  and dependency versions are included.
- The reload check passes, and a teammate can evaluate the CSVs directly or run
  the load-only notebook path without fitting the model again.
- Code/notebook changes are ready for Git; generated outputs and model files
  remain local or on Drive rather than being added to GitHub.
