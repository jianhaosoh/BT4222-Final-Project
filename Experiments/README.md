# Experiment standardisation

Keep model experiments here so the group can compare results and exchange saved
predictions and models without rerunning training notebooks.

## Stage-one task: score the recorded impression slate

The shared feature tables are **already expanded into impression–candidate
article pairs**. Load those rows directly and score **every candidate in each
recorded impression slate**. Do not explode the tables again, retrieve a new
candidate pool from the catalogue, or label unshown catalogue articles as negatives.

Slate sizes vary. There is **no fixed candidate-count requirement**: pass every
candidate and its score to the later shared reranking stage. Catalogue and vector
lookups may enrich the supplied candidates, but must not add or remove candidates.
Retain older articles and candidates with missing features or embeddings, using
documented fallbacks and missingness indicators.

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
Model scores provide the initial relevance ranking; session continuity,
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
ablation, or an explained ranking for a reader. State what the evidence shows,
its limitations and the resulting decision; an importance plot or attention
weight alone is not a causal explanation.

Use only information available at impression time as model inputs. Historical
reading depth can be an input; current/subsequent read-time or scroll outcomes
cannot. The report identifies these as leakage risks (Section 1.2.1, pp. 3–4).
Offline click-ranking improvements support a relevance claim, but do not by
themselves establish increased retention. Leave that distinction explicit in the
handoff to the shared evaluation/report owner.

## Tracks and numbering

| Track | ID prefix | Naming examples |
| --- | --- | --- |
| Traditional models | `1` | `1.1_<short_name>`, `1.2_<short_name>` |
| Neural networks | `2` | `2.1_<short_name>`, `2.2_<short_name>` |
| Pretrained models | `3` | `3.1_<model_name>` |

Each experiment directory sits directly under `Experiments/` and is named
`<ID>_<short_model_name>`. The first number identifies the track; there is no
separate track directory. Agree on the next unused ID before starting. Keep an ID
stable; record hyperparameter trials as separate runs inside that experiment
rather than renumbering it. Create only the experiments you own.

```text
Experiments/
  README.md
  <ID>_<short_model_name>/
```

When implementing an experiment, use this layout inside its directory:

```text
experiment.ipynb
requirements.txt
outputs/                               # local; ignored by Git
  <run_id>/
    predictions/
      train_scores.csv
      validation_scores.csv
      test_scores.csv
    rankings/
      train_ranked_articles.csv
      validation_ranked_articles.csv
      test_ranked_articles.csv
    model/
      <saved_model_or_weights>          # loadable saved form for the chosen stack
      preprocessing.pkl                # fitted transforms, if separate
      model_config.json                # architecture/config, when needed
      <recovery_checkpoint>             # when training resumption is supported
    metrics.json
    run_config.json
    requirements.txt
    tuning/                            # when hyperparameter tuning is used
      <study_snapshot>
      trials.csv
```

Existing work in [Baseline Testing](../Baseline%20Testing/README.md) remains
available as a reference.

Use a run ID such as `20261103_1530_seed42`. Save code and dependency declarations
in Git. Save CSVs, fitted models, checkpoints and run bundles locally under
`outputs/`; they are ignored by Git. Persist a complete copy to the declared
shared storage destination before ending the runtime; a session-local folder
alone is insufficient. Do not add generated artifacts to ordinary Git history.

## Shared inputs: pull from Google Drive

Use the group's **BT4222 Shared Final Folder** as the common data source. Download
the required files or access them through the mounted Drive in Colab before
running an experiment. Keep the shared train/validation/test split and candidate
rows unchanged across models.

Each row is already one candidate in an impression. Preserve `split`, `user_id`,
`impression_id`, `candidate_article_id` and the supplied `clicked` target. Within
a split, the candidate key is `(user_id, impression_id, candidate_article_id)`;
the impression key is `(user_id, impression_id)`. Investigate duplicate source
keys instead of silently deduplicating. Identifiers and the target are not ordinary
numeric predictors. Never replace unavailable test targets with zeros.

Inspect the mounted schema and manifests rather than hardcoding a historical
column count. Development sampling must preserve complete impression slates;
record its rule, seed and counts. Final exports cover the complete agreed splits.

Set the data root in one configuration cell. The existing Colab notebooks use:

```python
from pathlib import Path

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/BT4222 Shared Final Folder"
)
# Local downloaded copy, if applicable:
# DRIVE_ROOT = Path.home() / "Downloads/BT4222 Shared Final Folder"

EXPERIMENT_ID = "<assigned_id>"
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

`FEATURE_CASE = "raw"` means unlogged/unscaled engineered features, not the
original event records. If an experiment needs additional inputs, declare their
paths, versions, alignment keys and temporal cutoffs; do not infer information
that the shared feature matrix does not contain.

Supplied embedding tables are fixed lookup artifacts. Record which components
are frozen and which are trained; training a downstream component does not
fine-tune the encoder that produced those vectors. Before training, actually load
the vector columns and check dimensions, finite values, null handling, flags and
join coverage. Readable metadata alone does not establish that vectors decode
correctly. Diagnose unreadable artifacts rather than silently dropping candidates
or replacing an unreadable vector column with zeros. Keep legitimate missingness
separate from a file-loading failure.

## Validation and model-selection protocol

Use **mean impression AUC as the primary metric**, maximised for hyperparameter
selection, early stopping and checkpoint selection. Report **mean impression
nDCG@5 as a secondary metric**. Record the selected epoch/run/trial, objective,
direction and any deterministic selection tie-breaker. Use the same protocol
across experiments; do not silently select different models using different metrics.

Evaluate each complete slate, grouped by `(user_id, impression_id)`, then take an
equal-weight mean over eligible impressions:

- **AUC:** compare clicked and unclicked candidates within each impression. Give
  equal scores half credit and exclude impressions without both label classes.
  Pooled candidate-level ROC-AUC is not a substitute for this metric.
- **nDCG@5:** use binary click labels and a cutoff of `min(5, slate_size)`. Give
  tied scores expected ranking credit, including ties across the cutoff. Exclude
  impressions without a positive label.
- Report total candidate rows and impressions, eligible impressions for each
  metric, and all exclusion counts. Deterministic article-ID tie-breaking in
  exported lists must not determine metric credit.

The baseline's metric implementation is a useful reference, but adapt grouping
to the composite impression key. Previously saved baseline results use a separate
sample/pipeline and are not directly comparable with new experiment results.

State the partition used for tuning and early stopping. If validation is used
repeatedly for selection, report its scores as **development/validation results**,
not an untouched final test. Any reserved labelled evaluation partition must be
defined consistently across experiments before tuning. The official test split
has no click labels: export predictions, but do not tune on it or invent metrics.

Hyperparameter-search technology is optional. If using a tuner such as Optuna,
document the search space, budget, sampler/pruner, seeds, objective and outcomes;
save a resumable study snapshot and portable trial history. Manual searches must
also record the attempted configurations. Fit additional transforms and any
probability calibration only on the permitted training partition or inner split.

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
   saved model when `TRAIN_MODEL = False`. Follow the primary validation metric
   for selection and retain the selected checkpoint.
6. **Validation diagnostics and explainability** — use the agreed validation
   metrics and one or two focused explanations to assess the hypothesis. Record
   any justified adjustments, limitations and the selected configuration.
7. **Predict and export all splits** — generate train, validation and test
   predictions; save the six required CSVs and the complete
   model/run bundle before finishing the notebook session.
8. **Reload check** — load the model and preprocessing from the persistent copy,
   predict a small batch and compare with the original predictions within a
   stated numerical tolerance, without retraining.
9. **Business interpretation and handoff** — summarise the validation finding,
   how it relates to the business objective, remaining limitations and the bundle
   location for the reranking/evaluation owner. Leave final cross-model comparison
   and test analysis to the shared evaluation stage.

Keep the load-only path runnable without training. Training predictions are
in-sample diagnostics unless explicitly recorded as out-of-fold predictions.
Use validation for agreed model selection; do not tune on test results. The
current official test files have **no click labels**: still export test predictions;
labels may be joined separately only where genuine ground truth becomes
available. The shared evaluation stage must otherwise record test metrics as
unavailable. Notebook outputs alone do not establish that a model can be reloaded.

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

## Required per-candidate score CSVs

Every completed run exports **three score CSVs**, one for each split, using the
file names in the directory tree above. Each row is **one behaviour–article pair**,
including observed non-clicks. Required columns and one example data row:

```csv
behaviour_id,article_id,score
12345:67890,9876543,0.7312
```

| Column | Contract |
| --- | --- |
| `behaviour_id` | Shared behaviour identifier defined above. Repeats for each candidate in the same impression. |
| `article_id` | Original `candidate_article_id` from that behaviour's `article_ids_inview`; preserve its integer value. |
| `score` | Finite relevance score; higher means more relevant. May be a probability, logit or ranking score, as declared in metadata. |

The candidate key is **`(behaviour_id, article_id)`**. Preserve every candidate
row in the agreed split. Decode internal user/article index mappings before
export. Do not silently drop duplicate keys or candidates; resolve source-key
problems consistently across experiments. Labels are not included: shared
evaluation joins observed labels using the candidate key and checks complete
coverage. Unavailable test labels must never be replaced with zeros.

Declare `score_type` (`probability`, `logit` or `ranking_score`), the score source,
and any conversion or calibration in `run_config.json`. An optional
`click_probability` column is allowed when the model genuinely estimates click
probability; it must be finite and in `[0, 1]`. Applying a sigmoid to an arbitrary
ranking score does not establish a click-probability interpretation. Multiple
articles may be clicked, so per-candidate click probabilities need not sum to one.
Raw score scales are not automatically comparable across models for averaging or
probability-based combinations.

Write with `index=False`, retain numerical precision and use a consistent row
order such as `behaviour_id`, `article_id`. Do not export a pandas index column.
Full exports cover the agreed full splits; mark development-sample runs clearly
and record the sampling rule and counts.

### Migration from the earlier export proposal

This contract replaces `*_click_probabilities.csv` and its mandatory
`click_probability` column with `*_scores.csv` and `score`. It also replaces the
earlier fixed-count ranking filenames with `*_ranked_articles.csv`. Update
exporters and shared readers together; do not silently interpret an old file as
the new schema. Record `export_schema_version = 2` in run metadata. If a legacy
consumer remains, document an explicit compatibility adapter/export rather than
mislabel raw scores as probabilities or truncate candidate slates.

## Required complete-slate ranking CSVs

Export **one row per behaviour**, containing every candidate article ID in that
recorded impression exactly once. Do not combine candidates across a user's
impressions or impose a fixed list length.

Required columns and one example data row, for a behaviour with three candidates:

```csv
behaviour_id,article_ids
12345:67890,"[9876543, 8765432, 7654321]"
```

`article_ids` is a **JSON array stored in one quoted CSV field**, ordered by
descending score. Its first element is rank 1. A behaviour with three candidates
exports three IDs; a behaviour with more candidates exports all of them.

For each split:

1. Start from that split's complete score CSV.
2. Group by `behaviour_id` and sort candidates by descending `score`, breaking
   ties by ascending original article ID for reproducible exports.
3. Retain every candidate. Do not truncate, pad, repeat articles or add articles
   outside the behaviour's slate.
4. Encode each ordered list with `json.dumps(article_ids)` and save it with
   `to_csv(..., index=False)` so CSV quoting is handled correctly. Read it back
   with `json.loads`; do not use `eval`.

Save the three files under `rankings/`. Every behaviour in its split's score file
must have exactly one ranking row. Validate that each ranking's length and set of
article IDs exactly match that behaviour's candidate keys in the source data.
These are base-model outputs; implement reranking centrally. Metrics use all
candidate scores and fair score-tie handling, not the export's article-ID ordering.

## Save a model that someone else can load

Save the selected fitted estimator or inference weights in a format supported by
the chosen stack. Include the reconstruction configuration, ordered features,
fitted preprocessing, mappings and any required tokenizer/custom loading code.
A weights file alone is insufficient when those dependencies are needed to
reconstruct the model. Record external artifact identifiers/revisions and which
components are frozen versus trained.

Keep selected inference artifacts separate from recovery checkpoints. Where
training resumption is supported, save the optimiser/scheduler state, progress,
relevant random states and any precision state needed to resume, as applicable.
Use atomic checkpoint writes and copy completed snapshots to persistent storage
during longer runs. Do not claim exact resumption unless it has been checked.

`run_config.json` must identify the experiment/run, owner, seed, Python/package
versions, code commit (and whether there were uncommitted changes), input files,
feature order/case, preprocessing, model settings, chosen checkpoint, prediction
counts, complete-slate coverage, score semantics, `export_schema_version`, tuning
objective and persistent artifact location. Copy the actual dependency versions
needed to load the model into the run's `requirements.txt`. If tuning was used,
include trial configurations/outcomes and a consistent study snapshot; do not
export an incomplete live database while assuming it is resumable.

Save `metrics.json` with **validation development metrics**, metric definitions,
evaluated row/impression counts and exclusions. Follow the validation protocol
above: primary mean impression AUC, secondary nDCG@5. Final cross-model tables,
train/validation comparisons and reranking results belong in the shared evaluation
stage, using the same split and candidate rows for every model. The existing saved
baseline benchmark used its own sample and is not directly comparable by default.
Do not report a score for unlabelled test data.

The reranking/evaluation owner receives all three splits' complete predictions and
complete ranked slates. Tune any reranking/ensemble choices on validation or an
agreed inner training split, then freeze them before final evaluation. In-sample train predictions
must not be treated as out-of-fold predictions for training a stacking model.
The final report should explain the relevance, continuation, freshness and diversity
trade-offs in one place, including cases where added complexity did not help.

## Exchange a run without retraining

Keep a complete local copy under the experiment's `outputs/<run_id>/`, then
persist that run folder or a ZIP preserving its structure to the declared shared
storage destination. A suggested Drive location is
`BT4222 Shared Final Folder/Experiments/<experiment_id>/<run_id>/`.

Do not rely on a Colab/cloud runtime's local disk surviving the session. If
persistent storage is inaccessible, prepare an export bundle and transfer
instructions and mark remote persistence/reload as incomplete. A notebook with
saved metrics or output cells is not a substitute for the fitted model bundle.

Before handing it over, confirm:

- All six CSVs exist with the schemas above, original IDs, finite scores and
  complete split coverage; each ranking list contains every one of that
  behaviour's candidates exactly once in score order. Validate optional
  probabilities separately when provided.
- The model, preprocessing, config/tokenizer files, metrics, run configuration
  and dependency versions are included, together with recovery/tuning artifacts
  when applicable.
- The persistent copy is complete. Reload from that copy and verify sample
  predictions against the original within a recorded tolerance, without fitting
  again. State what was actually executed; preparing the check is not evidence
  that it passed.
- A teammate can evaluate the CSVs directly or run the load-only notebook path
  using the documented artifact locations and dependencies.
- Code/notebook changes are ready for Git; generated outputs and model files
  remain local or on Drive rather than being added to GitHub.
