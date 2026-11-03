# Experiment standardisation

Keep model experiments here so the group can compare results and exchange saved
predictions and models without rerunning training notebooks.

## Current notebook drafts

- [LSTM with target-aware attention](2.1_lstm/experiment.ipynb)
- [HUIM-inspired attention network](2.2_huim/experiment.ipynb)
- [Pretrained similarity experiment](3.1_pretrained_similarity/experiment.ipynb)

The latest neural notebooks and matching helpers are published on `main` for
Colab execution. They include catalogue Recall@100 selection and the shared
union-pool workflow, with small request-sampled smoke settings for the first run.
Committed notebook outputs do not yet establish training/evaluation results.
Run results will guide subsequent changes and the final notebook cleanup; later
updates can use these same experiment folders.

The 3.1 implementation supplies the reference workflow for catalogue construction,
request sampling and audit-first development below. Its current development run
does not need to be interrupted to adopt another model's tools or architecture.
The neural experiments should align their inference pools and comparison requests
with that workflow; model-specific feature transforms and training methods remain
independent. No committed notebook outputs establish training results yet.

## Stage-one task: learn relevance, then shortlist the eligible catalogue

The intended recommendation pipeline is:

**Eligible article catalogue → Stage 1 personalised top 100 → Stage 2 business
reranking → final top 5 recommendations.**

Distinguish the inputs used to learn from observed clicks from the pool scored
when making recommendations:

| Operation                            | Article pool                                         | Labels and purpose                                                                                                                               |
| ------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Training                             | Each behaviour's recorded `article_ids_inview` slate | Clicked = 1; shown but unclicked = 0. The shared feature tables already contain these expanded pairs. Keep this training setup.                  |
| Recorded-slate validation            | The same recorded candidate rows in validation       | Labelled click-ranking diagnostic; not the primary shortlist-selection objective.                                                                |
| Catalogue prediction, for each split | Every article eligible at each behaviour's timestamp | Score behaviour × eligible article, then select the top 100. Articles outside the recorded slate have **unknown** outcomes, not negative labels. |
| Stage 2                              | Stage 1's catalogue-derived shortlist                | Apply the agreed business priorities and produce the final top 5.                                                                                |

Do not re-explode the existing training feature tables or create catalogue-wide
negative labels. Catalogue prediction is a **new inference path**, not merely a
change to CSV filenames: construct model inputs for the additional pairs, reuse
the fitted preprocessing and score in batches. The current feature matrix covers
recorded candidates only; it is not the catalogue prediction matrix.

The top-100 shortlist is the current handoff size, not a requirement that every
input pool contain 100 articles. Return `min(100, eligible_article_count)` unique
IDs. Do not pad, automatically keep 50%, or exclude behaviours because their
recorded slates are small. Keep the recorded-slate benchmark separate from these
catalogue outputs.

Retain eligible articles with legitimate missing features or embeddings using
documented fallbacks and missingness indicators. Missing text alone is not a
reason to remove an article. Do not exclude older articles unless the shared
eligibility policy explicitly defines an age limit.

### Shared catalogue eligibility and inference features

Use the 3.1 policy **`published_at_or_before_request_v1`** for shared comparisons:

1. Construct one ordered union from `train_val/articles.parquet`, then
   `test/articles.parquet`, for **every** split. Keep the first source record for
   each article ID, including its missing fields. Do not use separate smaller
   train/validation and larger test pools for a like-for-like comparison.
2. An article is eligible only when its publication timestamp is known and
   `published_time <= impression_time`. Unknown dates and future publications are
   excluded; original known clicks on those articles still count in metric ideals.
   The reference policy adds no age/access cutoff and no embedding-coverage filter.
3. Combine vector sources in the same order, keeping the first source record for
   overlapping article IDs **including legitimate missing vectors**. Do not fill
   an earlier source's missing vector from a later snapshot. Missing vectors do
   not remove articles from the pool. Model-specific normalisation/projection and
   optional modalities remain legitimate architecture choices.

For each selected request, score its **full eligible union catalogue**, with no
sampled article negatives or vector-covered-only subset. Sampling requests is a
separate development/resource decision; it does not change the article pool for
any selected request. Use test-snapshot article metadata only as the declared
catalogue artifact, with the timestamp cutoff; this does not permit using test
outcomes for training/tuning or fitting learned transforms on test data.

Record source versions/order, duplicate precedence, catalogue ID fingerprint and
eligibility rules. Alternative policies must be explicitly named and compared
separately, never silently mixed in a results table. A supplied union is not proof
of complete historical production availability or an unknown production catalogue's
coverage. Do not assume every article existed when a split began.

Keep `(user_id, impression_id)` and its timestamp/context as the recommendation
request. Use only context known before making that recommendation. Audit every
feature: it must be constructible for an article outside `article_ids_inview`
without using clicks or later outcomes. Recompute any pool-dependent features
for the declared pool, consistently with training, or remove/redesign them and
retrain. Do not silently fill unavailable inference features with zeros.

Use shared article/history lookups, batch the Cartesian scoring operation, and
stream full score exports; a giant materialised behaviour × catalogue table is
not required. Before a full run, estimate eligible-pair counts, output size,
runtime and storage capacity. Exhaustive exports can contain billions of rows;
batching reduces memory pressure but does not remove compute or disk costs.
Article catalogue and embedding coverage can differ substantially: report both,
and never silently restrict the pool to articles with vectors to reduce its size.
If the exhaustive path is not feasible, document that limitation and evaluate a
declared development sample; do not claim a full export or silently switch pools.
Record catalogue coverage, exclusions and empty eligible pools.
Include behaviours with empty pools in ranking exports as `[]` rather than
silently dropping them. Record failures that prevent catalogue scoring and do
not claim a recorded-slate export implements this path.

### Audit, development and final execution

An audit-only phase is a supported workflow: inspect schemas, source identity,
feature reconstruction, vectors, eligible-pair counts and storage/runtime budgets
before training or exporting. Stream inputs/use disk indexes when useful; do not
require every model to materialise the full feature table in memory.

Sampled development runs are valid. The 3.1 reference currently uses up to 5,000
complete training requests, 100 validation-selection requests and 25 export
requests per split. Each sampled export request still uses its full eligible
article pool and has complete scores/top-100 output for that pool. Record these
three limits separately: training, selection and export counts are not interchangeable.
Do not force all models to use the same software, optimiser, trial count or feature
columns. For controlled comparisons, match data/pool/evaluation scope; report
different training budgets or modalities as differences rather than attributing
all performance differences to architecture.

Final/full scope uses all agreed training and export requests and the same pool
policy, after feasibility checks. A development run may have a complete persistent
bundle while remaining a sampled run; it must not be called a full submission.
Do not change a running development experiment solely to match another model's
larger request budget. Re-evaluate selected models on common requests later.

## One workflow, three responsibilities

| Stage                                     | Responsibility                                                                                                                                                                                      |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shared preprocessing — `Preprocessing/`   | Prepare the agreed splits, features and embeddings once; publish common inputs on Drive.                                                                                                            |
| Stage 1 — `Experiments/`                  | Train/load each relevance model on observed pairs, validate it, score the eligible catalogue, and export full catalogue scores, top-100 shortlists, recorded-slate diagnostics and the saved model. |
| Stage 2 — shared reranking and evaluation | Consume each model's catalogue-derived shortlists, apply business priorities, compare final top-5 lists and produce the final evaluation and report analysis.                                       |

Keep cross-model comparisons and final evaluation together in that shared stage.
Individual experiment notebooks analyse validation only for development; they
still generate all train/validation/test outputs. Do not analyse test results or
tune against them in experiment notebooks. Training loss curves can be used to
diagnose optimisation, but training predictions are not evidence of generalisation.

## Business objective and explainability

The **Project Interim Report**, Section 1.1 (p. 3) and Section 1.2.3 (pp. 4–5),
defines the objective as **meaningful user engagement and reader retention**
through relevant recommendations that encourage continued reading and exploration.
Stage 1 protects personal relevance through shortlist membership; Stage 2 applies
session continuity, freshness and content diversity within that shortlist. The
current intended reranker does not require click probabilities as an input.
Keep Stage 1 scores in the run bundle for diagnostics and comparisons even when
Stage 2 does not consume them. Shortlisting alone does not guarantee preserved
clicks: compare the final top 5 with Stage 1's top 5 and report the relevance and
business trade-offs. Offline clicks are an engagement proxy, not measured profit.

Start each experiment with a business-linked hypothesis: **what information does
this model use, why should that improve recommendations, and what validation
evidence would support the choice?** Prioritise a small number of justified
experiments and targeted adjustments over adding models or repeatedly searching
hyperparameters for a marginal score increase. Explain why a change is worth
keeping, including when a simpler model is sufficient.

Use the following links to guide the explanation, rather than requiring every
model to address every objective:

| Business link            | Example explanation in a model notebook                                                                                                  | Evidence brought together in shared evaluation                                                                                         |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Relevant recommendations | Does category/topic affinity, reading history or content similarity explain why an article scores highly?                                | Stage 1 catalogue known-click Recall@100 and observed-click nDCG@100; Stage 2 final-list relevance, alongside representative examples. |
| Continued reading        | If using historical reading depth or sequence information, explain how it could represent a reader's interests over time.                | Session-continuation performance where suitable outcome labels exist; distinguish this from click prediction.                          |
| Fresh content            | Does recency information help the model consider timely articles? Explain any relevance/freshness trade-off.                             | Article-age distribution of top-ranked items before and after reranking.                                                               |
| Exploration and variety  | Do the highest-scoring articles repeatedly concentrate on the same topics? Explain the model's behaviour and implications for reranking. | Topic/category diversity before and after reranking, alongside relevance metrics.                                                      |

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

| Track              | ID prefix | Naming examples                        |
| ------------------ | --------- | -------------------------------------- |
| Traditional models | `1`       | `1.1_<short_name>`, `1.2_<short_name>` |
| Neural networks    | `2`       | `2.1_<short_name>`, `2.2_<short_name>` |
| Pretrained models  | `3`       | `3.1_<model_name>`                     |

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
      train/submission_click_probabilities.csv
      validation/submission_click_probabilities.csv
      test/submission_click_probabilities.csv
    rankings/
      train/submission_top_100.csv
      validation/submission_top_100.csv
      test/submission_top_100.csv
    diagnostics/
      recorded_slates/
        train_scores.csv
        validation_scores.csv
        test_scores.csv
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
running an experiment. Keep the shared train/validation/test requests and
recorded candidate rows unchanged across models for training and the labelled
benchmark. Catalogue prediction adds a separate, consistently defined pool.

Each row is already one candidate in an impression. Preserve `split`, `user_id`,
`impression_id`, `candidate_article_id` and the supplied `clicked` target. Within
a split, the candidate key is `(user_id, impression_id, candidate_article_id)`;
the impression key is `(user_id, impression_id)`. Investigate duplicate source
keys instead of silently deduplicating. Identifiers and the target are not ordinary
numeric predictors. Never replace unavailable test targets with zeros.

Inspect the mounted schema and manifests rather than hardcoding a historical
column count. Development sampling must preserve complete impression slates;
record its rule, seed and counts. Final exports cover the complete agreed splits
and the shared eligible catalogue for each request.

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

| Input                              | Path under `DRIVE_ROOT`                                                                                   |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Raw engineered features            | `feature_engineering/{split}_features.parquet`                                                            |
| Log variants                       | `feature_engineering/log_variants/{split}_features.parquet`                                               |
| Scaled features                    | `feature_engineering/scaled_log_variants/{split}_features.parquet`                                        |
| Feature vocabulary                 | `feature_engineering/feature_vocabulary.json`                                                             |
| Train/validation article catalogue | `train_val/articles.parquet`                                                                              |
| Test article catalogue             | `test/articles.parquet`                                                                                   |
| Train/validation article vectors   | `train_val/article embeddings/{text,image}_embeddings.parquet` (some layouts use `embeddings/train_val/`) |
| Test article vectors               | `test/article embeddings/{text,image}_embeddings.parquet` (some layouts use `embeddings/test/`)           |

Use `train`, `validation`, and `test` consistently in file names and metadata;
`validation` is the split otherwise called “val”. See the
[preprocessing guide](../Preprocessing/README.md) for the four feature cases and
column selection. `scaled` and `log_scaled` use the same physical files with
different predictor selection. Resolve actual paths from the mounted files and
manifests; record them rather than assuming a particular embedding layout.

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

Stage 1 selects a catalogue-derived shortlist for Stage 2, so its model-selection
metric must measure that task:

| Role                          | Metric and evaluation pool                                                                                                                                                                    |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Stage 1 primary**           | **Mean known-click Recall@100** over validation requests, using the full time-eligible catalogue for each request. Maximise this for hyperparameter, checkpoint and early-stopping decisions. |
| **Stage 1 secondary**         | **Mean observed-click nDCG@100** on the same catalogue predictions, measuring the order within the shortlist.                                                                                 |
| Recorded-slate diagnostic     | Mean impression AUC on the original shown candidates; useful for debugging click ranking, not the final Stage 1 selection objective.                                                          |
| Stage 2 final-list evaluation | nDCG@5 for the final five recommendations, alongside the agreed business metrics and explicit observed-outcome limitations.                                                                   |

Stage 2 can reorder a retained clicked article but cannot recover one discarded
by Stage 1. That is why shortlist membership (Recall@100) is primary and its
initial ordering (nDCG@100) is secondary. nDCG@5 is not the Stage 1 objective.
Keep the training loss separate: a click-prediction loss may still train on the
recorded positive/negative pairs while catalogue Recall@100 selects the model.

Group requests by `(user_id, impression_id)` and macro-average over requests
with at least one known clicked article. Include empty eligible pools as zero
recall/nDCG when clicks are known; report requests without known clicks separately.
Record the selected epoch/run/trial, objective, direction, sampling and pool
policy. For equal Recall@100, prefer higher observed-click nDCG@100, then the
earlier trial/checkpoint. Use the same rule across experiments.

Earlier AUC-selected or recorded-slate-only runs remain diagnostic results.
Update their catalogue validation and selection logic before describing them as
models selected for top-100 shortlisting; renaming metrics or exports is not a
migration. Keep the original selection provenance for any existing run.

For recorded-slate AUC, compare clicked and unclicked candidates within each
complete impression, give equal scores half credit, exclude impressions without
both classes, and macro-average the remaining impressions. Report all counts and
exclusions. Pooled candidate-level ROC-AUC is not a substitute. Any additional
recorded-slate ranking metrics must be labelled diagnostics, not catalogue results.

The baseline's metric implementation is a useful reference, but adapt grouping
to the composite impression key. Previously saved baseline results use a separate
sample/pipeline and are not directly comparable with new experiment results.

### Required validation workflow inside each experiment

Each experiment must implement its own validation evaluation now; do not wait
for the later shared evaluation notebook. That shared stage consolidates model
comparisons and final results, while experiment owners need validation feedback
to develop their models.

### Shared request sampling and comparable metric scopes

Use 3.1's deterministic sampling for shared development comparisons. Build the
original integer-ID `behaviour_id`, then compute
`hashlib.blake2b(f"{seed}:{split}:{behaviour_id}".encode(), digest_size=8).hexdigest()`.
Order by this hash and then `behaviour_id`, take the first N requests, and process
the selected set chronologically (timestamp, user ID, impression ID). The
reference request-sampling seed is **42**, independent of model-training seeds.
Record selected request IDs; a seed alone does not prove two runs used the same
requests. Prefer reusing the running 3.1 run's saved IDs rather than asking its
owner to restart. Smaller N is a prefix of the same hash-priority order.

Keep every recorded candidate for a selected training/diagnostic request, and
every eligible catalogue article for a selected inference request. Keep the
declared fixed histories complete subject to the model's documented history
length/cutoff; request sampling does not turn later behaviour clicks into history.
This policy replaces independently sampled positional/request sets for shared
comparisons; earlier samples remain valid development runs with different scope.

The 3.1 workflow deliberately distinguishes **selection-validation** (100 requests
by default) from **exported-validation** (25 by default). Never compare those two
scores as if they used the same evaluation requests. The shared evaluator must
identify metric purpose, split, request IDs/count, pool fingerprint and selected
model. It may read each experiment's native JSON through a small adapter rather
than forcing the running implementation to rename its fields/files. For example,
3.1's `selection_validation` and `exported_validation` are separate metric scopes.
Compare models on identical request IDs and the same catalogue policy, then
report training budgets/modalities alongside results. A full-pool result on a
request sample is not a full-split result or an untouched test result.

### Development steps

1. **Fix the development data and protocol first.** Use the agreed train and
   validation partitions. Declare any reserved labelled final-evaluation partition
   before tuning, and keep it untouched. If using a development sample, fix its
   seed/request IDs, preserve their complete recorded slates for diagnostics and
   their full eligible catalogues for selection, and use the same sample across
   trials. Never replace the catalogue with sampled article negatives for the
   headline retrieval metric. Fit learned preprocessing and calibration on train
   or its permitted inner partition only.
2. **Train on train; select using validation.** For iterative models, evaluate
   catalogue validation Recall@100 at a declared checkpoint interval, save the
   best checkpoint, and use that objective for early stopping with a recorded
   patience/minimum-improvement rule. Patience counts selection evaluations, not
   necessarily epochs. For non-iterative models, compare completed fits on that
   same objective. Log training loss, Recall@100, observed-click nDCG@100 and
   diagnostic recorded-slate AUC. Training metrics diagnose optimisation and
   overfitting, not generalisation.
3. **Run a bounded, recorded search.** Try a small set of justified settings,
   manually or with an optional tuner such as Optuna. Declare the parameters,
   ranges, budget, seeds and selection rule before the search; retain all trial
   outcomes, including failed/pruned trials. Maximise the agreed validation mean
   known-click Recall@100, with the tie rule above. Do not select by training
   accuracy or silently switch to whichever metric makes a trial look best.
4. **Restore and evaluate the selected checkpoint.** Do not export the last epoch
   merely because it ran last. Report catalogue Recall@100 and observed-click
   nDCG@100, plus diagnostic recorded-slate AUC, with the agreed grouping, ties,
   denominators and exclusions. Plot progress where applicable and include a
   focused error/subgroup check or ablation. Compare identical requests and pools.
5. **Budget catalogue validation explicitly.** A fixed validation-request sample
   is acceptable for checkpoint/trial selection if full validation is too costly;
   every sampled request still uses its full eligible catalogue. Declare the
   sample and evaluation interval before searching. Compute top-100 metrics in
   batches without writing exhaustive score CSVs at every checkpoint. Report
   sample results as sampled, then evaluate the selected model on full validation
   when feasible and record what actually ran. Do not silently substitute AUC
   when catalogue validation is missing; mark shortlist selection as incomplete.
6. **Save evidence and hand off.** Save the trial table, selected configuration
   and checkpoint, validation metrics/counts, learning curves, input/pool versions
   and any sampling IDs in the run bundle. Label repeatedly used validation scores
   as development results. Export the selected model's outputs for later shared
   comparison; the shared evaluator should not need to retrain it.

### Test is never a tuning signal

**Do not tune or select models based on test-set evaluation.** Hyperparameters,
features, checkpoints, eligibility rules, shortlist size, calibration, ensemble
choices and Stage 2 reranking settings must be chosen using train/validation,
then frozen before final evaluation. Do not use test metrics, leaderboard feedback
or test business-metric comparisons to revise those choices.

The supplied official test has no click labels: generate predictions and perform
label-free schema, loading and export-integrity checks, but do not invent test
AUC, nDCG or recall. If a genuine untouched labelled test/evaluation set is
available later, evaluate the frozen pipeline for final reporting only. The
existence of test predictions does not make that split a development set.

### Catalogue shortlisting and end-to-end checks

For catalogue prediction, report **known-click Recall@100**: for each impression
with known clicks, divide the number of its clicked article IDs present in the
shortlist by its original number of clicked article IDs, then average equally
over those impressions. Keep clicked articles rejected by eligibility in that
denominator; report their count and exclusion reasons, plus eligible-only recall
and eligibility coverage separately. Also report shortlist lengths, empty pools,
scored pair counts and runtime. Use this validation recall as the primary
checkpoint/trial-selection objective and to investigate any proposed size changes.
Save the eligibility policy and shortlist size with results. Evaluate the actual
top-100 membership after the export's deterministic score/article-ID tie rule;
report ties crossing the cutoff.

For **observed-click nDCG@100**, give an article gain of 1 when it belongs to the
request's original known-click set, and 0 observed gain otherwise. Compute
`DCG@100 = sum(gain_at_rank / log2(rank + 1))` over the exported shortlist, and
divide by the ideal DCG obtained by placing all original known clicks first,
capped at 100. Include missed/excluded clicks in that ideal; do not recompute it
using only positives surviving in the shortlist. Use the same deterministic order
as the export, record the tie policy, and macro-average the same requests as
Recall@100. Zero observed gain for an unshown article is a retrieval bookkeeping
convention, **not an observed non-click label**. Name this metric explicitly as
observed-click nDCG; it is not fully observed catalogue relevance.

These are recovery/ranking measures of **observed** clicks, not unbiased estimates of catalogue
relevance, CTR, retention or profit. Unshown articles may be relevant: never label
them zero for catalogue-wide classification AUC/BCE. Record exposure bias and
catalogue coverage limitations. Observed-click retrieval metrics alone do not
establish how readers would respond to newly recommended articles.

Evaluate Stage 2's final top 5 against Stage 1's top 5 for relevance and the
business measures it targets. For the labelled recorded-slate experiment, apply
the same shortlist/reranking procedure to the recorded pool and calculate final
nDCG@5 against the **original full recorded slate's** labels/ideal ranking;
discarded positives must count against the pipeline. Catalogue recommendations
can report observed-click nDCG@5, known-click recovery at 5 and business measures,
with the same unobserved-outcome qualification used for Stage 1. Do not present a
metric that silently treats every unshown catalogue article as an observed
negative. Keep the two evaluation pools explicit in filenames, metadata and
report tables.

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
7. **Predict and export all splits** — score the eligible catalogue for train,
   validation and test requests; save the six catalogue/shortlist CSVs, the
   separate recorded-slate diagnostic scores and the complete model/run bundle.
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

A behaviour row identifies one user's recommendation request. In the exports below,
use **`behaviour_id = f"{user_id}:{impression_id}"`**, built from the original
integer IDs. For example, user `12345` and impression `67890` become
`12345:67890`. Treat this column as a string. Use the same definition for every
model and split; do not use a dataframe row number or an encoded model index.

This preserves the shared grouping key `(user_id, impression_id)`: test
`impression_id` values alone are not unique across users. The split is identified
by the file name. The evaluation owner can recover both original IDs by splitting
`behaviour_id` at `:` and join labels/context from the shared input tables.

## Required catalogue score CSVs

For each split, export
`predictions/{split}/submission_click_probabilities.csv`. Each row is one
**behaviour × eligible catalogue article** pair, including articles not in the
recorded slate. This file is not limited to the shortlist. Required columns and
one example data row for a probability-producing model:

```csv
behaviour_id,article_id,score,click_probability
12345:67890,9876543,1.0,0.7310585786
```

| Column              | Contract                                                                                                                                            |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `behaviour_id`      | Shared original-ID request identifier, repeated for every eligible article.                                                                         |
| `article_id`        | Original integer article ID from the eligible catalogue; not restricted to `article_ids_inview`.                                                    |
| `score`             | Finite Stage 1 relevance score; higher is more relevant. Declare whether it is a probability, logit or ranking score.                               |
| `click_probability` | Finite value in `[0, 1]` from a genuine click-probability model or documented fitted calibration. Uncalibrated estimates must be described as such. |

Keep logits in `score` and their sigmoid in `click_probability` when that is the
model's click-prediction interpretation. Do not put logits into a probability
column or sigmoid an arbitrary ranking score and call it a probability. If an
experiment produces ranking scores only, use the explicitly named
`submission_article_scores.csv` with `behaviour_id,article_id,score` instead and
declare that filename in the export manifest; shared readers must support this
documented alternative. Do not force probability calibration merely to satisfy
a filename. Multiple article click probabilities need not sum to one, and raw
score scales are not automatically comparable across models.

The unique candidate key is **`(behaviour_id, article_id)`**. For every request,
cover exactly the eligible catalogue IDs, not just the original slate. Decode
internal mappings before export. Save the request/eligibility manifest so empty
pools and full pair counts are verifiable. Labels are not exported; shared
evaluation joins genuine observed labels separately and leaves other outcomes
unknown. Never replace unshown or unavailable test labels with zeros.

Write with `index=False`, retain precision, and use deterministic request/article
order. Stream batches to avoid holding every pair in memory. Compression or
partitioned storage may accompany the CSV, with an explicit loader; it does not
justify silently exporting only top-100 scores. Mark sampled development runs
clearly; final exports cover all agreed requests.

## Required catalogue top-100 ranking CSVs

For each split, export `rankings/{split}/submission_top_100.csv`: **one row per
behaviour**, containing its highest-scoring eligible catalogue articles. Do not
combine separate impressions from the same user.

```csv
behaviour_id,article_ids
12345:67890,"[9876543, 8765432, 7654321]"
```

`article_ids` is a JSON array in one quoted CSV field, ordered by descending
Stage 1 `score`; the first element is rank 1. The three-ID example applies to a
pool with only three eligible articles. For each request:

1. Rank all eligible catalogue scores by descending `score`, breaking export
   ties by ascending original article ID.
2. Keep the first `min(100, eligible_article_count)` unique IDs. Do not pad or
   duplicate IDs. An empty eligible pool produces `[]` and an explicit count.
3. Encode with `json.dumps` and write with `to_csv(..., index=False)`; read back
   with `json.loads`, never `eval`.
4. Verify the list against the full catalogue score export: correct membership,
   exact top-100 ordering/length, and no article published after the request.

These files are **Stage 1 shortlists**, not the final business-reranked output.
Stage 2 receives the shortlist and the article/user information it needs. It need
not use click probabilities; preserve those separately for analysis. Record
`shortlist_size = 100` and `final_display_size = 5`. Alternative shortlist sizes
are validation experiments and require explicit filenames/metadata and an agreed
handoff update, rather than changing the meaning of `submission_top_100.csv`.

## Separate recorded-slate diagnostics and schema migration

Retain `diagnostics/recorded_slates/{split}_scores.csv` with
`behaviour_id,article_id,score` and optional genuine `click_probability`. These
files cover **every original recorded candidate exactly once**, including shown
non-clicks. They support the labelled benchmark; they are not catalogue
submissions. Obtain them by restricting catalogue scores to the recorded slate
only when features and eligibility are identical; otherwise generate and label
the benchmark separately. Investigate and report any recorded candidate absent
from the eligible pool rather than silently dropping it from evaluation.

Record **`export_schema_version = 3`**, `prediction_pool = "time_eligible_catalogue"`,
all export paths, score semantics, catalogue version, eligibility rules, counts
and shortlist size in `run_config.json` or its referenced export manifest.
Version 2's `predictions/{split}_scores.csv` and
`rankings/{split}_ranked_articles.csv` describe complete **recorded slates**.
They cannot be renamed and passed off as catalogue outputs. Update prediction
feature construction, exporters, validators and downstream readers together.
Until migrated, existing experiments must state that catalogue inference is not
implemented; the README contract alone does not implement it.

## Save a model that someone else can load

Save the selected fitted estimator or inference weights in a format supported by
the chosen stack. Include the reconstruction configuration, ordered features,
fitted preprocessing, mappings, catalogue feature construction/eligibility code
and any required tokenizer/custom loading code.
A weights file alone is insufficient when those dependencies are needed to
reconstruct the model. Record external artifact identifiers/revisions and which
components are frozen versus trained.

Keep selected inference artifacts separate from recovery checkpoints. Where
training resumption is supported, save the optimiser/scheduler state, progress,
relevant random states and any precision state needed to resume, as applicable.
Use atomic checkpoint writes and copy completed snapshots to persistent storage
during longer runs. Do not claim exact resumption unless it has been checked.

A persistent selected inference checkpoint is useful even when exact optimiser
resumption is not implemented. State that distinction. An `_in_progress` folder
is a recovery aid, not a completed handoff bundle. Input content hashes are
optional when expensive: label path/size/mtime/schema fingerprints honestly and
do not call them SHA256 hashes. At minimum, verify persistent file coverage/sizes
and a numerical reload probe from that copy; record stronger hash checks when run.

`run_config.json` must identify the experiment/run, owner, seed, Python/package
versions, code commit (and whether there were uncommitted changes), input files,
feature order/case, preprocessing, model settings, chosen checkpoint, prediction
counts, recorded-slate and eligible-catalogue coverage, eligibility policy,
shortlist size, score semantics, `export_schema_version`, tuning objective and
persistent artifact location. Copy the actual dependency versions
needed to load the model into the run's `requirements.txt`. If tuning was used,
include trial configurations/outcomes and a consistent study snapshot; do not
export an incomplete live database while assuming it is resumable.

Save `metrics.json` with **validation development metrics**, metric definitions,
evaluated row/impression counts and exclusions. Follow the validation protocol
above: primary catalogue known-click Recall@100, secondary observed-click
nDCG@100, diagnostic recorded-slate mean impression AUC and catalogue coverage.
Record the selection request sample/interval if used. Final cross-model tables,
train/validation comparisons and reranking results belong in the shared evaluation
stage, using the same split and candidate rows for every model. The existing saved
baseline benchmark used its own sample and is not directly comparable by default.
Do not report a score for unlabelled test data.

The reranking/evaluation owner receives all three splits' full catalogue scores,
top-100 shortlists and separate recorded-slate diagnostics. Tune any
reranking/ensemble choices on validation or an
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

- The six catalogue score/shortlist CSVs and three recorded-slate diagnostic
  score CSVs exist with the declared schemas, original IDs and complete request
  coverage. Full catalogue scores match each request's eligibility manifest;
  shortlist lists contain exactly its highest-scoring `min(100, pool_size)`
  unique IDs. Check future-article exclusions, empty pools and probability
  semantics separately. Do not validate catalogue exports against slate-only
  row counts.
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
