# 2.1 LSTM with target-aware attention

**Owner:** Winston Ho Rui Wei. A previous L4 smoke run completed successfully.
The final tuned, all-request schema-v4 workflow below has **not yet been run**.
The updated code passed syntax checks and tiny synthetic input-preparation,
scoring, export, metric and saved-copy reload checks without training or Optuna.
Those checks establish code behaviour on the fixture, not full-data results.

The model learns personal click relevance from recorded slates, then scores every
eligible article and retains the genuine top 200 for Stage 2 business reranking
to five recommendations. Unshown article outcomes remain unknown.

## Model and course connection

The course Week 8 concepts are chronological LSTM history encoding, embeddings
and candidate-specific attention. Each reader's fixed pre-period history retains
its latest 50 events in chronological order, with masked padding. Each candidate
queries the LSTM states; an MLP combines the resulting history representation,
article features, candidate age and pre-request context to produce a click logit.

The rating-regression recipe becomes binary click prediction: BCE on logits fits
clicked = 1 and shown but unclicked = 0 within each original recorded slate.
There are no catalogue-wide negative labels. Frozen text/image vectors are
projected alongside metadata embeddings and train-standardised numeric features.
Missing vectors use zero placeholders plus explicit availability flags, and 20%
training-only candidate text dropout teaches this fallback. Missing vectors never
exclude an otherwise eligible article. Fitted transformations use training
requests/history only and are reused unchanged at inference.

## Run the notebook

1. Open `experiment.ipynb` in a Colab L4 GPU runtime.
2. Upload `lstm_data.py` and `lstm_model.py` to `/content`, beside the notebook.
   Locally, keep both helpers beside the notebook and use that working directory.
3. Add the shared Drive folder shortcut at
   `/content/drive/MyDrive/Colab Notebooks/BT4222 Shared Final Folder`.
   This is `DATA_ROOT`; local execution can set it to the local shared-data copy.
4. Run the cells in order. Setup mounts Drive, imports the uploaded helpers and
   installs DuckDB/Optuna if missing. It uses the runtime's core scientific stack.

The 26-cell notebook follows the nine shared experiment sections. Its single
path is **Optuna → apply winning parameters → fit on all training requests →
restore the selected checkpoint → export all three splits → reload from the
persistent copy**. No Git checkout or credentials are required.

| Stage | Declared budget and scope |
| --- | --- |
| Optuna | Up to 12 trials, three epochs/trial, at most 60,000 complete training requests/epoch; 150-minute timeout checked between trials, so an active trial can finish later. |
| Trial selection | Same fixed 100 validation requests, seed 42, with the complete eligible catalogue for each request. Every model/data-order trial seed is 42. |
| Final fitting | All recorded training requests each epoch; up to eight epochs, patience two, batch size 256, AMP on CUDA. |
| Export | All requests in train, validation and test; every eligible article scored in chunks of 32,768, retaining a running top 200. |

Optuna uses seeded TPE and a median pruner (three startup trials, one warmup step).
Its search space is LSTM hidden size `{64,128,256}`, attention size `{32,64,128}`,
article size `{64,128}`, dropout `0.1–0.4`, learning rate `3e-4–3e-3` (log), weight
decay `1e-6–1e-3` (log), and history dropout `0–0.3`. All seven winning settings
are applied before final fitting. Exact eligible-pair counts, retained-output
size estimates and available space are printed before training; export progress
reports elapsed time and an ETA. Full-run runtime remains unmeasured.

## Inputs, catalogue and selection

Inputs live under `DATA_ROOT`:

- `feature_engineering/{split}_features.parquet`: existing recorded candidate
  rows, labels and pre-request context; no re-explosion of those rows.
- `feature_engineering/feature_vocabulary.json`: shared feature/category vocabulary.
- `train_val/{train,validation}/history.parquet` and `test/history.parquet`: fixed
  pre-period histories with article/time/read-time/scroll fields.
- `train_val/articles.parquet`, then `test/articles.parquet`: ordered metadata union.
- `embeddings/{train_val,test}/{text,image}_embeddings.parquet`: frozen vectors;
  the supplied `{train_val,test}/article embeddings/` layout is also accepted.
- Original behaviour Parquets, when present, are cross-checked against the feature
  candidates, labels and composite request keys.

The path helper supports the shared folder layouts and supplied `loader_paths.json`.
For nullable fixed-size-vector Parquets affected by the known PyArrow read error,
it uses DuckDB's reader while preserving dimensions, row order, IDs and nulls.

All splits use `published_at_or_before_request_v1`: the ordered union of train_val
then test, keeping the first article/vector source record, including missing
values. Known publication time must be at or before the request time, compared
in microseconds. There is no age/access cutoff or embedding-coverage filter.
Catalogue features use only metadata, prior history and pre-request context;
recorded slate size, pool-relative ranks and future article totals are excluded.

`behaviour_id = "user_id:impression_id"` uses exact original integer IDs. The
selection sample is the smallest 100 `(BLAKE2b-64("42:validation:behaviour_id"),
behaviour_id)` priorities from the complete validation population, processed in
(timestamp, user_id, impression_id) order, matching the 3.1 sampling rule.
Only checkpoint/trial selection uses this sample; final fitting and exports use
all recorded requests.

Selection maximises macro **known-click Recall@100**, then **observed-click
nDCG@100**, then preserves the earlier checkpoint/trial. Recorded-slate mean
impression AUC is diagnostic. Final export reports the selected model on full
validation at both K=100 and K=200; export size does not change the selection
objective. Recall denominators and nDCG ideals retain all original known clicks,
including missed/ineligible clicks. Unshown articles' zero observed gain is
retrieval bookkeeping, not a non-click label. Validation is repeatedly used
**development** data. Project test labels are kept on disk for final evaluation and masked during development.

## Outputs and persistent model bundle

Local Colab outputs are `/content/lstm_outputs/<run_id>/` (locally,
`outputs/<run_id>/` beside the helpers). The persistent destination is
`DATA_ROOT/Experiments/2.1_lstm/<run_id>/`. Completed study snapshots and epoch
checkpoints are copied during the run; the selected inference bundle is persisted
before the long catalogue export. Exact interruption resumption is not claimed.

Schema v4 exports the following compressed Parquets for **every request**:

| File | Contents |
| --- | --- |
| `predictions/{split}/submission_article_scores.parquet` | Genuine retained top-200 pairs: string `behaviour_id`, int64 `article_id`, 1-based integer `rank`, float32 `score` (click logit). |
| `rankings/{split}/submission_top_200.parquet` | One row/request with string `behaviour_id` and native `list<int64>` `article_ids` in identical rank order; empty pools are `[]`. |
| `predictions/{split}/eligibility_manifest.parquet` | Every request's ID, exact timestamp, eligible count and shortlist length. |
| `diagnostics/recorded_slates/{split}_scores.parquet` | Every original recorded candidate, including non-clicks and ineligible candidates: `behaviour_id`, `article_id`, `score`. Original labels are joined separately. |

Rank by descending score, then ascending original article ID. Return
`min(200, eligible_count)` unique IDs without padding or inserting clicked/slate
articles. Pair scores contain **only the top 200**, not all eligible articles.
No exhaustive score matrix is exported. `selection_k=100`, `export_k=200`,
`shortlist_size=200`, and `final_display_size=5` are saved explicitly.
Training predictions are in-sample diagnostics, not out-of-fold stacking inputs.
Scores are uncalibrated logits; no calibrated probabilities are claimed.

The reloadable `model/` folder contains:

- `model_state.pt`, `model_config.json`: selected inference weights and architecture.
- `preprocessing.json`: fitted transformations, vocabulary and feature settings.
- `feature_dictionary.json`: ordered human-readable features, category/type/topic
  index meanings, missing-value conventions and history/score semantics.
- `article_inputs.npz`: frozen article metadata/vector lookup, including missingness.
- `mappings.npz`: article-row/original-ID mapping and training-referenced IDs.
- `catalogue_policy.json` and `code/{lstm_data,lstm_model}.py`: policy and source helpers.

Other evidence includes `tuning/{study.db,trials.csv,summary.json}` and trial
histories; `selection/{declaration.json,validation_requests.parquet}`;
`model_training/` with selected/recovery checkpoints and training history;
`training_history.{json,csv}`; `catalogue/{split}.json`; `metrics.json`;
`run_config.json`; `export_manifest.json`; exact run `requirements.txt`;
`manifest.json`; and `reload_check.json`. Input path/size/mtime and the catalogue-ID
fingerprint are recorded; source helpers are bundled, with Git provenance when
available. Generated run artifacts stay outside Git.

The notebook checks score/ranking consistency, all request IDs/counts and
persistent file coverage/sizes. It then reloads the helpers, weights, frozen
article inputs, mappings and fitted transformations from the saved copy and
compares a small reconstructed batch with the original model at absolute logit
tolerance `1e-4`, without refitting or retraining. New requests still require the
documented pre-request context and fixed-history inputs.

## Handoff and later evaluation

The reranking owner receives all splits' genuine top-200 scores/rankings,
complete recorded-slate diagnostics and the model/run bundle. Join labels by
original IDs; preserve Stage 1 ranks/scores during business reranking to five.
Observed-click recovery is affected by exposure bias and catalogue coverage and
does not establish unseen-article CTR, retention or profit.

The planned later evaluation notebook described in [the shared experiment
README](../README.md) will load saved artifacts without retraining and compare
models on identical validation requests/pools. Learning curves, Optuna plots,
ablations, history/coverage subgroups, meaningful feature/history explanations and
Stage 2 trade-offs belong there. Attention weights alone are not causal
explanations.
