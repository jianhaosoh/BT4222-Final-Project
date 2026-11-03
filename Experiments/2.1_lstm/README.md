# 2.1_lstm — LSTM with target-aware attention (Stage 1)

This model learns personal click relevance from recorded shown/clicked pairs. It then scores every catalogue article eligible at each request's timestamp and exports a top-100 shortlist per request for Stage 2 reranking.

The model has three parts:

- **History:** the reader's chronological fixed pre-period clicks, encoded by an LSTM.
- **Attention:** each candidate article is projected into a query over the LSTM states, giving a candidate-specific history context.
- **Head:** an MLP that predicts a click logit from that context, the article and the request context.

It adapts the course Week 8 recipe from rating regression to binary clicks, using BCE on logits.

**Training and recorded-slate benchmark.** Training uses each request's recorded `article_ids_inview` slate: clicked = 1, shown but unclicked = 0. No catalogue negatives are created.

**Catalogue prediction.** Prediction uses each split's article snapshot: `train_val` for train and validation, `test` for test. An article is eligible if it was published at or before the request, with no age limit unless the group declares one. Outcomes for unshown articles are unknown.

The inputs a catalogue request cannot have were removed: recorded slate size and within-slate age rank. Article age is computed from the catalogue, and the notebook checks it against the shared `article_age_h`.

| File | Purpose |
| --- | --- |
| `experiment.ipynb` | Colab notebook with the common nine-section structure: configuration, preflight checks, training or loading, diagnostics, exports, persistence and reload check. |
| `lstm_data.py` | Path resolution, candidate and history loading, vector decoding, and train-fitted preprocessing. Also catalogue eligibility and coverage, impression metrics, known-click recall, and streaming schema-v3 exports and validators. Has no PyTorch dependency. |
| `lstm_model.py` | Article encoder, LSTM, target-aware attention, head, batching, and training with early stopping and checkpoints. Also the batched catalogue scorer, which gives an exact top-K with ties broken by article ID. |
| `requirements.txt` | Dependencies. Each run also writes its exact versions. |

## Inputs (under the shared `DATA_ROOT`)

- `feature_engineering/{split}_features.parquet` provides only the candidate keys, labels, impression context, and shared `article_age_h`. The engineered history aggregates are not used.
- `train_val/{train,validation}/history.parquet` and `test/history.parquet` provide the ordered fixed history: article, time, read time and scroll %.
- `train_val/articles.parquet` and `test/articles.parquet` provide article metadata. Future article totals are never read.
- `{train_val,test}/article embeddings/{text,image}_embeddings.parquet` hold the frozen vectors. The canonical `embeddings/{train_val,test}/` locations are also accepted.
- `feature_engineering/feature_vocabulary.json` is the shared train vocabulary.

Paths are resolved explicitly, in this order:

1. `PATH_OVERRIDES`
2. `loader_paths.json`
3. the documented candidates

A missing required file stops the run.

The shared folder is owned by another account. Add a Drive shortcut so it appears under `MyDrive`.

## Running

1. In Colab, choose a GPU runtime and open `experiment.ipynb`.
2. Code version. The notebook pins `CODE_COMMIT` and the sha256 of both helper modules. It then clones, or fetches into an existing clean clone, and checks out that commit detached. It stops if the hashes or commit differ, or if an existing clone has local changes.
3. Dependencies are checked before the helpers are imported. DuckDB is installed only if it is missing.
4. The first execution uses the defaults:
   - `SMOKE_TEST = True` (5% of users, one epoch, run ID tagged `_smoke`)
   - `RUN_ABLATIONS = False`
   - `RUN_OPTUNA = False`
   - `RUN_RESUME_CHECK = True`
5. Full main model: set `SMOKE_TEST = False`.
6. Controls without retraining the main model:
   - set `TRAIN_MODEL = False`, `LOAD_RUN_ID = "<main run>"` and `RUN_ABLATIONS = True`;
   - the controls reuse the saved hyperparameters;
   - the run ID ends in `_load_controls`.
7. Resume after a disconnect: set `RESUME_RUN_ID = "<run id>"`. It resumes from the Drive recovery checkpoint and is not guaranteed to be bitwise exact.

## Vector files

Some pyarrow versions (18–20) cannot read nullable fixed-size-list vector columns. They fail with `Expected all lists to be of size=768 but index … had size=0`. For exactly that error, the loader falls back to DuckDB's Parquet reader, which keeps values, dimensions, IDs, row order and nulls. `run_config.json` records the reader used for each file.

As an alternative, set `WRITE_COMPATIBLE_VECTORS = True`. This writes verified copies with variable-length lists under `OUTPUT_ROOT/compatible_vectors/` and uses them in place of the originals. Each copy is compared exactly with its source before it is accepted, and the originals are never modified.

## Outputs

Local outputs go to `/content/bt4222_outputs/2.1_lstm/<run_id>/` in Colab, or `outputs/` locally. Both are ignored by Git.

Each run is copied to `OUTPUT_ROOT/Experiments/2.1_lstm/<run_id>/` and verified there with sha256 hashes. By default, `OUTPUT_ROOT` is `MyDrive/BT4222 Experiment Outputs`, a folder you own and separate from the shared inputs. Setting it to the shared folder also works if you have edit access. The notebook checks write access before it trains.

The run folder contains (export schema v3):

- `rankings/{split}/submission_top_100.csv`: one row per request with its top `min(100, pool)` eligible article IDs as a JSON array. An empty pool is written as `[]`.
- `predictions/{split}/eligibility_manifest.csv`: per-request pool size and shortlist length.
- `predictions/{split}/submission_click_probabilities.csv`: full catalogue scores (`score` = logit, `click_probability` = uncalibrated sigmoid). This file is written only if the estimated size fits `FULL_SCORE_EXPORT_MAX_GB` and the free space. Otherwise the scores of a declared, seeded request sample go to `predictions/{split}/sampled_requests/`, and `export_manifest.json` records that the full export is incomplete.
- `diagnostics/recorded_slates/{split}_scores.csv`: every recorded candidate, for the labelled AUC/nDCG benchmark.
- `model/`:
  - `model_state.pt` (state_dict)
  - `model_config.json`
  - `preprocessing.json`
  - `catalogue_policy.json`
  - `mappings.npz`
  - `code/` (bundled modules)
  - `recovery/last_checkpoint.pt`
- `metrics.json`, `run_config.json`, `export_manifest.json`, `catalogue_feasibility.json`, `training_history.{json,csv}`, `requirements.txt`, `reload_check.json`, `resume_check.json`, `manifest.json`
- `ablations/<variant>/`: `summary.json` and `validation_per_impression.npz`, plus a `COMPLETE.json` marker that is written last and holds both files' hashes.
- `logs/notebook_log.txt`
- `review_<run_id>.zip`

## Catalogue scale

With publication-time eligibility only, the shared artifacts give about 8.5 billion request × article pairs. That is roughly 4.3 billion for train, 0.57 billion for validation and 3.6 billion for test.

- **Scoring all of them is feasible on a GPU.** Each request's eligible articles form one contiguous slice of the snapshot sorted by publication time. Article encodings and user LSTM states are computed once, and pairs are scored in chunks.
- **Storing every score as CSV is not feasible** at about 40 bytes per row, hundreds of GB. Section 7 therefore:
  1. counts the pairs exactly;
  2. measures the CSV row size and the scoring speed;
  3. checks free disk space;
  4. stops before scoring if the estimated time exceeds `MAX_CATALOGUE_HOURS`.
- **Text-vector coverage is low for test.** The test catalogue has 125,541 articles, but only about 21,000 have text vectors. These articles are kept, with missingness flags, and training hides 20% of candidate text vectors so the model learns to score without them. Section 4 reports pair-weighted coverage, and Section 7 reports the text-vector share inside the shortlists.
- **An age limit would shrink the pool,** but it changes the task. Section 4 tabulates pairs and known-click eligibility for a few limits as evidence for the group's decision. It is not applied by default.

Optuna, when enabled, writes `OUTPUT_ROOT/Experiments/2.1_lstm/tuning/<study>/` containing a SQLite snapshot and `trials.csv`.

Validation metrics are **development** results, because validation is used for early stopping and selection. Test labels are unavailable, so no test metric is produced.
