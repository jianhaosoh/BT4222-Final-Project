# 2.1_lstm — LSTM with target-aware attention (Stage 1)

> **Status.** Selection uses catalogue Recall@100. Pools and request sampling follow 3.1's `published_at_or_before_request_v1` workflow. The implementation has been checked statically, on the mini sample and against 3.1's own helper functions. No training or evaluation results exist yet.

This model learns personal click relevance from recorded shown/clicked pairs. It then scores every catalogue article eligible at each request's timestamp and exports a top-100 shortlist per request for Stage 2 reranking.

The model has three parts:

- **History:** the reader's chronological fixed pre-period clicks, encoded by an LSTM.
- **Attention:** each candidate article is projected into a query over the LSTM states, giving a candidate-specific history context.
- **Head:** an MLP that predicts a click logit from that context, the article and the request context.

It adapts the course Week 8 recipe from rating regression to binary clicks, using BCE on logits.

**Training.** Training uses each selected request's complete recorded `article_ids_inview` slate: clicked = 1, shown but unclicked = 0. No catalogue negatives are created, and unshown outcomes stay unknown.

**Catalogue pool (shared with 3.1).** The pool policy is `published_at_or_before_request_v1`. Every split's requests use the **same** article pool:

1. **Catalogue:** the ordered union of `train_val/articles.parquet` then `test/articles.parquet`, keeping the first record per article ID with its missing fields.
2. **Eligibility:** an article is eligible when its publication time is known and at or before the request time, compared exactly to the microsecond.
3. **No other filters:** there is no age or access cutoff and no embedding requirement. An article without vectors stays eligible, with missing flags.
4. **Vector overlaps:** where both vector sources have the same article ID, the first source record wins (train_val, then test), even if its vector is missing. A later source never backfills it.

Each selected request is scored against **all** of its eligible articles. The inputs a catalogue request cannot have were removed: recorded slate size and within-slate age rank. Article age is computed from the catalogue and checked against the shared `article_age_h`.

**Request sampling (shared with 3.1).** Only requests are sampled, never articles.

- **Identifier:** `behaviour_id = f"{user_id}:{impression_id}"`.
- **Priority:** `blake2b(f"{seed}:{split}:{behaviour_id}", digest_size=8)`, with seed 42.
- **Selection:** the N smallest by (priority, behaviour_id), taken from the **complete** request population.
- **Processing order:** (timestamp, user_id, impression_id).

Training, selection-validation and export limits are independent. The IDs are saved under `selection/`. Optionally, `REFERENCE_INPUT_AUDIT` and `REFERENCE_TRAINING_IDS` point to 3.1's saved IDs, and the notebook asserts they are identical.

| `RUN_SCOPE` | Training requests | Selection requests | Export requests per split |
| --- | --- | --- | --- |
| `smoke` (default) | 500 | 20 | 5 |
| `aligned_dev` (3.1's limits) | 5,000 | 100 | 25 |
| `full` | all | 100 | all |

**Selection.** Checkpoints are evaluated once per epoch, the declared interval. For the validation selection requests, the model's top 100 is computed over each request's full eligible union catalogue. The selected checkpoint has the highest macro known-click Recall@100. Ties are broken by observed-click nDCG@100, then by the earlier checkpoint or trial.

- **Early stopping, the learning-rate schedule and Optuna** all use this same objective.
- **Recorded-slate AUC** is a diagnostic only, and nDCG@5 is left to Stage 2.
- **Checkpoints record the selection declaration**, including a protocol fingerprint (pool policy, source files, union-ID hash, sample IDs and encoder identity). Resuming a checkpoint selected under a different declaration is refused. That includes legacy AUC selection and the earlier per-split pools.
- **Load-only runs of an older protocol** may reuse its weights for a new evaluation or export under the current protocol. The original selection provenance is kept, and controls are refused in that case.
- **Provenance follows the weights.** `weights_selected_under` points back to the training run's protocol and objective, including through a load of a load run. A resume with changed training settings is refused; raising `max_epochs` is still allowed.

**Metric scopes.** Selection-validation metrics (selection requests, at the selected checkpoint) and exported-validation metrics (export requests) are reported separately in `metrics.json`. With these limits, the exported validation requests are a subset of the selection requests. Compare models only on identical request IDs and pools; each scope records its ID hash. Test has no outcome metrics.

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
   - `RUN_SCOPE = "smoke"`
   - `RUN_ABLATIONS = False`
   - `RUN_OPTUNA = False`
   - `RUN_RESUME_CHECK = True`
5. Aligned development: set `RUN_SCOPE = "aligned_dev"`. For the full run, set `RUN_SCOPE = "full"`, but only after inspecting the Section 7 feasibility table.
6. Controls without retraining the main model:
   - set `TRAIN_MODEL = False`, `LOAD_RUN_ID = "<main run>"` and `RUN_ABLATIONS = True`;
   - the controls reuse the saved hyperparameters;
   - the run ID ends in `_load_controls`.
7. Resume after a disconnect: set `RESUME_RUN_ID = "<run id>"`. This continues a training run from the Drive recovery checkpoint (not guaranteed to be bitwise exact). With `TRAIN_MODEL = False` and `LOAD_RUN_ID`, it continues a load or controls run, reusing finished controls and catalogue splits.

## Vector files

Some pyarrow versions (18–20) cannot read nullable fixed-size-list vector columns. They fail with `Expected all lists to be of size=768 but index … had size=0`. For exactly that error, the loader falls back to DuckDB's Parquet reader, which keeps values, dimensions, IDs, row order and nulls. `run_config.json` records the reader used for each file.

As an alternative, set `WRITE_COMPATIBLE_VECTORS = True`. This writes verified copies with variable-length lists under `OUTPUT_ROOT/compatible_vectors/` and uses them in place of the originals. Each copy is compared exactly with its source before it is accepted, and the originals are never modified.

## Outputs

Local outputs go to `/content/bt4222_outputs/2.1_lstm/<run_id>/` in Colab, or `outputs/` locally. Both are ignored by Git.

Each run is copied to `OUTPUT_ROOT/Experiments/2.1_lstm/<run_id>/` and verified there with sha256 hashes. By default, `OUTPUT_ROOT` is `MyDrive/BT4222 Experiment Outputs`, a folder you own and separate from the shared inputs. Setting it to the shared folder also works if you have edit access. The notebook checks write access before it trains.

The run folder contains (export schema v3):

- `rankings/{split}/submission_top_100.csv`: one row per request with its top `min(100, pool)` eligible article IDs as a JSON array. An empty pool is written as `[]`.
- `predictions/{split}/eligibility_manifest.csv`: per-request pool size and shortlist length.
- `predictions/{split}/submission_click_probabilities.csv`: full catalogue scores (`score` = logit, `click_probability` = uncalibrated sigmoid). This file is written only if the estimated size fits `FULL_SCORE_EXPORT_MAX_GB` and the free space. Otherwise the scores of a declared, seeded request sample go to `predictions/{split}/score_sample/`, and `export_manifest.json` records that the full export is incomplete.
- A request-limited run, such as the smoke run, writes all of the above under `rankings/{split}/sampled_requests/` and `predictions/{split}/sampled_requests/`.
- `diagnostics/recorded_slates/{split}_scores.csv`: every recorded candidate of the split's export requests, for the labelled AUC/nDCG diagnostics. Sampled scopes write it under `sampled_requests/`.
- `model/`:
  - `model_state.pt` (state_dict)
  - `model_config.json`
  - `preprocessing.json`
  - `catalogue_policy.json`
  - `mappings.npz`
  - `code/` (bundled modules)
  - `recovery/last_checkpoint.pt`
- `metrics.json`, `run_config.json`, `export_manifest.json`, `catalogue_feasibility.json`, `training_history.{json,csv}`, `requirements.txt`, `reload_check.json`, `resume_check.json`, `manifest.json`
- `ablations/<variant>/`: `summary.json` and `validation_per_impression.npz` (per-request Recall@100 and nDCG@100 on the selection sample, and diagnostic per-impression AUC), plus a `COMPLETE.json` marker that is written last and holds both files' hashes.
- `selection/`: `validation_requests.csv` (selection requests), `train_requests.csv` (training requests, when sampled) and `{split}_export_requests.csv`.
- `logs/notebook_log.txt`
- `review_<run_id>.zip`

## Catalogue scale

The union pool is larger than the earlier per-split pools. A train or validation request now sees nearly the whole union, because most test-snapshot articles were published before the validation period. Section 4 prints exact eligible-pair counts for the selected requests, and Section 7 prints them for every request in each split, with the measured CSV size.

As a rough guide:

- A full train export is on the order of 230k requests × 10⁵ articles, about 10¹⁰ pairs.
- Validation and test are each on the order of 10⁹ pairs.
- The `aligned_dev` scope is about 10⁷ pairs: 25 export requests per split, plus 100 selection requests per epoch.

- **Scoring the development scopes is cheap on a GPU; a full run must be timed first.** Each request's eligible articles form one contiguous slice of the union sorted by publication time. Article encodings and user LSTM states are computed once, and pairs are scored in chunks.
- **Storing every score as CSV is not feasible** at about 64 bytes per row, hundreds of GB. Section 7 therefore:
  1. counts the pairs exactly;
  2. measures the CSV row size and the scoring speed;
  3. checks free disk space;
  4. skips scoring in that session if the estimated time exceeds `MAX_CATALOGUE_HOURS`. The run is still recorded, and the remaining splits can be scored later with `RESUME_RUN_ID` and a smaller `CATALOGUE_SPLITS`. Each finished split writes `catalogue/{split}.json` last, so a resumed run reuses finished splits.
- **Text-vector coverage is low for the union.** The test snapshot alone has 125,541 articles, but only about 21,000 of the union's articles have text vectors. These articles are kept, with missingness flags, and training hides 20% of candidate text vectors so the model learns to score without them. Section 4 reports pair-weighted coverage, and Section 7 reports the text-vector share inside the shortlists.
- **The shared policy has no age limit.** An age limit would be a different, separately named policy, and the code rejects one.

Optuna, when enabled, writes `OUTPUT_ROOT/Experiments/2.1_lstm/tuning/<study>/` containing a SQLite snapshot and `trials.csv`.

Validation metrics are **development** results, because validation is used for early stopping and selection. Test labels are unavailable, so no test metric is produced.
