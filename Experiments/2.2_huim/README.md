# 2.2_huim — Candidate-aware attention for relevance shortlisting

The final notebook runs one path: **12 Optuna trials → apply the winning settings → fit all recorded training requests → restore the selected checkpoint → export every request → save and reload from shared Drive**.

Earlier bounded Colab checks completed training and saved-model reload; the plotting-directory error discovered there was fixed. Those checks do not establish the results or coverage of this updated final workflow. The final tuned all-request run has not been executed. Notebook outputs are cleared for that execution.

Course connection: the implementation combines categorical embeddings, attention, feature interactions, dense networks and dropout/weight-decay regularisation. The complete HUIM-inspired architecture is an adaptation rather than a model taught as a whole in class.

## Model and business hypothesis

Readers have several interests. Candidate-conditioned attention over individual past clicks may protect relevant articles in a larger shortlist better than a single history average. This HUIM-inspired model combines metadata embeddings, frozen text/image vector projections, historical reading-depth/recency bins, target attention and a separate request/pair feature branch. It has no recurrence or positional encoding. The final fusion uses concatenation and LayerNorm.

Training uses the supplied recorded candidates: clicked = 1 and shown-unclicked = 0. No catalogue negatives are created. Fixed histories are never augmented with later clicks. Slate size/rank, future popularity totals and current/subsequent reading outcomes are excluded. The feature audit in the notebook documents the permitted inputs.

The business pipeline is **eligible catalogue → genuine Stage 1 top 200 → Stage 2 business reranking → final five**. Offline click recovery is an engagement proxy; it does not establish retention or profit.

## Run in Colab

1. Open `experiment.ipynb` in a GPU runtime.
2. Upload `huim_helpers.zip` to `/content`, or put the editable `huim/` folder there. The ZIP contains exactly the editable `huim/*.py` files. Locally, the folder or ZIP can sit beside the notebook. No Git credentials are needed.
3. Create the Drive shortcut at `/content/drive/MyDrive/Colab Notebooks/BT4222 Shared Final Folder`.
4. Run the notebook from top to bottom. It mounts Drive, loads inputs once, runs Optuna, applies the winning parameters and fits the main model.
5. Read the actual trial count/states, checkpoint, stop reason, validation metrics and persistent reload result before reporting outcomes.

Results are saved to:

```text
DATA_ROOT/Experiments/2.2_huim/<YYYYMMDD_HHMMSS_seed42>/
```

Run IDs use Singapore time. A session-local working copy lives at `/content/huim_runs/<run_id>/`. The selected model, fitted preprocessing, feature names, ID mappings, code, selection declarations and training history are copied to Drive immediately after fitting, before catalogue export. Tuning histories/study snapshots are persisted after each trial; the complete run is copied and checked after export.

## Budgets and model selection

| Scope | Budget |
| --- | --- |
| Optuna | 12 trials, up to 2 epochs and 60,000 complete training requests per trial |
| Main training | All recorded training requests, up to 4 epochs |
| Validation selection | Fixed 100 validation requests, shared seed-42 hash rule |
| Selection interval | Every 300 optimiser steps and each epoch end |
| Export | Every request in train, validation and test; every eligible catalogue article is scored |

TPE samples learning rate, weight decay, dropout, model dimension and modality dropout. All trials use the same model seed (42) and training request prefix; only the seeded TPE search varies parameters. Median pruning uses intermediate Recall@100. Completed trials are ordered by Recall@100, then observed-click nDCG@100, then earlier trial. Failed/pruned trials stay in the history. The winning configuration is saved and applied before the main fit.

AdamW, gradient clipping and GPU mixed precision train the model on recorded-slate BCE. Early stopping and the learning-rate scheduler follow catalogue Recall@100. The selected checkpoint is restored and re-evaluated; exports use those weights.

**Selection remains K=100 while export K=200.** Recall is macro-averaged over requests with original known clicks. Missed/ineligible clicks stay in the denominator. Observed-click nDCG uses those same original clicks in its ideal, capped at K; zero observed gain for an unshown article is retrieval bookkeeping, never an observed non-click label. Exact checkpoint ties retain the earlier checkpoint.

Validation is reused for development selection. Recorded-slate AUC is diagnostic only. Exported validation reports Recall@100/200 and observed-click nDCG@100/200 separately from the 100-request selection sample. Project test labels are reserved for final evaluation and masked in this notebook. Training predictions are in-sample, not out-of-fold stacking inputs.

## Shared catalogue and request identity

Use `published_at_or_before_request_v1` for every split:

- Ordered union of `train_val/articles.parquet`, then `test/articles.parquet`; first source record per original article ID wins, including missing fields.
- Known publication time at or before request time; no recency/access cutoff or embedding-coverage filter.
- Vector sources follow the same first-record precedence; a later source never backfills a legitimately missing vector.
- Missing vectors use zero projection contributions plus explicit missingness flags. They do not exclude articles.
- Score the entire eligible pool for each request, using pair chunks and a running top-K. No sampled articles, injected clicks or exhaustive stored score matrix.

`behaviour_id = f"{user_id}:{impression_id}"` uses original integer IDs. Validation-selection and tuning request subsets use BLAKE2b-64 priority for `f"42:{split}:{behaviour_id}"`, ordered by priority then ID, and processed by timestamp/user/impression. All main training/export requests remain included.

Pool source order, duplicate precedence, catalogue fingerprint, request IDs/counts, eligible counts, exclusions and empty pools are saved. Legitimate missing vectors are distinguished from undecodable/malformed files. The vector reader uses PyArrow, with DuckDB/Polars fallbacks when available, coerces backend readers to a concrete Table and records its reader/dimensions/nulls/finite-value checks. Input preparation reads each vector source once.

## Parquet contract (v4)

| Path | Contents |
| --- | --- |
| `predictions/{split}/submission_article_scores.parquet` | Genuine retained top-200 pairs: `behaviour_id` string, `article_id` int64, 1-based integer `rank`, float32 `score`, float64 `click_probability` |
| `rankings/{split}/submission_top_200.parquet` | Exactly one row per request: `behaviour_id`, native `list<int64>` `article_ids`; `[]` for an empty pool |
| `diagnostics/recorded_slates/{split}_scores.parquet` | Every original shown candidate, including non-clicks and those outside the top 200, with IDs/logit/probability; labels joined separately |
| `diagnostics/catalogue/{split}_requests.parquet` | Request IDs/timestamps, eligible counts, shortlist lengths, eligibility exclusions, tie counts and retrieval bookkeeping |

Retain `min(200, eligible_count)` unique articles, ordered by descending score then ascending original article ID. Never merge the recorded slate or clicked IDs into this list. Keep `selection_k=100`, `export_k=200`, `shortlist_size=200`, `final_display_size=5` in the manifests. `score` is a BCE-model logit; its sigmoid is an **uncalibrated** click-probability estimate conditional on recorded-slate training, not catalogue CTR.

Parquet writes are compressed and bounded by request blocks. The running top-K merge is checked while chunk scores are available; persisted validation checks all retained rows for request coverage, IDs, eligibility, length, order, finite scores, probability semantics and pair/ranking agreement. Empty requests remain explicit. No exhaustive score file or second full-catalogue export replay is needed.

## Saved model and reuse

The run includes:

- `model/weights.pt`, `model/model_config.json`, `model/preprocessing.json`;
- `model/feature_dictionary.json`: ordered meaningful input names and categorical index-to-value maps;
- `model/article_ids.npy`: original article-ID row order;
- `model/code/huim/`: actual executed helper sources;
- `tuning/optuna_study.sqlite3`, `tuning/trials.csv`, `tuning/summary.json`, `tuning/tuned_config.json`;
- training history, selection/request scopes, run/export configuration, metrics, input manifest, actual dependency versions and `reload_check.json`;
- all three splits' retained scores, rankings and complete recorded-slate diagnostics.

`preprocessing.json` preserves fitted vocabularies, engagement/recency bins, cross-feature indexing and dense scaling. Do not refit these or guess feature labels from weights. Embedding dimensions are not interpretable word coefficients.

For later inference, add `<run>/model/code/` to the Python path and call `huim.persist.load_model(run_dir, device)`. Pass its returned preprocessor into `P.prepare_inputs(saved_config, data_root, pre=preprocessor)` with the same source artifacts. The final notebook verifies feature-name order, article-ID order and small recorded/catalogue logit batches loaded from the persistent copy within `1e-4`, without retraining.

Internal recovery helpers remain available for development, but the submitted notebook presents one final execution. Recovery checkpoints and selected inference artifacts have different roles; exact interruption recovery is not claimed for this final run.

## Later evaluation

The planned shared evaluation notebook loads these selected bundles without retraining. It will use saved histories and trials for learning/search plots, compare identical validation requests and pools, inspect representative recommendations and history/feature groups, and assess coverage, missing-vector effects, uncertainty and Stage 2 relevance/freshness/diversity/continuity trade-offs. Attention weights alone are not causal explanations. Keep test outcomes out of selection and reranker tuning.

Static/synthetic checks use `python3 -m unittest discover -s tests -v` from this directory. They do not run training or an Optuna search. Source ZIP regeneration must include exactly the current `huim/*.py` files.
