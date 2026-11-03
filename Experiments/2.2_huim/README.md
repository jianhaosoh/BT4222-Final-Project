# 2.2_huim — HUIM-inspired Stage 1 relevance model and catalogue shortlist

> **Branch status vs the draft on main.** The draft merged to main (AUC selection, PyArrow-only vector reader) is
> superseded on this branch: catalogue Recall@100 selection, the DuckDB/Polars reader-to-Table fix and the shared
> `published_at_or_before_request_v1` union pool are implemented here, pending execution in Colab.

**Status:** the implementation is prepared and has passed static and synthetic checks (no optimiser steps). It has **not** been
executed on the shared data. No training, metric, export, persistence or reload result exists yet. The
executed Colab notebook and its review bundle will be the first evidence.

## What this experiment does

It implements Stage 1 only:

*eligible catalogue → **personalised top 100** → (Stage 2 reranker, not here) → final top 5*.

| Operation | Pool | Labels |
| --- | --- | --- |
| Training | Shared recorded pairs (`feature_engineering/{split}_features.parquet` keys + `clicked`) | clicked = 1, shown-unclicked = 0; no re-explosion, no catalogue negatives |
| Catalogue validation (selection) | The fixed `selection_validation` requests, each scored against its **full** eligible union pool | **known-click Recall@100 (primary)**, observed-click nDCG@100 (secondary); unshown outcomes unknown |
| Catalogue prediction (export) | The `export` requests of each split × every article of the shared union with a known `published_time <= impression_time` | `exported_validation` Recall@100 / nDCG@100; coverage only for test |
| Recorded-slate diagnostic | Every recorded candidate, grouped by `(user_id, impression_id)` | mean impression AUC, **diagnostic only** |

## Metric contract and model selection (Experiments/README.md, commit 2869a48)

* **Primary.** Macro known-click Recall@100 over validation requests with known clicks:
  * clicked articles rejected by eligibility stay in the denominator;
  * an empty pool scores 0;
  * membership follows the export order (descending score, ties by ascending `article_id`).
* **Secondary.** Observed-click nDCG@100 on the same shortlists:
  * gain is 1 for an original known click; any other article has 0 *observed* gain, which is bookkeeping, not a negative label;
  * the ideal DCG places **all** original known clicks first, capped at 100, so missed and ineligible clicks count against it.
* **Diagnostic.** Recorded-slate AUC. nDCG@5 belongs to Stage 2's final list.
* **Selection.**
  * Every `selection_interval_steps` (development 20, smoke 4, full 300) and at each epoch end, the fixed `selection_validation` requests are scored against their full union pools. There are 100 such requests (smoke: 25): the shared BLAKE2b prefix with seed 42, or 3.1's saved IDs. They are saved in `selection/request_scopes.json` and `selection/selection_requests.json`.
  * No score CSVs are written during selection.
  * The best checkpoint is chosen by Recall@100, then nDCG@100, then the earlier checkpoint.
  * Early stopping counts selection evaluations without improvement (patience 3, `min_improvement` 0). `ReduceLROnPlateau` follows Recall@100.
  * The selected checkpoint is restored and re-evaluated. A mismatch with its recorded metrics is logged and stored in `selection`.
  * Controls and Optuna trials use the same sample and rule. Optuna prunes on intermediate Recall@100 and keeps pruned and failed trials.
* **Provenance.** Recovery checkpoints carry the selection protocol: `catalogue_recall@100_v2_union_pool_hash_requests`, with the sample hash, interval and pool fingerprint. A checkpoint from another protocol, such as the earlier AUC-selected version of this experiment, is refused. A loaded legacy bundle is labelled diagnostic and is not described as Recall-selected.
* **Partitions.** Validation is development data. No reserved labelled final partition is declared by the shared protocol. Test is never evaluated or used for any choice.

Files:

* `experiment.ipynb`: the Colab notebook, using the common headings 1–9.
* `huim/`: helper modules.
* [`huim_helpers.zip`](huim_helpers.zip): the same helper source files packaged for
  direct upload to Colab; no datasets, weights or generated run outputs.
  * `data`: inputs and checks.
  * `features`: request/article/history tensors and the feature audit.
  * `model`, `train`, `metrics`.
  * `catalogue`: eligibility, scoring and streamed v3 exports.
  * `validate`, `persist`.
  * `sampling`: shared request sampling (3.1 rule), saved-ID reuse, pool policy and fingerprint.
  * `selection`: catalogue Recall@100 checkpoint selection.
  * `pipeline`: orchestration, request scopes and budgets.
  * `tune`: optional Optuna search.
* `tests/test_static.py`: synthetic checks (no shared data, no optimiser steps).

Keep `huim/` as the editable source. Rebuild `huim_helpers.zip` from its Python
files whenever those helpers change so the drag-and-drop copy stays current.

Run the tests with:

```bash
cd Experiments/2.2_huim && python -m unittest discover -s tests -v
```

## Model (an adaptation, not a reproduction of HUIM)

One shared **article tower** is used for candidates and history items. Its trained parts are:

* embeddings for category, type and sentiment;
* mean bags over topics and subcategories;
* premium and body-length features;
* **separate** trained linear projections of the frozen text vectors (gemini-embedding-2, 768-d) and image vectors (EB-NeRD, 1024-d), plus `has_text` / `has_image` flags.

The text and image spaces are never compared with each other. Missing vectors contribute zero, and the article is kept.

**History events** are the last 50 fixed-history clicks. Each event is the article tower output plus three
embedded bins, projected to d = 64:

* read time (quantiles fitted on train);
* scroll (quantiles fitted on train);
* recency to the request (fixed bins).

**Candidate-conditioned attention** follows the DIN/HUIM form `a_j = MLP([x_j, c, x_j−c, x_j⊙c])`:

* The first layer is factorised as `A x_j + B c + P(x_j⊙c)`. This is the same function class, and it lets catalogue scoring reuse per-request and per-article terms.
* Modes:
  * `softmax` (default): stable masked softmax.
  * `unnormalised`: the HUIM/DIN masked sum, plus our LayerNorm.
  * `uniform`: a control with no candidate conditioning.
* An empty history uses a learned vector.
* There is no recurrence and no positional encoding, so permuting aligned events leaves scores unchanged (tested).

**Matching and scoring:**

* The matching vector `[h, c, h−c, h⊙c]` is projected to 64.
* A separate **feature branch** takes request context and pair features (see the audit).
* Fusion is LayerNorm(concat), with an optional sigmoid gate (`fusion = "gated"`).
* The head is an MLP (128, 64) producing a logit.

**Training choices:**

* BCE on logits; AdamW, with no decay on biases, norms or embeddings.
* Dropout 0.2 and modality dropout 0.15. The test catalogue mostly lacks vectors, so the model must also score from metadata alone.
* Gradient clipping and AMP.
* Selection, early stopping and `ReduceLROnPlateau` follow validation catalogue Recall@100 (see above); at most 4 epochs.

These defaults are hypotheses. Optuna (`RUN_TUNING`) is optional and bounded, and off by default.
When enabled, it now runs before main training and applies the selected trial's
parameters to that training run. Selection uses validation only, never test.

Differences between the HUIM paper, the BlackPearl code and this adaptation are tabulated in notebook §1. In
summary, the "instant interest" features (exposure counts, slate size, in-slate ranks, session co-occurrence) are
**not** implemented. They depend on the recorded slate, on other users' test-period logs or on future totals.
History is **not** augmented with later behaviour clicks.

## Feature-availability audit (drives the input branches)

| Used (available for any eligible article at request time) | Excluded (reason) |
| --- | --- |
| Article metadata, body length, frozen text/image vectors (with masks) | `slate_size`, `slate_relative_age_h` — recorded-slate dependent, no catalogue equivalent (omitted from training *and* inference) |
| Article age = request time − `published_time` | `is_front_page` / current-page `article_id` — absent from official test behaviours; the shared test features mark every test row front page |
| premium × subscriber access | `total_inviews/pageviews/read_time`, `last_modified_time` — future aggregates |
| Category/type/topic preference crosses from fixed history; previously-clicked flag | Impression `read_time`/`scroll_percentage`, `next_*` — post-impression outcomes |
| Device, SSO, subscriber, hour, weekday; history size, median read time, mean scroll, 7-day clicks, hours since last click | In-view popularity, click counts, session co-occurrence; user/article ID embeddings |

The notebook's preflight compares our pair features with the shared Step 3 features on the recorded pairs.

On the attached mini fixture, this static inspection gave:

* history size and the category/type preferences match exactly;
* topic preference matches within 3e-8;
* 7-day clicks match within 4e-5;
* article age matches to about 0.03 h. This difference comes from float32 log round-trip, and future-published candidates are clipped to 0.

## Vector files

`data.read_vector_table` tries PyArrow first. If PyArrow cannot decode the column (for example, a nullable
fixed-size list), it falls back to DuckDB and then Polars, when installed.

* Every backend result is coerced to a concrete `pyarrow.Table` inside the reader. DuckDB 1.5.2's `.arrow()` returns a `RecordBatchReader`, which is read with `read_all()`.
* The only cast changes the list container to `FLOAT[]`, so float32 values, dimensions and null masks are preserved.
* The reader used, its raw return type, the source type and earlier errors are recorded. A file that no reader can decode raises; it is never treated as missing vectors or replaced by zeros.
* Notebook §3 checks all four original files (`vector_file_check.json`) before building inputs.
* Regression tests cover real DuckDB 1.5.2, a forced reader-type stub, Polars, and the all-readers-fail case.
  * The on-disk fixture uses portable nullable **variable-length** lists, because PyArrow 24 refuses to write nullable fixed-size lists ("Lists with non-zero length null components are not supported").
  * Fixed-size-list coercion is tested in memory, including through a `RecordBatchReader`.
  * Expected values, dimensions and the per-row null mask come from the source values, not from any reader.
  * Set `HUIM_DATA_ROOT` to run the original-artifact check.
* **Precedence (published_at_or_before_request_v1).** Sources are combined train_val then test, and the first source *record* per article ID wins, even when its vector is legitimately missing. A later snapshot never backfills it.
  * The report counts overridden rows, non-backfilled later vectors and overlap differences.
  * Flag/vector disagreements, wrong dimensions, non-finite values and duplicate IDs stop the run; they are not missingness.

The original full-data image files were not checked here: at 70 MB and 18.6 MB, they exceed the Drive connector's
download limit. The Colab preflight cell is the first check on them.

## Pool policy and request scopes (aligned with 3.1)

**Pool: `published_at_or_before_request_v1`, the same for train, validation and test requests.**

* Build one ordered union: `train_val/articles.parquet`, then `test/articles.parquet`. The first record per article ID is kept, including its missing fields; a later snapshot does not backfill a missing `published_time`.
* An article is eligible iff its `published_time` is known and `published_time <= impression_time`.
* There is no age or access cutoff and no embedding-coverage filter.
* Every selected inference request is scored against **all** its eligible articles, in batches. There are no sampled article negatives. The top `min(100, pool)` articles are kept, and an empty pool gives `[]`.
* Known clicks on ineligible articles stay in the Recall and ideal-DCG denominators.
* The pool fingerprint is a SHA-256 over the policy, sorted IDs and publication times. It is recorded in `request_scopes.json`, the selection protocol, `export_manifest.json` and `run_config.json`, and is re-checked by the reload check.
* Article-lookup coverage (history and recorded IDs outside the union) is reported per split.

**Request sampling (shared 3.1 rule).**

* `behaviour_id = f"{user_id}:{impression_id}"`; priority is `blake2b(f"42:{split}:{behaviour_id}", digest_size=8)`.
* The smallest N requests by (priority, `behaviour_id`) are selected and then processed by (`impression_time`, `user_id`, `impression_id`).
* Three separate limits apply: training, selection-validation and export per split. A smaller N is a prefix of the same order, so the 25 exported validation requests are a subset of the 100 selection requests.
* Requests come from the complete population of each split. A training request keeps its complete recorded slate, and fixed histories are not augmented.
* With `REFERENCE_31_RUN` set, 3.1's saved IDs are reused (`input_audit.json` `selection_requests`, `eligibility/{split}_requests.csv`), but only for a scope of the **same size**.
  * A different-size reference never overrides a limit, including `None` (all requests); it is recorded as `not_used`.
  * A reference with another sampling seed or pool policy raises an error.
  * Our own hash selection is always recorded, with `matches_own_hash_selection`. "Sampled" is computed from the actual scope sizes.
* A synthetic test and a check on the real mini feature tables both show identical IDs **and** processing order to 3.1's indexer, imported read-only.

**Metric scopes stay separate.** `metrics.json` has `selection_validation` (the selection requests, re-scored for
the selected checkpoint) and `exported_validation` (the exported validation requests, computed from the validated
shortlist files), as 3.1 names them. Each scope records:

* purpose, split, request count and limit;
* the IDs' SHA-256, plus the IDs file;
* the pool fingerprint;
* the selected model.

Compare models only on identical request IDs and pools.

**New protocol.** These pool and sample changes define selection protocol v2. Earlier v1 recovery checkpoints are
refused. A loaded v1 bundle is evaluated on the current union pool, but keeps its original selection provenance and
is labelled diagnostic.

## Export contract (v3) and validation

| File | Content |
| --- | --- |
| `predictions/{split}/submission_click_probabilities.csv` | `behaviour_id,article_id,score,click_probability`: every request × eligible article. `score` is the logit; `click_probability = sigmoid(score)`, uncalibrated |
| `rankings/{split}/submission_top_100.csv` | `behaviour_id,article_ids` (JSON array): all eligible scores sorted descending, ties broken by ascending `article_id`, keeping `min(100, pool)` |
| `diagnostics/recorded_slates/{split}_scores.csv` | every recorded candidate exactly once; ineligible recorded candidates are listed in `{split}_catalogue_ineligible.csv` |
| `diagnostics/catalogue/{split}_requests.csv` | per-request eligible count, shortlist length, clicked-article exclusions, Recall inputs, and the shortlist's vector share, median age and category spread |
| `diagnostics/catalogue/{split}_score_blocks.csv` | byte ranges of request blocks in the score CSV (used for validation and spot reads) |
| `export_manifest.json` / `run_config.json` | `export_schema_version = 3`, `prediction_pool`, eligibility rules, catalogue versions, `shortlist_size = 100`, score semantics, paths, counts, gate decisions |

The validators check the following:

* complete request coverage and the `behaviour_id` format;
* JSON parsing (never `eval`), shortlist length, uniqueness, and that no shortlisted article is future, missing or outside the catalogue;
* independently recomputed eligible counts;
* for written score files, re-reading sampled blocks (all blocks in the dev sample) to check that the scored IDs equal the eligible set and that the top 100 equals the full scores under the tie rule;
* `click_probability = sigmoid(score)`;
* recorded-slate scores equal catalogue scores for eligible recorded candidates.

## Union-pool budgets (measured in the notebook, not assumed)

The preflight computes **exact** eligible-pair counts for the planned scopes and for every agreed request. The
bounded throughput benchmark then refines bytes per row and scoring time.

The union's exact size depends on the train_val/test overlap: it is at most 125,541 + 20,738 = 146,279 articles,
and is likely about 125,541 if the train_val snapshot is a subset. I could not read the full catalogues here. The
upper bounds are therefore requests × 146,279:

| Scope | Requests | Upper bound on pairs | CSV at ~57 B/row |
| --- | ---: | ---: | ---: |
| smoke selection, per evaluation | 25 | ≤ 3.7 M | — (no CSV) |
| smoke export (5 per split) | 15 | ≤ 2.2 M | ≤ 0.13 GB |
| development selection, per evaluation | 100 | ≤ 14.6 M | — (no CSV) |
| development export (25 per split) | 75 | ≤ 11.0 M | ≤ 0.63 GB |
| all validation requests | 29,110 | ≤ 4.26 B | ≤ 243 GB |
| all test requests | 29,085 | ≤ 4.25 B | ≤ 242 GB |
| all train requests | ≈233 k (exact count printed by preflight) | ≤ 34 B | ≤ 1.9 TB |

Under the separate-catalogue policy, the reference was about 8.5 B pairs. The shared union raises the full-scope
total to about **40 B pairs** (≈2–2.4 TB of CSV), mostly because train and validation requests now also score
test-snapshot articles published before them.

Compute is about 0.2 M multiply-adds per pair. Storage and I/O are the blocker.

Guards:

* `max_export_pairs` (default 20 M) stops an export before it starts; it never shrinks the pool.
* `max_selection_pairs_per_evaluation` (50 M) bounds selection.
* The storage gate (free space ≥ 1.25 × estimate on the local and persistent roots) decides whether full score CSVs are written. A split that fails keeps complete top-100 files, which are re-validated against recomputed full scores, and the manifest records why scores are absent.

Full scope therefore needs an explicit `max_export_pairs` override, plus storage, or one of the group proposals:

1. Partitioned Parquet scores.
2. An agreed, named alternative pool policy, compared separately.
3. A shortlist-plus-manifest handoff amendment.

Exported results on a request sample are sampled development results, not full-split results or a submission.

## Colab execution

1. Open `experiment.ipynb` from main in Colab, on a GPU runtime.
2. Download [`huim_helpers.zip`](huim_helpers.zip) from this folder (use GitHub's
   **Download raw file** button) and drag it into Colab's Files panel (`/content`).
   Leave it zipped: the notebook unpacks and imports it automatically, with no
   GitHub token needed. Uploading the `huim/` source folder also works.
   Alternatively, use the existing Git checkout route with a Colab `GITHUB_TOKEN`
   secret that can read the private fork.
3. In the first code cell, set the following:
   * `DATA_ROOT`: the shared project folder or its shortcut. Source datasets are not modified.
   * `OUTPUT_ROOT`: defaults to `DATA_ROOT/Experiments/2.2_huim`, so runs are shared with the group. You need edit access to this experiment folder.
   * `REPO_BRANCH`: used only for the Git checkout route.
4. Run with `RUN_MODE = "smoke"` end to end. Then use `"development"`. Use `"full"` only after reading the preflight and refined budgets; it also needs `CONFIG_OVERRIDES = {"max_export_pairs": ...}`.

   | Setting | smoke (labelled) | development (aligned with 3.1) | full |
   | --- | --- | --- | --- |
   | training requests | 1,000 | 5,000 | all (`None`) |
   | selection-validation requests | 25 | 100 | 100 |
   | export requests per split | 5 | 25 | all (`None`) |
   | epochs / batch / selection interval | 1 / 1,024 / 4 steps | 6 / 1,024 / 20 steps | 4 / 4,096 / 300 steps |
   | request sampling seed | 42 | 42 | 42 |

   Optionally set `REFERENCE_31_RUN` to Max's run folder, to reuse its saved selection and export IDs.
5. **Resume.** Rerun with the same `RUN_ID`.
   * Training resumes from `PERSIST_DIR/recovery/last.pt`. It holds the model, optimiser, scheduler, GradScaler, progress and RNG states; the CPU RNG state is restored as a CPU tensor.
   * Export resume uses the progress file in the local run folder. Each resume is tied to a weights digest and a catalogue hash. A file shorter than its recorded progress raises an error and is never padded. To survive a lost runtime, set `LOCAL_ROOT` to a Drive folder for long full exports.
   * Optional bounded resume check: `RUN_INTERRUPTION_CHECK = True`.
6. **Load-only.** Set `TRAIN_MODEL = False` and `LOAD_RUN_DIR = <persisted run folder>`. Preprocessing and reconstruction settings come from the bundle, and nothing is refitted.

The run bundle is written to `PERSIST_DIR = OUTPUT_ROOT/<run_id>/`. It contains:

* the model files: `model/weights.pt`, `model_config.json`, `preprocessing.json`,
  `article_ids.npy`, `feature_dictionary.json` and a copy of the helper source in `model/code/huim/`;
* `run_config.json`, `metrics.json`, `export_manifest.json`, `requirements.txt` and `training_history.json`;
* `recovery/`, the exports, the logs, and `_COMPLETE_RUN.json`.

Every file is copied and verified by SHA-256. `_COMPLETE_RUN.json` is written last, and only after the reload
check from the persistent copy has passed. Load-only mode accepts only a valid marker, so a summary file alone
never marks a run complete. The selected weights are also stored inside the recovery checkpoint, so a crash
between the two saves cannot mislabel them.

`feature_dictionary.json` preserves ordered input names and categorical index-to-value
mappings. `preprocessing.json` preserves the fitted vocabularies, bins and scaling;
`article_ids.npy` preserves original article identity. Reuse these files on load,
rather than fitting new preprocessing or guessing feature labels from weights.
Reload checks include feature-name order as well as predictions. Embedding coordinates
are learned/frozen representation dimensions, not interpretable word coefficients.

## What to return for review

Return the executed `experiment.ipynb` and `review_bundle_<run_id>.zip`. The notebook saves the zip to
`PERSIST_DIR` and downloads it. The zip contains:

* `run_config.json`, `metrics.json`, `export_manifest.json`, `preflight.json` and `input_manifest.json`;
* `training_history.json`, `reload_check.json`, `interruption_check.json` (if run) and `requirements.txt`;
* `logs/run.log` and `logs/errors.log`;
* `controls/controls.csv` and `tuning/trials.csv` / `summary.json` (if run);
* the attention diagnostics, learning curves, `selection/request_scopes.json` (all scope IDs + pool fingerprint), `selection/selection_requests.json`, `vector_file_check.json`, the per-request catalogue manifests, the export progress files, and the ineligible-recorded-candidate lists.

Full data, score CSVs and model binaries are not needed for review.
