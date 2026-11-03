# 2.2_huim — HUIM-inspired Stage 1 relevance model and catalogue shortlist

> **Draft structure preview on main.** Follow-up fixes are pending for catalogue
> Recall@100 model selection and the full-file vector fallback reader. The
> [shared experiment contract](../README.md) defines the intended evaluation
> protocol; the AUC-based selection described below reflects this draft.

**Status:** the implementation is prepared and has passed static and synthetic checks. It has **not** been
executed on the shared data. No training, metric, export, persistence or reload result exists yet. The
executed Colab notebook and its review bundle will be the first evidence.

## What this experiment does

It implements Stage 1 only:

*eligible catalogue → **personalised top 100** → (Stage 2 reranker, not here) → final top 5*.

| Operation | Pool | Labels |
| --- | --- | --- |
| Training | Shared recorded pairs (`feature_engineering/{split}_features.parquet` keys + `clicked`) | clicked = 1, shown-unclicked = 0; no re-explosion, no catalogue negatives |
| Recorded-slate benchmark | Every recorded candidate, grouped by `(user_id, impression_id)` | mean impression AUC (primary, selection), nDCG@5 (secondary) |
| Catalogue prediction | Every article of the split's declared catalogue with `published_time <= impression_time` | unknown outside the recorded slate; known-click Recall@100 and coverage only |

Files:

* `experiment.ipynb`: the Colab notebook, using the common headings 1–9.
* `huim/`: helper modules.
  * `data`: inputs and checks.
  * `features`: request/article/history tensors and the feature audit.
  * `model`, `train`, `metrics`.
  * `catalogue`: eligibility, scoring and streamed v3 exports.
  * `validate`, `persist`.
  * `pipeline`: orchestration.
  * `tune`: optional Optuna search.
* `tests/test_static.py`: synthetic checks (no shared data, no optimiser steps).

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
* `ReduceLROnPlateau` on validation AUC, early stopping with patience 1, and at most 4 epochs.

These defaults are hypotheses. Optuna (`RUN_TUNING`) is optional and bounded, and off by default.

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

## Eligibility policy and catalogue versions (declared)

The policy is the same for every split:

* The article belongs to the split's declared catalogue snapshot:
  * `train` / `validation` → `train_val/articles.parquet`;
  * `test` → `test/articles.parquet` (overlapping IDs keep the train_val row, as in Step 3).
* `published_time` is present. Missing publication times are excluded and counted.
* `published_time <= impression_time`.

There is no recency window and no access/premium rule. The pool is not restricted to articles with vectors.
`run_config.json` records file SHA-256 hashes.

**Group decision needed:** the two snapshots differ greatly in size (20,738 vs 125,541 articles on Drive). Test
pools are therefore roughly six times larger and much older than train/validation pools. This is a distribution
shift for any model. Harmonising the catalogues, for example by using the 125,541-article snapshot for every split,
is a shared-policy change, not something this experiment should decide alone. `catalogue_policy` is configurable.

Requests are all behaviour rows, `(user_id, impression_id)`, including small recorded slates and the test
beyond-accuracy impressions. An empty pool produces `[]`.

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

## Full-catalogue feasibility (measured in the notebook, not assumed)

The preflight computes exact eligible-pair counts, and the dev export measures throughput and bytes per row. What I
could establish here comes from Drive metadata and the two small behaviour files I could download:

| Split | Requests | Catalogue | Upper bound on pairs (requests × catalogue) |
| --- | ---: | ---: | ---: |
| validation | 29,110 | 20,738 | ≤ 603.7 M |
| test | 29,085 (incl. 436 beyond-accuracy) | 125,541 | ≤ 3.65 B |
| train | not downloadable via the connector (10.3 MB file) | 20,738 | — |

The user's reference total is about 8.52 B pairs across all three splits. At about 57 bytes per CSV row, that is
roughly **450–490 GB** of full-score CSV.

**Compute.** The factorised model needs about 0.2 M multiply-adds per pair, so all splits come to roughly
3×10¹⁵ FLOP. That is minutes to a few hours on a T4, depending on memory-bound gathers. Compute is not the blocker.

**Storage and I/O are the blocker.** Standard Colab local disk and typical Drive quotas cannot hold these files.

The notebook therefore uses three modes:

* **smoke:** a small training subset and 1 epoch, with tiny request samples. It is labelled smoke.
* **development:** full training. It writes full scores and top 100 for a whole-request sample (≈1.3 GB) under `dev_sample/` (not a submission). It also writes top 100 for **every validation request**, so Recall@100 covers all validation requests, without the full validation score CSV.
* **full:** top 100 for every request of every split. Full score CSVs are written only for splits that pass a storage gate: free space ≥ 1.25 × measured estimate, checked on both the local and persistent roots. A split that fails keeps complete shortlists, and the manifest records that its scores are absent and why. The pool never changes.

Proposals for the group, if full score files remain infeasible. None is implemented here.

1. Accept the README's "compression or partitioned storage" route as the archival form of full scores. That would be partitioned Parquet with `article_id` int32 and `score` float32, at about 8 bytes per row and about 70 GB in total, with the documented loader. Keep CSV for validation only.
2. Agree a data-justified shared availability rule. An example is a maximum article age chosen from train click-age distributions, applied identically to all models. This changes the pool for everyone and needs agreement first.
3. Treat the top-100 files plus per-request manifests as the Stage 2 handoff. Keep full scores only for validation and request samples. This is an explicit contract amendment.

## Colab execution

1. Add a Colab secret `GITHUB_TOKEN` (a fine-grained token with read access to the fork) and enable it for the notebook.
2. Open `experiment.ipynb` from the branch or from GitHub in Colab, on a GPU runtime.
3. In the first code cell, set the following:
   * `DATA_ROOT`: the shared folder or its shortcut. It is read-only and is never written.
   * `OUTPUT_ROOT`: writable persistent storage. The default is `MyDrive/BT4222 Experiment Runs/2.2_huim`. Use `BT4222 Shared Final Folder/Experiments/2.2_huim` only if you can write there; the write probe checks this.
   * `REPO_BRANCH`: currently `claude/focused-shannon-s4lmma`.
4. Run with `RUN_MODE = "smoke"` end to end. Then use `"development"`. Use `"full"` only after reading the preflight and refined estimates.
5. **Resume.** Rerun with the same `RUN_ID`.
   * Training resumes from `PERSIST_DIR/recovery/last.pt`. It holds the model, optimiser, scheduler, GradScaler, progress and RNG states; the CPU RNG state is restored as a CPU tensor.
   * Export resume uses the progress file in the local run folder. Each resume is tied to a weights digest and a catalogue hash. A file shorter than its recorded progress raises an error and is never padded. To survive a lost runtime, set `LOCAL_ROOT` to a Drive folder for long full exports.
   * Optional bounded resume check: `RUN_INTERRUPTION_CHECK = True`.
6. **Load-only.** Set `TRAIN_MODEL = False` and `LOAD_RUN_DIR = <persisted run folder>`. Preprocessing and reconstruction settings come from the bundle, and nothing is refitted.

The run bundle is written to `PERSIST_DIR = OUTPUT_ROOT/<run_id>/`. It contains:

* the model files: `model/weights.pt`, `model_config.json`, `preprocessing.json` and `article_ids.npy`;
* `run_config.json`, `metrics.json`, `export_manifest.json`, `requirements.txt` and `training_history.json`;
* `recovery/`, the exports, the logs, and `_COMPLETE_RUN.json`.

Every file is copied and verified by SHA-256. `_COMPLETE_RUN.json` is written last, and only after the reload
check from the persistent copy has passed. Load-only mode accepts only a valid marker, so a summary file alone
never marks a run complete. The selected weights are also stored inside the recovery checkpoint, so a crash
between the two saves cannot mislabel them.

## What to return for review

Return the executed `experiment.ipynb` and `review_bundle_<run_id>.zip`. The notebook saves the zip to
`PERSIST_DIR` and downloads it. The zip contains:

* `run_config.json`, `metrics.json`, `export_manifest.json`, `preflight.json` and `input_manifest.json`;
* `training_history.json`, `reload_check.json`, `interruption_check.json` (if run) and `requirements.txt`;
* `logs/run.log` and `logs/errors.log`;
* `controls/controls.csv` and `tuning/trials.csv` / `summary.json` (if run);
* the attention diagnostics, the dev-sample request lists and manifests, the export progress files, and the ineligible-recorded-candidate lists.

Full data, score CSVs and model binaries are not needed for review.
