# 3.1 Pretrained similarity - catalogue shortlist experiment

Provisional next unused pretrained-track ID: confirm `3.1_pretrained_similarity`
with the group. This is a new shared-input implementation. Earlier experiments
in `Pretrained Models/` remain recorded-slate diagnostic references and are not
renamed, overwritten, or claimed to satisfy catalogue inference.

## Open in Colab

1. Open `experiment.ipynb` in Colab and mount the shared Drive.
2. Make `catalogue_pipeline.py` available alongside the notebook/checkout. If
   opening the notebook alone from GitHub, its setup prompts for that helper file.
3. Set `DRIVE_ROOT` in the configuration cell. Confirm input paths using `paths`
   overrides when the actual Drive layout differs. Set your owner and the shared
   preprocessing release and encoder identity when known. Do not assume the
   embeddings are E5/OpenAI/Gemini merely from old code or a filename.
4. Keep `RUN_PHASE='audit'` initially. Run sections 1-4 to inspect real schemas,
   vector decoding, counts, source provenance and sampled/full catalogue costs.
5. Set `RUN_PHASE='run'` to train a development experiment after inspecting the
   budgets. Follow all remaining sections for validation, exports, persistence
   and reload. Outputs are local under `outputs/<run_id>/`; a completed copy goes
   to `BT4222 Shared Final Folder/Experiments/<experiment_id>/<run_id>/`.
6. To reuse a saved model in a new runtime, use a new run ID, set
   `TRAIN_MODEL=False`, and point `LOAD_RUN` to the completed persistent source
   run. The notebook checks source versions and reconstructs the same features.

No API embedding generation or paid service calls are used. The supplied fixed
text embedding tables are decoded and validated. Encoder identity/revision is
recorded from configuration and must be confirmed with the preprocessing owner
before final scope. The pretrained encoder stays frozen.

## Hypothesis and model

Personal text similarity should help recover a reader's observed clicked articles
from the whole eligible catalogue for a top-100 shortlist, before Stage 2 business
reranking. Train a logistic classifier with `SGDClassifier(loss='log_loss')` in
batches. This deliberately changes the old in-memory classifier implementation
so full shared training can stream without loading a giant dense matrix.

The model consumes existing shared candidate rows directly: no re-explosion,
negative catalogue construction, or test-target filling. It selects a catalogue-
safe raw subset, retraining from scratch rather than reusing incompatible old
weights. The 13 shared source columns are:

- Request: `hour`, `day_of_week`, `is_weekend`, `is_subscriber`, `is_sso_user`,
  `is_front_page`, `history_size`.
- Article: `premium`, `body_word_count`, `topic_count`, `has_topics`,
  `article_age_h`, `article_age_missing`.

Four new features are candidate-to-normalized-mean-history text cosine,
candidate-vector missingness, profile missingness and history vector coverage.
All other engineered fields are intentionally excluded, especially
`slate_size` and `slate_relative_age_h`, which depend on the original pool.
Every used article feature is reconstructed and audited against the shared raw
columns across all three splits. Missing/reconstruction failures are not silently
replaced by zeros. Train-only mean imputation (with explicit missingness indicators)
and scaling are fitted by streaming passes; all-missing training fields use zero
plus a flag. IDs and targets are not model features.

Raw fixed history ID/time arrays are additional declared shared inputs. They must
precede the split's first request, matching shared preprocessing. Repeated clicks
retain weight. Fixed history is not updated from behavior labels. Profiles live
in a disk-backed lookup with a bounded cache; article vectors are stored once.
This is a content representation experiment, not fine-tuning the text encoder.

## Selection protocol

The predeclared search compares two `alpha` values over at most three epochs each.
Each epoch evaluates the same hash-selected validation requests against their
**entire eligible catalogue**, with no sampled article negatives. Select by:

1. Mean catalogue known-click Recall@100.
2. Mean catalogue observed-click nDCG@100.
3. Earlier evaluation/checkpoint.

Patience counts these selection evaluations. Save each attempted outcome, training
loss and selection metrics under `tuning/`. Failed runs are recorded. Best inference
checkpoints are copied to a Drive `_in_progress` folder during training. This is
not a complete handoff bundle or a verified exact training-resumption mechanism.
Load-only runs restore the selected model, not the final epoch by default.

Recall denominators retain original known clicks excluded by eligibility. Observed
nDCG ideals likewise retain all original clicks, capped at 100. Empty pools get
zero retrieval metrics when clicks are known; requests without known clicks are
reported separately. Eligible-only recall and click eligibility coverage are
additional diagnostics. Test has no outcome metrics. Recorded-slate AUC is macro-
averaged by the composite user/impression key and is diagnostic, never a selection
substitute. Ties use descending score then ascending original article ID, and
requests with ties across rank 100 are counted.

## Pool and inference contract

The proposed, explicitly named policy `published_at_or_before_request_v1` uses the
ordered union of declared train/validation and test catalogue snapshots, first
snapshot per article ID retained, consistent with the shared feature generator.
Only articles with a known `published_time <= impression_time` are eligible.
There is no age/access cutoff. Confirm this catalogue release/policy matches the
group's other models before comparison. Snapshot availability is not proof of an
unknown production catalogue's coverage. Unknown publication dates are excluded
because availability is unestablished; their known clicks remain in metric ideals.

Legitimately missing text vectors keep the article eligible, with zero cosine
and explicit flags. Malformed/unreadable vectors fail the run; they are not
silently treated as missing. Overlapping embedding rows keep the first snapshot,
including legitimate missing values. Catalogue and embedding coverage are reported
separately. Image embeddings are not used in this experiment.

## Costs and completeness

Default development limits are 5,000 complete training requests, 100 fixed
validation-selection requests and 25 export requests per split. Request sampling
uses a seeded hash of the original composite key and keeps every recorded
candidate. Every sampled request is scored against its full eligible catalogue.
These outputs are **sampled development**, not complete final submissions.

The initial audit estimates disk-index size, exact eligible scoring-pair counts,
empty pools and CSV size (130 bytes/pair planning assumption), both for the
selected export sample and every agreed request. The later scoring-time estimate
uses measured catalogue selection, explicitly excluding CSV I/O, indexing and
persistence overhead. Pair, storage and time limits stop an infeasible export.
Batching lowers memory use but does not remove billions of rows' compute/disk cost.

Final scope requires all training requests, all export requests (`None` limits),
owner, input release and actual encoder identity. Inspect feasibility before
raising limits. Infeasible/missing persistent copies are recorded as incomplete;
do not rename recorded-slate CSVs as catalogue outputs or call samples full runs.
Metadata source paths, file sizes/mtime/schema are recorded; optional full input
SHA256 hashing is configurable and may take time on Drive. A size/mtime fingerprint
is explicitly not a content hash.

## Output bundle

```text
outputs/<run_id>/
  predictions/{train,validation,test}/submission_click_probabilities.csv
  rankings/{train,validation,test}/submission_top_100.csv
  diagnostics/recorded_slates/{train,validation,test}_scores.csv
  eligibility/{train,validation,test}_requests.csv
  model/selected.joblib
  model/model_config.json
  model/catalogue_pipeline.py
  model/coefficients.csv
  model/reload_probe.npz
  tuning/trials.json
  tuning/trials.csv
  metrics.json
  run_config.json
  requirements.txt
  input_audit.json
  validation_explanation.json
  reload_check.json
```

Temporary disk indexes stay local and are excluded from the persistent bundle;
source versions, lookup construction code and selected request IDs are preserved.
`selected.joblib` contains the classifier, fitted scaler and imputation means.
`run_config.json` declares schema version 3, original-ID composite keys, logit
score semantics, uncalibrated sigmoid click probabilities, catalogue policy,
feature order, source/candidate coverage, sample scope and persistence status.
Probabilities describe a logged-click model, not unbiased catalogue-wide CTR.

All six catalogue score/ranking files and three recorded-slate diagnostics are
checked. The full score export must exactly cover each request's eligible IDs,
with unique top-100 membership/order and `[]` for empty pools. CSVs contain no
labels. A saved explained validation ranking decomposes the linear logit; it is
not causal feature importance. Training predictions are in-sample, not OOF inputs
for stacking. Unknown outcomes must never become classification negatives.

Persistence copies the completed run to shared Drive, verifies file coverage and
sizes, reloads the saved classifier/preprocessing and compares a raw-feature probe
without fitting (`rtol=1e-7`, `atol=1e-8`). Exact shared-source feature reconstruction
is audited separately. The load-only notebook path can reconstruct/export using
the same versioned shared sources. Cross-model tables and Stage 2 trade-offs belong
in shared evaluation. Offline observed clicks do not establish retention gains.

## Verification and current status

Run locally from this folder:

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_catalogue_pipeline.py
```

Synthetic tests cover composite IDs, duplicate rejection, test labels, vector
failures, first-snapshot missingness, future exclusions/history cutoffs, exact
rank-100 ties, original-click denominators, empty pools, all split exports,
corruption detection, persistence, reload and a fresh load-only run without fitting.
These are implementation checks, not measured project scores.

**The actual shared Drive data is not accessible in this local task. Real-data
schema compatibility, runtime/storage feasibility, Recall@100, complete exports
and the real persistent reload remain unverified until the Colab notebook runs.**
