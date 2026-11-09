# 3.1 Gemini embedding similarity

Stage 1 retrieval: average a user's past-click embeddings, normalise the profile,
then rank the time-eligible article catalogue by cosine similarity and keep the
best 200. There is no classifier training, fine-tuning or business re-ranker.
The separate Stage 2 owner consumes the shortlist to produce the final five.

## Run

1. Open `gemini_experiment.ipynb` in Colab and upload `catalogue_pipeline.py` when prompted.
2. Set your actual shared Drive path. Start with `RUN_PHASE = "audit"` to check
   input coverage and catalogue scoring/export costs.
3. Switch to `RUN_PHASE = "run"` after the audit and continue through the notebook.
   Use a new run ID for changed sources, seed or profile settings.

The notebook retains the group's nine-section structure. The helper remains one
file so Colab requires only one Python upload.

## Files

| File | Purpose |
| --- | --- |
| `gemini_experiment.ipynb` | Configuration, audit, validation, export and handoff |
| `gemini_individual_history.ipynb` | Individual-history similarity comparison and exports |
| `catalogue_pipeline.py` | Loading, user profiles, cosine scoring and export checks |
| `requirements.txt` | CPU dependencies |
| `requirements-gpu.txt` | Optional PyTorch dependency for CUDA similarity |
| `tests/test_similarity.py` | Synthetic correctness and reload checks |
| `outputs/<run_id>/` | Generated local runs, ignored by Git |

## Settings to change

Edit the notebook configuration, rather than the helper:

| Setting | Purpose |
| --- | --- |
| `DRIVE_ROOT` | Location of the shared folder in your account |
| `history_limit` | `None` for all available past clicks; an integer for latest N |
| `validation_requests` | Fixed validation sample; 100 is a pilot, increase after timing |
| `export_requests` | Per-split export limits; `None` means all requests |
| `similarity_device` | `auto`, `cpu`, or required `cuda` |
| `scope` | `development` for samples; `final` requires complete split exports |

Confirm the shared embedding encoder/revision with the preprocessing owner.
Embeddings are loaded, not generated. Shared feature tables provide only original
IDs, timestamps and evaluation labels; engineered predictors are not used.
History must predate the split. Eligibility requires known publication time no
later than the request, with no extra age cutoff. Missing vectors stay eligible
with score zero; missing profiles, coverage and ties are reported.

## Gemini history comparison

User-reported Colab validation results using the shared Gemini embeddings and
candidate-to-user cosine similarity, with a top-200 shortlist. Each run evaluated
1,000 validation requests containing 1,006 known clicked articles. The history
limit changes the number of most recent past clicks averaged into the user profile;
`None` uses all available past clicks.

| History setting | `history_limit` | Known-click Recall@200 | Observed-click nDCG@200 |
| --- | --- | ---: | ---: |
| All available clicks | `None` | 0.039 | 0.006027 |
| Latest 5 clicks | `5` | 0.036 | 0.006168 |
| **Latest 10 clicks** | **`10`** | **0.044** | **0.007222** |
| Latest 20 clicks | `20` | 0.037 | 0.006218 |

Latest 10 is the strongest configuration on this development sample for both
metrics. Compared with all history, recall improves by 0.005 (0.5 percentage
points, or 12.8% relative), and nDCG improves by approximately 19.8% relative.
This suggests a useful balance between recent interests and profile coverage;
it does not establish the cause of the difference or statistical significance.
Keep latest 10 as the current Gemini reference for the DistilBERT comparison,
using identical request IDs, eligibility rules and cutoff.

For the latest-20 run, evaluation and exported-validation metrics agree. All
1,000 requests had known clicks, there were no empty eligible pools, and no known
clicks were excluded by eligibility (100% eligibility coverage). No requests had
ties crossing the top-200 boundary. Validation export verification rescored
123,961,830 eligible request-article pairs and verified 200,000 shortlisted IDs
across 1,000 rows. Train and test exports each covered 25 requests and 5,000 IDs;
no exhaustive full-score rows were exported.

These are sampled development results, not an untouched test evaluation.
`handoff_complete` was false for these sampled runs. The values above were
reported from Colab; they have not been independently reproduced on local Drive
data, and per-run IDs/source manifests should accompany the final project report.

## Comparison with DistilBERT

The corrected DistilBERT text-variant sample results and its title/subtitle
full-validation result are recorded in the
[BERT experiment README](../3.2_pretrained_similarity_bert/README.md#reported-validation-results).
On the reported 1,000-request comparison, the Gemini latest-10 reference achieved
0.044 Recall@200 and 0.007222 nDCG@200, while the strongest DistilBERT variant,
title + subtitle, achieved 0.005 and 0.001341 respectively. DistilBERT title/subtitle
subsequently achieved 0.003263 recall and 0.000653 nDCG on all 29,110 validation
requests. A full-validation Gemini result is still needed for a comparison on
that full scope. All figures are user-reported Colab results.

## Results and handoff

Rank configurations by validation **known-click Recall@200**, then
**observed-click nDCG@200**, on identical requests and full eligible catalogues.
Recorded-slate AUC is diagnostic. Unshown outcomes remain unknown; test has no
outcome metrics. Observed clicks do not establish retention gains.

Each run writes:

- `rankings/{split}/submission_top_200.parquet`: one row per `behaviour_id`, with a native ordered `article_ids` integer list.
- `rankings/{split}/submission_top_200_scores.parquet`: companion file with ordered IDs and aligned similarity scores.
- `eligibility/{split}_requests.csv`: per-request eligibility and coverage audit.
- Model JSON, metrics, source/configuration manifests and reload checks.

All eligible articles are scored in memory, but only the ordered top-200 IDs and their similarity scores are exported.
The shortlist is independently recomputed to verify the saved CSV. Metrics include
recorded-slate AUC without exporting individual recorded-candidate scores.
This top-200-only bundle deliberately omits the original shared contract's exhaustive
score files; the manifest records that difference. Use a fresh run ID after this update.

Scores are cosine similarities, not probabilities. The default run is sampled
development, not a final submission. Completed bundles are copied to the declared
shared Drive destination and reloaded without fitting; infeasible exports or
incomplete persistence are reported. See the notebook for detailed rules.

## Check the implementation

From this experiment folder:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

From the repository root:

```bash
python -m unittest discover -s Experiments/3.1_pretrained_similarity_gemini/tests -p 'test_*.py'
```

These checks use synthetic fixtures. Actual shared-data results reported from
Colab are recorded above; local checks do not independently reproduce them.
CUDA hardware checks are skipped when no CUDA device exists.

Earlier hybrid source is archived locally at `Pretrained Models/hybrid_reference/`,
which is ignored by Git. It is separate from the active similarity-only experiment;
its checkpoints and probability outputs are not used here.

The score Parquet has `behaviour_id,article_ids,similarity_scores`. Both lists
are native arrays of equal length; each score belongs to the ID at the same position.
Scores are in [-1, 1], with zero as the existing missing-vector/profile fallback.
Use a new run ID and re-upload the helper after this schema update.

Updated test files may contain a `clicked` column with values. The loader accepts
these files but excludes that column from test reads and stores null test outcomes
in its inference index. The audit records source-column presence and the ignore
policy. Test AUC/recall/nDCG are not computed by this inference pipeline. Validation
remains the basis for configuration selection. Re-upload the updated helper and
use a new run ID after an input release or loader-version change; saved article
embeddings remain reusable when their article sources and encoder are unchanged.

## Individual-history similarity experiment

`gemini_individual_history.ipynb` compares these retrieval rules with the same
latest-10 history and request sample:

| `history_aggregation` | Score |
| --- | --- |
| `mean_profile` | Original cosine to the normalized average history vector |
| `max_history` | Maximum candidate-to-history-article cosine |
| `topk_history` | Mean of the largest 3 candidate-to-history cosines |

Set `history_top_k=3` for the last method; fewer available events use their actual
count. Select the latest events before dropping missing vectors. Repeated clicks
retain repeated weight/events. Missing candidates or histories get zero. Individual
methods require a positive bounded `history_limit` (default 10). Metadata, labels
and availability flags do not affect ranking. GPU and CPU paths use the same rule.

Start on the same 1,000 validation request IDs. Use a fresh run ID for each method,
then compare Recall@200 and nDCG@200 with the matching mean-profile baseline. Saved
article embeddings remain reusable. Source/versioned history caches, score exports,
recorded-slate diagnostics and reload probes all use the declared aggregation.
More cosine comparisons are required, so benchmark before full validation. The
pair budget counts candidate pairs, not the expanded number of history dot products.
No recommendation quality improvement is claimed before running these experiments.

## Individual-history Parquet handoff

The canonical shortlist is now `rankings/<split>/submission_top_200.parquet` with
exactly two columns: `behaviour_id` (string) and `article_ids` (native list<int64>).
Each request has one row. List order is recommendation order, not article-ID
order, and is preserved exactly from scoring. Lists contain up to 200 unique IDs;
empty eligible pools retain an empty list. This applies to all three history
aggregation choices; it does not change ranking or metrics.

New runs write both the Parquet handoff and a separate score-bearing Parquet companion.
Existing completed runs can call `cp.export_parquet_submissions(WORK, persistent)`
to convert saved CSVs and copy the new files/manifest to the existing persistent
run folder. This requires no new embeddings or catalogue scoring. The notebook
includes the conversion-only cell. Keep WORK pointing to the existing run for
this format-only conversion; do not accidentally generate a new RUN_ID first.

### Direct Parquet export

New runs create no `submission_top_200.csv`. The two ranking files are:

- `submission_top_200.parquet`: `behaviour_id`, ordered native `article_ids`.
- `submission_top_200_scores.parquet`: the same request/ID lists plus aligned
  native `similarity_scores`.

The highest-ranked article remains first. Request eligibility audits may still
use CSV, but recommendation submissions and score companions are Parquet. Existing
legacy CSVs can be converted without rescoring; their source files are preserved.
Saved validation and optional test evaluation now read the Parquet recommendations.
