# Multilingual DistilBERT similarity experiment

Open `experiment.ipynb` in Colab, enable a GPU, and upload this folder's
`catalogue_pipeline.py` when prompted. Set your shared Drive root.

The notebook generates frozen sentence embeddings with
`sentence-transformers/distiluse-base-multilingual-cased-v2`, a DistilBERT encoder,
then runs the same top-200 cosine retrieval as the Gemini experiment.
This experiment uses a sentence-embedding model, not plain BERT hidden states.

Defaults: title/subtitle/full body, latest 10 past clicks, seed 42,
1,000 validation requests, 1,000 validation exports, 25 train/test exports.
Output: `rankings/<split>/submission_top_200.parquet` with request IDs and ordered
article-ID lists. Aligned scores are stored in `submission_top_200_scores.parquet`. No classifier fitting or fine-tuning.

Long articles are split into nonoverlapping model-token windows. Each window's
vector is normalized, then pooled by content-token count and normalized again.
The exact model revision, source fingerprints and text settings are saved.
Vectors and a resumable article cache are stored on Drive, independent of run IDs.
Changing source data requires a fresh embedding cache directory.

Use a fresh run ID for each history/configuration experiment. After audit,
set `RUN_PHASE = 'run'` in a separate cell and continue, preserving CONFIG/WORK.
Keep identical requests, eligibility and history settings for Gemini comparison.
Different chunking and pooling remain part of the encoder experiment.
A title/subtitle-only experiment should be compared with Gemini vectors generated
from those same fields if you want to isolate encoder differences.

The actual shared Drive embedding generation must be run in Colab. Local checks
use synthetic fixtures and do not establish recommendation quality or runtime.

Model documentation: https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2

## Separate text-variant notebooks

| Notebook | `TEXT_VARIANT` | Article text | Default experiment/cache ID |
| --- | --- | --- | --- |
| `bert_title_subtitle.ipynb` | `title_subtitle` | Title + subtitle | `3.2_pretrained_similarity_bert_title_subtitle` |
| `bert_title_subtitle_intro.ipynb` | `title_subtitle_intro` | Title + subtitle + first 1,000 stripped body characters | `3.2_pretrained_similarity_bert_title_subtitle_intro` |
| `bert_full.ipynb` | `title_subtitle_body` | Title + subtitle + full body | `3.2_pretrained_similarity_bert_full` |

All three share `catalogue_pipeline.py`, use latest-10 mean-profile cosine by
default, and retain 1,000 fixed validation requests/seed 42 for controlled text
comparison. `history_aggregation` can be changed explicitly for a separate
history experiment. Their outputs are separated by EXPERIMENT_ID and run ID.
Optional genuine held-out test evaluation is disabled until EVALUATE_TEST=True.

Matching embeddings already in each experiment's cache are reused. To reuse a
matching cache under a legacy folder name, change EMBEDDING_EXPERIMENT_ID to the
old ID without changing the output EXPERIMENT_ID. Inputs, encoder revision and
text variant still must match. A source-fingerprint change requires a fresh
cache or an explicit audited reuse of existing ID-matched vectors; do not silently
mix vectors generated from different article content.

## Native-list Parquet shortlist

The canonical output is `rankings/<split>/submission_top_200.parquet`, containing
exactly `behaviour_id` (string) and `article_ids` (native list<int64>). One row is
one request. The highest-ranked article is first; tie handling is descending
similarity, then ascending original article ID. Preserve empty lists for empty
eligible pools. The separate score Parquet retains aligned similarity scores.

All BERT notebooks now include a conversion-only cell for existing completed
CSV exports: `cp.export_parquet_submissions(WORK, persistent_location)`. Reload
the updated helper, keep WORK/RUN_ID pointing to the existing completed run, and
convert without embedding generation or catalogue scoring. The helper updates
the manifest and copies new files into the existing persistent bundle when its
path is supplied. For new runs, Parquet export is direct and verified against
independently recomputed full-catalogue rankings. The canonical Parquet has no score column;
the manifest identifies the separate score Parquet as the companion source for similarities.

## Reported validation results

Results below were reported from Colab, using frozen embeddings, an equal average
of the latest 10 available past-click vectors, cosine ranking, and a top-200
shortlist. They have not been independently reproduced against the shared Drive
data locally. Keep each run's `run_config.json` and exact request IDs with the
final report to verify input versions, sampling and eligibility alignment.

### Shared 1,000-request development comparison

Each reported sample contained 1,000 evaluated validation requests and 1,006 known
clicked articles. Recall and nDCG are averaged over requests with known clicks;
unshown outcomes remain unknown. The Gemini row is the shared-embedding reference,
not a DistilBERT run or an encoder-only controlled comparison.

| Encoder | Article text | History | Known-click Recall@200 | Observed-click nDCG@200 |
| --- | --- | --- | ---: | ---: |
| **Gemini reference** | Shared full-article embeddings | Latest 10 | **0.044** | **0.007222** |
| DistilBERT | **Title + subtitle** | Latest 10 | **0.005** | **0.001341** |
| DistilBERT | Title + subtitle + intro | Latest 10 | 0.003 | 0.000452 |
| DistilBERT | Title + subtitle + full body | Latest 10 | 0.002 | 0.000314 |

For the intro variant, the notebook defines intro as the first 1,000 characters
of the stripped body by default; confirm the actual value in its run manifest.
The table uses the corrected intro result of 0.003 recall and 0.000452 nDCG.

Gemini is strongest on the reported development sample. Title + subtitle is the
strongest DistilBERT text variant on that sample. This does not establish that
pretrained models generally fail, or that the differences are statistically
significant. Changing encoder also changes token limits and chunk pooling.

### Full-validation result: DistilBERT title + subtitle

| Validation scope | Evaluated requests | Known clicked articles | Known-click Recall@200 | Observed-click nDCG@200 |
| --- | ---: | ---: | ---: | ---: |
| Development sample | 1,000 | 1,006 | 0.005 | 0.001341 |
| **Full validation split** | **29,110** | **29,249** | **0.003263** | **0.000653** |

Full-validation recall is 0.3263%, compared with 0.5% on the smaller sample.
The full result is the broader estimate for this configuration. It reported no
requests without known clicks, no empty eligible pools, no unusable user profiles,
and no known clicks excluded by eligibility (100% known-click eligibility
coverage). There were 87 requests with ties crossing the top-200 boundary.

This run evaluated all 29,110 validation requests but exported only 1,000
validation shortlists. Consequently, its exported-validation metrics remained
0.005 recall and 0.001341 nDCG on those 1,000 requests. Export verification checked
123,961,830 eligible pairs and 200,000 shortlisted IDs. Train and test exports each
contained 25 requests and 5,000 shortlisted IDs. `handoff_complete` was false;
these files do not constitute an all-request handoff.

No full-validation Gemini result or genuine held-out test metric has been
reported here yet. Do not compare the full-validation BERT figure directly with
a sample-only Gemini figure as a final model comparison. Evaluate models on
identical requests/catalogues/eligibility policies, export the group's agreed
request scope, and select configurations using validation before inspecting test
performance. The separate `bert_full_with_test_metrics.ipynb` can evaluate saved
test shortlists against genuine held-out labels and write `test_metrics.json`.

See the [Gemini history comparison](../3.1_pretrained_similarity_gemini/README.md#gemini-history-comparison)
for the all-history/latest-5/latest-10/latest-20 experiment results.

## Batching chunks across articles

The embedding cell buffers 128 articles (`ARTICLE_BATCH_SIZE`) and feeds chunks
from multiple articles to the GPU in batches of 64 (`ENCODING_BATCH_SIZE`). Each
chunk retains its article index, so token-weighted pooling produces one vector
per article. Memory is bounded by the article buffer and chunk batch. Only the
current GPU batch is padded. Cache writes commit once per article buffer.

Try chunk batch 128 after measuring throughput and memory. Reduce it if CUDA
runs out of memory. Batch size changes processing speed and small floating-point
rounding effects, not text fields, token windows or the pooling definition.
Existing completed vectors are reused; partial caches resume at the last commit.
For a duplicated title/subtitle notebook, copy the updated configuration settings
and complete embedding-generation cell, retaining its EXPERIMENT_ID/TEXT_VARIANT.

Updated test files may contain a `clicked` column with values. The loader accepts
these files but excludes that column from test reads and stores null test outcomes
in its inference index. The audit records source-column presence and the ignore
policy. Test AUC/recall/nDCG are not computed by this inference pipeline. Validation
remains the basis for configuration selection. Re-upload the updated helper and
use a new run ID after an input release or loader-version change; saved article
embeddings remain reusable when their article sources and encoder are unchanged.

## Individual-history similarity experiment

`experiment_individual_history.ipynb` compares these retrieval rules with the same
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

### Direct Parquet export

New runs create no `submission_top_200.csv`. The two ranking files are:

- `submission_top_200.parquet`: `behaviour_id`, ordered native `article_ids`.
- `submission_top_200_scores.parquet`: the same request/ID lists plus aligned
  native `similarity_scores`.

The highest-ranked article remains first. Request eligibility audits may still
use CSV, but recommendation submissions and score companions are Parquet. Existing
legacy CSVs can be converted without rescoring; their source files are preserved.
Saved validation and optional test evaluation now read the Parquet recommendations.
