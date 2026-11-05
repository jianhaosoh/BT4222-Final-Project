# Multilingual DistilBERT similarity experiment

Open `experiment.ipynb` in Colab, enable a GPU, and upload this folder's
`catalogue_pipeline.py` when prompted. Set your shared Drive root.

The notebook generates frozen sentence embeddings with
`sentence-transformers/distiluse-base-multilingual-cased-v2`, a DistilBERT encoder,
then runs the same top-200 cosine retrieval as the Gemini experiment.
This experiment uses a sentence-embedding model, not plain BERT hidden states.

Defaults: title/subtitle/full body, latest 10 past clicks, seed 42,
1,000 validation requests, 1,000 validation exports, 25 train/test exports.
Output: `rankings/<split>/submission_top_200.csv` with request IDs, article-ID
lists and aligned cosine-similarity lists. No classifier fitting or fine-tuning.

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
