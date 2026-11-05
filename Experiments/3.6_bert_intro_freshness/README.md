# BERT intro / maximum-history similarity plus freshness

Open `bert_intro_freshness.ipynb` in Colab and upload BOTH `catalogue_pipeline.py`
and `freshness_pipeline.py` from this folder. It reuses the existing frozen
DistilBERT title/subtitle/intro cache; there is no embedding generation, neural
training or fine-tuning. CPU works; CUDA is used for similarity when available.

## Default inputs and comparison

BERT_CACHE points to:
`<DRIVE_ROOT>/Experiments/3.2_pretrained_similarity_bert_title_subtitle_intro/embeddings/distiluse-base-multilingual-cased-v2_title_subtitle_intro_1000chars/`.
Change that path if the existing cache has another location. Metadata checks
ensure the encoder/text variant/dimension/completion flag match. Intro normally
means the first 1,000 stripped body characters, as recorded by the embedding run.
The existing cache's original model revision and pooling policy are preserved.

Use the same smaller shared catalogue, seed 42, latest 10 past-click events and
1,000 validation request IDs as the selected BERT intro experiment. Maximum-history
similarity uses the ONE highest cosine among available selected history vectors;
history_top_k is ignored. Select recent events before dropping missing vectors.
Repeated events retain their original history slots. Histories must predate the
entire behaviour split according to the existing fixed-snapshot rule.

| Variant | Score |
| --- | --- |
| Similarity only | max candidate-to-history cosine |
| Similarity + soft freshness | similarity - lambda * log1p(age_hours), lambda in 0.01, 0.03, 0.1 |
| Freshness only | -log1p(age_hours) |

Age is request timestamp minus publication time, never current notebook time.
All formulas rank the same full eligible catalogue: known publication time at
or before the request, without an extra age cutoff. Older articles are retained.
Missing vectors/profiles have raw similarity zero before any age penalty.
Known-click eligibility exclusions remain in metric denominators/ideals; unshown
outcomes remain unknown. Existing article-snapshot provenance limitations apply.

## Evaluation, selection and outputs

Selection maximises macro **Recall@100**, then **observed-click nDCG@100**. Exact
ties prefer the earlier baseline/lower-weight option. Similarities are computed
once per batch and reused across the grid. Freshness-only can win; do not assume
a blend is better. Test click labels never select the weight. Validation has
already informed development; it is not an untouched holdout.

Exports retain **top 200**, one request per row, highest-ranked article first:

- `rankings/<split>/submission_top_200.parquet`: `behaviour_id` string and native
  ordered `article_ids` list<int64>, exactly two columns.
- `rankings/<split>/submission_top_200_scores.parquet`: same IDs/order plus FINAL
  `scores`, raw `similarity_scores`, `article_age_hours`, pool and tie metadata.
- `freshness_grid.json`, `freshness_comparison.csv`, `validation_per_request.csv`:
  all controls/weights and paired validation results.
- `run_config.json`, `metrics.json`, `freshness_export.json`: policy, sources,
  selected formula, score location, coverage, export scope and replay checks.

Final scores can be below -1 and are not click probabilities. The evaluator and
reranker must preserve the ordered IDs or use FINAL `scores` from the companion,
not accidentally reorder by raw similarities. No submission CSV or exhaustive
catalogue-score table is generated. Comparison/per-request diagnostic CSVs are
small experiment summaries, not recommendation submissions.

Each saved shortlist/score companion is independently rescored for verification.
Exported validation metrics use the saved Parquet without an extra scoring pass.
Default exports: train 25, validation 1,000, test 25 requests. A full validation/test
export may require over one billion candidate pairs; inspect the audit and raise
the declared pair budget only after checking feasibility. Use a fresh run ID when
changing config/scope. Matching existing article vectors are reused throughout.

Copy the completed bundle to
`<DRIVE_ROOT>/Experiments/3.6_bert_intro_freshness/<run_id>/`. SQLite indexes are
excluded; copy checks verify file sizes. Final test metrics require a separate
held-out evaluator after fixing the formula. No test outcome metrics or AUC are
computed by this notebook, and no Stage 2 business reranker is applied here.

This compares a retrieval scoring extension, not improved embedding generation.
Comparing it with Gemini top-3 also changes the history rule; keep request IDs,
catalogue policy and metric cutoff fixed and describe that distinction.

Synthetic checks: `python -m unittest discover -s Experiments/3.6_bert_intro_freshness/tests -p 'test_*.py'`.
Actual data performance and runtime must be measured in Colab.
