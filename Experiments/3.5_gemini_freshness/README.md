# Gemini top-3 similarity plus soft freshness

Open `gemini_freshness.ipynb` in Colab and upload BOTH `catalogue_pipeline.py` and
`freshness_pipeline.py` from this experiment folder. Set DRIVE_ROOT to the current
shared dataset and verify the exact Gemini embedding inputs. No new article
embeddings, fine-tuning or neural training are required. GPU similarity is used
when PyTorch/CUDA is available; CPU remains supported.

## Controlled comparison

Keep the same 1,000 validation request IDs, seed 42, smaller catalogue, latest-10
past-click history and top-3 history aggregation as the existing Gemini reference.
Compare:

| Formula | Meaning |
| --- | --- |
| similarity only, lambda=0 | Existing Gemini top-3 baseline |
| similarity - lambda * log1p(age_hours) | Soft age penalty, lambda in 0.01, 0.03, 0.1 |
| -log1p(age_hours) | Freshness-only control |

Age is measured at each request timestamp, never at notebook execution time.
All variants retain the same full time-eligible catalogue, including older news.
Known publication at/before the request remains required. Missing article vectors
or profiles have raw similarity zero, then receive the same age penalty. Unshown
outcomes remain unknown. Original positive recall denominators/nDCG ideals are
retained even when a click is publication-ineligible.

Similarities are calculated once per batch and reused for all grid options.
Validation Recall@100 selects the formula, then nDCG@100; exact ties keep the
unmodified baseline or earlier lower-weight option. Freshness-only can win.
Test labels never select weights. Validation has already informed development;
this is not an untouched evaluation. Any improvement belongs to the retrieval
score, not a new encoder or established retention benefit.

## Workflow

1. Run setup/configuration, then audit candidate/vector coverage and costs.
2. Switch RUN_PHASE to run in a separate cell, preserving RUN_ID/WORK.
3. Run the validation comparison and inspect similarity-only, freshness-only and
   all blends. The same request IDs are written into the comparison manifest.
4. Export the validation-selected formula. No separate business reranker is run.
5. Copy the completed run bundle to Drive. Generated SQLite indexes are omitted.

Default exports: train 25, validation 1,000, test 25 requests. Change these to the
team's agreed scope in a fresh prepared run and review costs. All-request scoring
can require substantially more time; request sampling never shrinks each
request's eligible article pool. The helper independently recomputes selected
shortlists to verify saved output. Exported-validation metrics are read from
those saved lists without a third catalogue-scoring pass.

## Files produced

- `freshness_grid.json`: source/config signatures, all variant metrics and selected rule.
- `freshness_comparison.csv`: compact comparison table.
- `validation_per_request.csv`: paired per-request metrics for all variants.
- `rankings/<split>/submission_top_200.csv`: top-200 IDs with aligned score/age lists.
- `run_config.json`: model/weight, catalogue eligibility, source versions, export scope.
- `metrics.json`: selected validation and exported-validation metrics; test remains null.
- `freshness_export.json`: independently verified export counts and output budget.

Shortlist CSV columns: `behaviour_id, article_ids, scores, similarity_scores,
article_age_hours, eligible_article_count, ties_crossing_100, ties_crossing_200`. Array columns are
JSON lists. `scores` contains FINAL ranking values, which can be below -1.
`similarity_scores` contains raw Gemini history cosines, which may not be sorted
once freshness is applied. They are not click probabilities. The shared evaluator
and reranker should consume `scores` or preserve the exported ranking, rather than
reordering by raw similarities. No exhaustive catalogue-score file is generated.

The notebook uses existing Gemini vectors. Exact encoder provenance must be
confirmed from the shared preprocessing metadata, not inferred from filenames.
Fitted neural weights, downstream engagement labels and post-request counts are
not introduced. The normal shared snapshot/provenance limitations remain.

The bundle is copied to `<DRIVE_ROOT>/Experiments/3.5_gemini_freshness/<run_id>/`.
Copy checks verify file sizes, not a saved neural model reload. For final test
metrics, join genuine held-out labels to the saved test lists only after fixing
the formula; test export itself ignores labels. Recorded-slate AUC is omitted.

Synthetic checks: `python -m unittest discover -s Experiments/3.5_gemini_freshness/tests -p 'test_*.py'`.
Actual recommendation quality and runtime must be measured on shared data in Colab.

Evaluation and weight selection use **K=100**. Shortlist exports retain 200 IDs
for the downstream reranker; metrics use only their first 100. Recall keeps the
original known-click denominator and nDCG uses min(100, known clicks) for its ideal.
Both boundary tie counts are stored in exports; selection metrics report ties
crossing rank 100. Re-upload freshness_pipeline.py and use a fresh run ID after
switching from the earlier K=200 experiment. The base catalogue helper is unchanged.
