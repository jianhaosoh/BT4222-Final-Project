# BT4222 Evaluation and reranking

This folder contains the shared Stage 1 evaluator, Stage 2 reranker, weighted RRF generator, comparison utilities, and Colab workflow. The evaluator consumes saved model rankings; it does not train the LSTM, XGBoost, or pretrained models. Installable packages are listed in `requirements.txt`.

## Maintained Colab workflow

Use these current notebooks; older copies in the project may have stale paths or split behavior.

| Notebook | Role | Split inputs and outputs |
|---|---|---|
| `Shared_Model_Evaluation.ipynb` | Stage 1 evaluation. Current configuration includes LSTM, XGBoost, Gemini, and 75:25 LSTM:XGBoost RRF. | Evaluates validation and test separately. Writes reports, per-request tables, top-item exports, and a summary CSV to `Evaluation/Stage 1/validation/` or `Evaluation/Stage 1/test/`. |
| `RRF_Ensemble_Generation.ipynb` | Fixed weighted RRF generation from selected LSTM and XGBoost runs. | Reads each split's top-200 rankings; produces its own top-200 RRF ranking and manifest under `Evaluation/ensemble/validation/` and `Evaluation/ensemble/test/`. Weights are 75:25 and the rank constant is 60. |
| `Stage2_Reranker.ipynb` | Stage 2 validation comparison across LSTM, XGBoost, Gemini, and RRF; final test run on RRF only. | Validation reads validation labels/rankings for all four configured models. Test reads test labels and the test RRF ranking only. Results go to `Evaluation/Stage 2 - Reranking/validation/` and `/test/`; shared diversity constants go to `/shared/`. |
| `Reranker_Weight_Optimization.ipynb` | Optional search for freshness weight and MMR lambda. | Validation only, with sample search and full-validation confirmation. Writes to `Evaluation/Stage 2 - Reranking/weights/`. Never tune on test. |

### Colab setup and sequence

The notebooks use this Drive root:

```text
/content/drive/MyDrive/Colab Notebooks/BT4222 Shared Final Folder
```

Copy the shared Python modules to `Evaluation/code/` in that Drive folder:
`evaluate.py`, `build_catalogue_union.py`, `rerank.py`, `similarity.py`,
`compare_models.py`, and `optimize_reranker.py` when running the optimizer.
The notebooks mount Drive, install their small dependency set, validate input
paths, and write outputs under the same project's `Evaluation/` folder.

1. Run the catalogue-building cell in the Stage 1 notebook. It uses
   `train_val/articles.parquet` and prefers the corrected
   `test/articles (1).parquet`, falling back to `test/articles.parquet`. Verify
   the printed paths and adjacent manifest under `Evaluation/catalogue/`.
2. If RRF is part of the evaluation, run the 75:25 RRF notebook. It requires
   full top-200 LSTM and XGBoost exports for both splits and produces separate
   validation and test ensembles.
3. Run Stage 1 evaluation for cross-model retrieval metrics. Use validation
   for model selection. Freeze the selected model/ensemble before interpreting
   test results.
4. Run any reranker weight search on validation only. Record the search and
   guard, then freeze the weights.
5. Run the Stage 2 notebook. It compares all four configured models on
   validation and reranks only the chosen 75:25 RRF on test. Test is final
   evaluation only; do not use it to change the model, ensemble ratio, or
   reranker.

The labelled project `test` is a held-out user cohort from the original
validation period, not a later-week temporal test. It is separate from the
official EB-NeRD challenge test, which has no click labels. State this scope in
the report.

## Inputs, metrics, and interpretation

- A ranking file has one row per request with `behaviour_id` and an ordered
  `article_ids` list. The list is a ranked shortlist of up to 200 eligible
  catalogue articles. Use the same request IDs, catalogue, and publication
  eligibility policy across models within a split.
- Stage 1 reports known-click Recall@100/200 and observed-click nDCG@100/200,
  with age, topic-diversity, and category diagnostics. Recall's denominator
  includes all known clicked articles, even if catalogue eligibility excludes
  one; eligible-only recall and eligibility coverage are reported separately.
- AUC requires scores for every originally displayed candidate and is a
  recorded-slate diagnostic. It cannot be calculated from top-200 retrieval
  rankings or from reranked top-five lists. If full recorded-slate scores are
  absent, do not claim AUC was measured.
- Stage 2 reorders only the Stage 1 top 200 and returns five items. Evaluate
  the Stage 1 top-five prefix and the Stage 2 top five with the same @5
  relevance definitions and same requests. It cannot retrieve an article
  Stage 1 left outside its shortlist.
- Current Stage 2 defaults are `A0_HOURS=0`, half-life 24 hours,
  `FRESHNESS_WEIGHT=0`, `CONTINUITY_WEIGHT=0`, and `MMR_LAMBDA=0.9`.
  Freshness-only is therefore a no-op, and freshness-plus-diversity is the
  same as diversity-only until `FRESHNESS_WEIGHT` is deliberately set above
  zero. Record the actual values used; call them exploratory/untuned unless
  validation tuning was performed.
- Session continuity is disabled in the maintained Stage 2 run. The code can
  consume external candidate continuation scores, but it does not infer a
  counterfactual outcome for an unclicked article. Any continuity analysis
  must disclose click-conditioned selection and observation limits.

## Local execution

From the repository root:

```powershell
python -m pip install -r "Evaluation/requirements.txt"
```

The required Python packages are NumPy, pandas, and PyArrow. The Colab
notebooks use Colab's built-in Drive integration. Local CLI arguments are
documented in `CONTRACT.md`; update file paths to match the local checkout.
