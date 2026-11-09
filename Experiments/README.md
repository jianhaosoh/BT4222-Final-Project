# Stage 1 Retrieval Experiments

## Overview

Stage 1 candidate generation forms the retrieval backbone of the two-stage news recommendation framework. Operating over the Ekstra Bladet News Recommendation Dataset (EB-NeRD Small), each model evaluates the entire catalogue of 20,738 articles for every user impression request under strict point-in-time publication constraints (`published_time <= impression_time`). On the validation partition, this full-catalogue retrieval protocol scores over 566 million candidate pairs across 29,110 impression requests to generate a high-recall top-200 candidate shortlist per request for downstream ensembling and Stage 2 diversity reranking.

---

## Directory Structure and Model Families

The directory contains experimental notebooks, configurations, and documentation across three model families:

```
Experiments/
├── 1.1_lightgbm_ranker/              # LightGBM GBDT ranker (31 leaves, 75 trees)
├── 1.2_logistic_regression/          # L2-regularized linear baseline (C = 0.01)
├── 1.3_xgboost/                      # XGBoost GBDT ranker (31 leaves, 25 trees)
├── 2.1_lstm/                         # Target-Aware LSTM with soft freshness decay (lambda = 2.0)
├── 2.2_huim/                         # Deep Interest Network (DIN) with parallel event attention
├── 3.1_pretrained_similarity_gemini/ # Frozen Gemini embedding similarity baseline
├── 3.2_pretrained_similarity_bert/   # Frozen Danish DistilBERT text similarity baseline
├── 3.3_pretrained_retrieval/         # Contrastive retrieval pipeline
├── 3.4_finetuned_distilbert/         # Domain fine-tuned DistilBERT bi-encoder
├── 3.5_gemini_freshness/             # Gemini semantic similarity with freshness penalties
├── 3.6_bert_intro_freshness/         # DistilBERT intro similarity with freshness penalties
├── CONTRIBUTING.md                   # Experiment standardization contracts and development guide
└── README.md                         # This navigation guide
```

### Family 1: Point-in-Time Tabular Rankers

Tabular rankers are trained on 202 engineered features spanning user dwell times, category-level click-through rates, rolling affinities, and article recency metrics.

- [`1.1_lightgbm_ranker/`](1.1_lightgbm_ranker/): LightGBM Gradient Boosted Decision Tree (GBDT) ranker configured with 31 leaves and 75 trees.
  - Validation Recall@100: 0.828
  - Validation Recall@200: 0.907
  - Validation nDCG@100: 0.217
  - Validation nDCG@200: 0.228
- [`1.2_logistic_regression/`](1.2_logistic_regression/): L2-regularized linear baseline with inverse regularization parameter C = 0.01.
  - Validation Recall@100: 0.769
  - Validation Recall@200: 0.915
  - Validation nDCG@100: 0.175
  - Validation nDCG@200: 0.196
- [`1.3_xgboost/`](1.3_xgboost/): XGBoost GBDT ranker configured with 31 leaves and 25 trees. Tree SHAP feature attributions confirm that publication recency and short-term category affinity dominate ranking decisions. Selected for the Stage 1 final ensemble.
  - Validation Recall@100: 0.858
  - Validation Recall@200: 0.923
  - Validation nDCG@100: 0.221
  - Validation nDCG@200: 0.230

### Family 2: Sequential Neural Networks

Sequential models process chronological reader interaction sequences, combining multimodal content embeddings with attention mechanisms and publication freshness decay.

- [`2.1_lstm/`](2.1_lstm/): Target-Aware LSTM with recency-biased query-key attention and soft freshness decay (lambda = 2.0). Combines an LSTM recurrent sequence backbone with candidate-queried attention over 768-D Gemini text embeddings, 1024-D ResNet visual representations, and categorical embeddings. Selected as the primary backbone for the Stage 1 ensemble (75% fusion weight).
  - Validation Recall@100: 0.907
  - Validation Recall@200: 0.965
  - Validation nDCG@100: 0.251
  - Validation nDCG@200: 0.342
- [`2.2_huim/`](2.2_huim/): Deep Interest Network (DIN) with an order-invariant parallel event encoder, 10 logarithmic recency bins, and ReLU match projection sparsity.
  - Validation Recall@100: 0.914
  - Validation Recall@200: 0.965
  - Validation nDCG@100: 0.286
  - Validation nDCG@200: 0.343

### Family 3: Pretrained and Fine-Tuned Language Representations

Language representation experiments investigate semantic text retrieval across zero-shot embeddings, domain fine-tuning, and publication freshness adjustments.

- [`3.1_pretrained_similarity_gemini/`](3.1_pretrained_similarity_gemini/) and [`3.2_pretrained_similarity_bert/`](3.2_pretrained_similarity_bert/): Unsupervised semantic similarity baselines using frozen multilingual Gemini embeddings (`gemini-embedding-2`, 768-D) and Danish DistilBERT representations.
- [`3.3_pretrained_retrieval/`](3.3_pretrained_retrieval/) and [`3.4_finetuned_distilbert/`](3.4_finetuned_distilbert/): Contrastive retrieval workflows and domain-specific bi-encoders fine-tuned on Danish news articles.
- [`3.5_gemini_freshness/`](3.5_gemini_freshness/) and [`3.6_bert_intro_freshness/`](3.6_bert_intro_freshness/): Content similarity formulations combined with explicit publication recency adjustments to address static semantic drift.

---

## Standardized Export Contract

To ensure interoperability with downstream ensembling and Stage 2 diversity reranking, every model outputs a standardized ranking artifact:

- **Target Location:** `outputs/<run_id>/rankings/<split>/submission_top_200.parquet` where `split` is `train`, `validation`, or `test`.
- **File Schema:**
  - `behaviour_id` (`string` or `int64`): Unique impression identifier formatted as `user_id:impression_id` or original integer ID matching project split manifests.
  - `article_ids` (`list<int64>`): Exactly `min(200, eligible_catalogue_size)` native `int64` candidate article identifiers sorted in descending order of model score.
- **Strict Data Integrity:** No clicked articles or recorded-slate candidates are artificially injected into shortlists. All recommendations reflect point-in-time model scoring over eligible catalogue articles.
- **Diagnostic Exports:** Complete candidate scores for recorded in-view slates are stored under `outputs/<run_id>/diagnostics/recorded_slates/` for ROC-AUC validation and ranking fidelity diagnostics.

---

## Google Drive Storage Link

Trained model checkpoints, Optuna trial databases, precomputed embeddings, and full prediction Parquet files are stored on Google Drive in `BT4222 Shared Final Folder/Experiments/`:

```
BT4222 Shared Final Folder/Experiments/
├── 1.1_lightgbm_ranker/        # LightGBM booster bundles and full-split submission rankings
├── 1.2_logistic_regression/    # Linear baseline weights and prediction files
├── 1.3_xgboost/                # XGBoost models, top-200 rankings, and Tree SHAP artifacts
├── 2.1_lstm/                   # Target-Aware LSTM weights, Optuna trials, and ranking exports
├── 2.2_huim/                   # HUIM DIN model states, freshness configs, and prediction files
└── 3.x_pretrained_.../         # Gemini, DistilBERT, and freshness similarity score Parquets
```

Google Colab notebooks mount this directory using:
```python
PROJECT_ROOT = Path('/content/drive/MyDrive/Colab Notebooks/BT4222 Shared Final Folder')
```

---

## Contributing and Historical Development Standards

For historical development guidelines, data contracts, and Optuna hyperparameter tuning recipes, refer to [CONTRIBUTING.md](CONTRIBUTING.md). All models and pipelines reflect collaborative group work, with contributor attribution in manifests being optional.
