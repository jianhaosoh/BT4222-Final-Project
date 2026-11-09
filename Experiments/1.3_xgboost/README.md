# 1.3 XGBoost Ranker

## Overview
Point-in-time Gradient Boosted Decision Tree (GBDT) ranker selected for the final Stage 1 ensemble alongside the neural sequential model.

## Configuration
- Tree structure: Maximum 31 leaves
- Estimators: 25 trees
- Subsample ratio: 0.8
- Column sample by tree (`colsample_bytree`): 0.8

## Evaluation
Evaluated on the full 20,738 article catalogue under point-in-time publication eligibility:
- Validation Recall@100: 0.858
- Validation Recall@200: 0.923
- Validation nDCG@100: 0.221
- Validation nDCG@200: 0.230
- Holdout Test Recall@5: 0.138
- Holdout Test nDCG@5: 0.088
- Holdout Test Shared ILD@5: 0.626

## Notebooks
- [`experiment.ipynb`](experiment.ipynb): Model training, validation, full-split catalogue inference, and Tree SHAP feature attribution analysis.

## Key Observations
Outperforms other tabular baselines. Contributes vital tree-based category affinity and freshness interaction signals to the 75:25 LSTM+XGBoost Stage 1 ensemble.
