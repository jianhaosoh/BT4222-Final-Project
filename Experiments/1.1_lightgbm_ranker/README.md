# 1.1 LightGBM Ranker

## Overview
Point-in-time Gradient Boosted Decision Tree (GBDT) ranker trained on 202 engineered tabular features covering user engagement history, category click-through rates, request context, and article recency.

## Configuration
- Tree structure: 31 leaves
- Estimators: 75 trees
- Learning rate: 0.05

## Evaluation
Evaluated on the full 20,738 article catalogue under point-in-time publication eligibility:
- Validation Recall@100: 0.828
- Validation Recall@200: 0.907
- Validation nDCG@100: 0.217
- Validation nDCG@200: 0.228

## Notebooks
- [`experiment.ipynb`](experiment.ipynb): Model training, validation, and full-split catalogue inference.

## Key Observations
Rapid inference throughput and strong baseline ranking driven primarily by category click-through rates and publication recency.
