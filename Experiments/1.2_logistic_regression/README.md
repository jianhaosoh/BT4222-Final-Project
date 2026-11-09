# 1.2 Logistic Regression Baseline

## Overview
L2-regularized linear baseline trained on engineered tabular features for candidate ranking.

## Configuration
- Regularization: L2 penalty with regularization strength C = 0.01
- Feature preprocessing: Standard scaled numeric features, one-hot encoded categorical variables

## Evaluation
Evaluated on the full 20,738 article catalogue under point-in-time publication eligibility:
- Validation Recall@100: 0.769
- Validation Recall@200: 0.915
- Validation nDCG@100: 0.175
- Validation nDCG@200: 0.196

## Notebooks
- [`experiment.ipynb`](experiment.ipynb): Model training, validation, and full-split catalogue inference.

## Key Observations
High broad retrieval recall at K=200, but limited ranking precision (nDCG) due to linear model capacity.
