# BT4222 Final Project: News Recommendation

## Machine learning problem

We aim to recommend Ekstra Bladet news articles that readers are likely to engage with. For each user impression, a model predicts the probability of a click for every article shown and ranks the candidates accordingly. A later re-ranking step considers continued reading, article freshness, and topic diversity.

## Dataset

This project uses the **EB-NeRD SMALL** dataset from Ekstra Bladet, a Danish news publisher. We use its labelled training and validation splits. The dataset contains anonymized user interactions and Danish news articles; the training and validation behavior logs cover consecutive seven-day periods in May and June 2023.

- `behaviors.parquet`: Impression logs with the articles shown (`article_ids_inview`), articles clicked (`article_ids_clicked`), timestamps, and context such as device type.
- `history.parquet`: Each user's prior article clicks, with timestamps and reading interactions.
- `articles.parquet`: Article information such as title, category, topics, publication time, and premium status.

Each shown article is a candidate: clicked articles receive a positive label, while shown but unclicked articles receive a negative label. 

Dataset source: [EB-NeRD dataset and documentation](https://recsys.eb.dk/dataset/).

## Preprocessing and embeddings

See [Preprocessing/README.md](Preprocessing/README.md) for the current engineered
feature files and the regular-API article embedding script. The script saves
Gemini text vectors and supplied EB-NeRD image vectors separately, keyed by
`article_id`, for lookup using `candidate_article_id`.
