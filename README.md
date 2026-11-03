# BT4222 Final Project: News Recommendation

## Machine learning problem

We aim to recommend relevant Ekstra Bladet news while supporting continued
reading, freshness and topic diversity. Stage 1 learns personal relevance from
recorded shown/clicked article pairs. For recommendations, it scores the article
catalogue eligible at the request timestamp and retains up to 200 articles.
Stage 2 applies business priorities within that shortlist to produce the final
top 5 recommendations.

Select Stage 1 models using catalogue validation known-click Recall@100, with
observed-click nDCG@100 as secondary. Report Recall@200 and observed-click
nDCG@200 on the retained pool too. Recorded-slate AUC is diagnostic; nDCG@5
evaluates the final Stage 2 list. Unshown article outcomes are unknown, so these
offline results do not establish catalogue-wide CTR, retention or profit. Never
tune model or reranker choices using test-set evaluation. See the experiment
guide for the separate evaluation pools and export contracts. Existing slate-only
experiments still need a catalogue inference/validation path before they
implement this intended pipeline.

## Dataset

This project uses the **EB-NeRD SMALL** dataset from Ekstra Bladet, a Danish news publisher. We retain its training split and sample disjoint validation and labelled test users from its original validation period, giving an approximate 80–10–10 split by behaviour rows. The dataset contains anonymized user interactions and Danish news articles; the training and validation behavior logs cover consecutive seven-day periods in May and June 2023.

- `behaviors.parquet`: Impression logs with the articles shown (`article_ids_inview`), articles clicked (`article_ids_clicked`), timestamps, and context such as device type.
- `history.parquet`: Each user's prior article clicks, with timestamps and reading interactions.
- `articles.parquet`: Article information such as title, category, topics, publication time, and premium status.

For training, each shown article is a candidate: clicked articles receive a
positive label, while shown but unclicked articles receive a negative label.
Do not expand training to all catalogue articles and label unshown articles zero.

Dataset source: [EB-NeRD dataset and documentation](https://recsys.eb.dk/dataset/).

## Preprocessing and embeddings

See [Preprocessing/README.md](Preprocessing/README.md) for the current engineered
feature files and the regular-API article embedding script. The script saves
Gemini text vectors and supplied EB-NeRD image vectors separately, keyed by
`article_id`, for lookup using `candidate_article_id`.

## Experiments

Use [Experiments/README.md](Experiments/README.md) for track numbering, the common
notebook structure, business-linked explainability, Parquet export formats and
saved-model handoff requirements. Keep preprocessing shared, model training and
validation development in the experiment notebooks, and cross-model comparison,
reranking/ensembling and final evaluation together in one shared stage. Each model
scores the full eligible catalogue but saves only the genuine top-200 articles
and their scores for every train, validation and test request, alongside scores
for every original recorded-slate candidate in separate diagnostic files.
Recorded slates and known clicks are never inserted into the genuine shortlist.
Exhaustive catalogue-score exports are not required.
