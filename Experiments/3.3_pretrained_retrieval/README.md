# Sampled training data for pretrained-embedding retrieval

This experiment prepares data for a later learned user/article retrieval layer.
It does not train weights, fine-tune a transformer or replace the existing
Gemini/BERT similarity baselines.

Open `prepare_training_examples.ipynb` in Colab. Upload BOTH `training_examples.py`
and `catalogue_pipeline.py` from this folder. Set your shared Drive root. Default:
50,000 sampled complete TRAIN requests, seed 42, latest 10 fixed past clicks,
up to four displayed-but-unclicked negatives per usable positive.

Existing shared Gemini vectors are the default. For BERT, override the two
embedding parquet paths with your existing generated cache paths and record the
encoder/revision/text variant. No embedding generation is needed. Preparation
uses CPU and disk; no GPU is needed until later model fitting.

## What is sampled

- Requests are selected using the existing deterministic hash rule. Their
  recorded slates remain complete before constructing examples.
- Each usable clicked article generates an example. ALL clicked articles in
  that same slate are excluded from its negative pool.
- Negatives come only from displayed-but-unclicked candidates in that request.
  Sampling is deterministic per seed/request/positive, without replacement.
- Keep fewer than four when fewer usable negatives exist. A later training loss
  must mask padding rather than treating padding as another negative.
- User profiles use a normalized mean of the latest 10 train-history events,
  selecting recent events before dropping unavailable vectors. The source
  history must strictly predate the full training split.
- Missing profiles, unavailable vectors and publication-ineligible training
  candidates are excluded from examples and counted. This does not change
  catalogue inference coverage or evaluation denominators.
- No validation/test behaviour table or labels are read. The declared ordered
  train_val/test ARTICLE lookup union is used, retaining first snapshots, just
  as in the existing frozen-embedding experiment. Snapshot provenance can postdate
  requests and must be reported alongside those existing baseline limitations.
- Displayed-but-unclicked articles are imperfect negative signals, while unshown
  catalogue outcomes remain unknown. No catalogue-wide negative expansion occurs.

## Outputs

| File | Contents |
| --- | --- |
| `training_examples.parquet` | Request/user IDs, one positive ID, list of up to four negative IDs, coverage |
| `user_profiles.parquet` | One normalized mean profile per user used by examples |
| `article_vectors.parquet` | One normalized vector per article used by examples |
| `training_config.json` | Data recipe, eligibility/filtering and label interpretation |
| `training_sources.json` | Config, source fingerprints and helper versions |
| `training_summary.json` | Request/example counts, exclusions, sampling IDs and preparation timing |

Join examples to profiles on `user_id` and article vectors on the positive and
negative article IDs. Vectors are not repeated in every example. The local
`outputs/<run_id>/_indexes/` directory contains generated SQLite indexes and is
omitted from the portable dataset. The notebook copies the six output artifacts
to `<DRIVE_ROOT>/Experiments/3.3_pretrained_retrieval/<run_id>/` without overwriting
existing destinations. Use fresh run IDs when changing input/settings.

The existing training feature file still needs one disk-indexing pass; this is
not a full-catalogue scoring job. The selected request count may exceed the count
that produces usable examples; a multi-click request may produce several examples.
There are no training losses or recommendation metrics until a model is fitted.

Next: train small user/article projections on these examples, select using the
same validation catalogue Recall@200, and keep test labels out of model selection.
The existing fixed-similarity notebooks remain separate reference experiments.

Synthetic checks: `python -m unittest discover -s Experiments/3.3_pretrained_retrieval/tests -p 'test_*.py'`
