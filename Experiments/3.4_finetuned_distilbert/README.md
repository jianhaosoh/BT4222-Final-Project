# Fine-tuned pretrained DistilBERT retrieval

`finetune_distilbert.ipynb` updates the actual pretrained sentence encoder, rather
than training the projection-only model proposed in experiment 3.3. The existing
frozen Gemini/BERT experiments and caches are preserved.

Open in Colab with a GPU. Upload BOTH `finetune_encoder.py` and
`catalogue_pipeline.py`. Set the shared Drive root and check the history-text
coverage audit. Pilot defaults: 10,000 training requests, 1,000 fixed validation
requests, title+subtitle text, latest 10 fixed past clicks, one epoch, learning
rate 1e-5, final two DistilBERT blocks plus the encoder's existing dense module.
Set trainable_transformer_layers=None to update all pretrained encoder weights
in a separately declared experiment. No additional recommendation layers exist.

## Training objective

One example contains a user's preceding history, a clicked article, and up to
four displayed-but-unclicked alternatives from the same training request. Every
clicked article in that request is excluded from negative sampling. Examples
without usable history or alternatives, and candidates with missing text or
publication-ineligible metadata, are excluded and counted. Unshown articles are
not labeled as non-clicks. Displayed-but-unclicked labels remain imperfect signals.

The shared encoder produces normalized article vectors. The user vector is the
normalized mean of selected history vectors; repeated clicks retain weight.
A temperature-scaled softmax loss increases the clicked article's score relative
to the sampled alternatives. Gradients flow through BOTH history and candidate
encoding. Training uses forward(), not the no-gradient encode() convenience API.
Only training behaviour supplies optimizer examples. Validation selects checkpoints;
no test behaviour table or labels are read.

## Matched evaluation

The exact pretrained revision, article sources, encoding policy, request IDs and
history cutoffs are recorded. A frozen checkpoint is evaluated first. Each epoch
then regenerates vectors for the SAME eligible catalogue and all available history
texts. Known-click Recall@200 and observed-click nDCG@200 use full eligible pools,
not the sampled alternatives from training. Mean-profile scoring is used on both
sides of the comparison. The frozen checkpoint remains selected if better.

Titles/subtitles are joined with real newline separators. Token-overflow chunks
cover the complete selected text, using normalized chunk vectors, content-token
weighting and final normalization. A newly generated matched frozen baseline is
the reference: old text serialization or pooling differences can make older
saved metrics unsuitable for isolating fine-tuning effects.

The source train_val/test article union defines the candidate catalogue. Optional
additional_text_catalogues fill raw text for missing historical IDs only; they do
not expand candidate eligibility. Histories must predate the entire behaviour
split, not merely the sampled request dates. The inherited catalogue snapshot
provenance limitation remains; this is not proof of historical production supply.

## Outputs and later use

Each checkpoint includes an `encoder/` folder, newly generated
`text_embeddings.parquet`, and `validation_metrics.json`. Completed epochs also
save optimizer state. `finetuning_summary.json` records the matched comparison
and selected checkpoint. Copy the complete run to Drive, excluding SQLite indexes,
and retain source manifests. No automatic mid-epoch recovery is implemented.

Use the selected vector file as both embeddings_train_val and embeddings_test in
the existing similarity/export notebook. It is one union lookup; the actual
candidate catalogue still comes from the shared article sources. Keep latest-10
mean-profile scoring and record the fine-tuning checkpoint in embedding_source.
Use a new run ID and the group-agreed validation/export scope. Evaluate held-out
test outcomes only after choosing configuration and checkpoint.

The 1e-5/one-epoch/last-two-block settings are pilot choices, not guaranteed optimal
hyperparameters or a promise to beat the LSTM. Benchmark training/evaluation on a
small request sample before increasing the budget. Local tests use synthetic data
and randomly initialized tiny transformer weights; actual fine-tuning and retrieval
quality must be measured on Drive data in Colab.

References:
- https://www.sbert.net/docs/sentence_transformer/training_overview.html
- https://www.sbert.net/docs/package_reference/sentence_transformer/model.html
