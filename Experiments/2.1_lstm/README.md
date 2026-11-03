# 2.1_lstm — LSTM with target-aware attention

This experiment scores every recorded candidate in each impression slate, using:

- **History:** the reader's chronological fixed pre-period clicks, encoded by an LSTM.
- **Attention:** each candidate is projected into an attention query over the LSTM states, giving a candidate-specific history context.
- **Head:** an MLP that predicts a click logit from that context, the candidate's own information, and their interactions.

The model adapts the course Week 8 recipe from rating regression to binary clicks, using BCE on logits.

| File | Purpose |
| --- | --- |
| `experiment.ipynb` | Colab notebook with the common nine-section structure: configuration, preflight checks, training or loading, diagnostics, exports, persistence and reload check. |
| `lstm_data.py` | Path resolution, candidate and history loading, vector decoding, train-fitted preprocessing, impression metrics, and schema-v2 exports. Has no PyTorch dependency. |
| `lstm_model.py` | Article encoder, LSTM, target-aware attention, head, batching, and training with early stopping and checkpoints. |
| `requirements.txt` | Dependencies. Each run also writes its exact versions. |

## Inputs (under the shared `DATA_ROOT`)

- `feature_engineering/{split}_features.parquet` provides only the candidate keys, labels, impression context, and shared `article_age_h`. The engineered history aggregates are not used.
- `train_val/{train,validation}/history.parquet` and `test/history.parquet` provide the ordered fixed history: article, time, read time and scroll %.
- `train_val/articles.parquet` and `test/articles.parquet` provide article metadata. Future article totals are never read.
- `{train_val,test}/article embeddings/{text,image}_embeddings.parquet` hold the frozen vectors. The canonical `embeddings/{train_val,test}/` locations are also accepted.
- `feature_engineering/feature_vocabulary.json` is the shared train vocabulary.

Paths are resolved explicitly, in this order:

1. `PATH_OVERRIDES`
2. `loader_paths.json`
3. the documented candidates

A missing required file stops the run.

The shared folder is owned by another account. Add a Drive shortcut so it appears under `MyDrive`.

## Running

1. In Colab, choose a GPU runtime and open `experiment.ipynb`.
2. Code version. The notebook pins `CODE_COMMIT` and the sha256 of both helper modules. It then clones, or fetches into an existing clean clone, and checks out that commit detached. It stops if the hashes or commit differ, or if an existing clone has local changes.
3. Dependencies are checked before the helpers are imported. DuckDB is installed only if it is missing.
4. The first execution uses the defaults:
   - `SMOKE_TEST = True` (5% of users, one epoch, run ID tagged `_smoke`)
   - `RUN_ABLATIONS = False`
   - `RUN_OPTUNA = False`
   - `RUN_RESUME_CHECK = True`
5. Full main model: set `SMOKE_TEST = False`.
6. Controls without retraining the main model:
   - set `TRAIN_MODEL = False`, `LOAD_RUN_ID = "<main run>"` and `RUN_ABLATIONS = True`;
   - the controls reuse the saved hyperparameters;
   - the run ID ends in `_load_controls`.
7. Resume after a disconnect: set `RESUME_RUN_ID = "<run id>"`. It resumes from the Drive recovery checkpoint and is not guaranteed to be bitwise exact.

## Vector files

Some pyarrow versions (18–20) cannot read nullable fixed-size-list vector columns. They fail with `Expected all lists to be of size=768 but index … had size=0`. For exactly that error, the loader falls back to DuckDB's Parquet reader, which keeps values, dimensions, IDs, row order and nulls. `run_config.json` records the reader used for each file.

As an alternative, set `WRITE_COMPATIBLE_VECTORS = True`. This writes verified copies with variable-length lists under `OUTPUT_ROOT/compatible_vectors/` and uses them in place of the originals. Each copy is compared exactly with its source before it is accepted, and the originals are never modified.

## Outputs

Local outputs go to `/content/bt4222_outputs/2.1_lstm/<run_id>/` in Colab, or `outputs/` locally. Both are ignored by Git.

Each run is copied to `OUTPUT_ROOT/Experiments/2.1_lstm/<run_id>/` and verified there with sha256 hashes. By default, `OUTPUT_ROOT` is `MyDrive/BT4222 Experiment Outputs`, a folder you own and separate from the shared inputs. Setting it to the shared folder also works if you have edit access. The notebook checks write access before it trains.

The run folder contains:

- `predictions/{split}_scores.csv`: `behaviour_id,article_id,score,click_probability`. The `score` is a logit.
- `rankings/{split}_ranked_articles.csv`: complete slates as a JSON array, ordered by descending score, with ties broken by ascending article ID.
- `model/`:
  - `model_state.pt` (state_dict)
  - `model_config.json`
  - `preprocessing.json`
  - `mappings.npz`
  - `code/` (bundled modules)
  - `recovery/last_checkpoint.pt`
- `metrics.json`, `run_config.json`, `training_history.{json,csv}`, `requirements.txt`, `reload_check.json`, `resume_check.json`, `manifest.json`
- `ablations/<variant>/summary.json`
- `logs/notebook_log.txt`
- `review_<run_id>.zip`

Optuna, when enabled, writes `OUTPUT_ROOT/Experiments/2.1_lstm/tuning/<study>/` containing a SQLite snapshot and `trials.csv`.

Validation metrics are **development** results, because validation is used for early stopping and selection. Test labels are unavailable, so no test metric is produced.
