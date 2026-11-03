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
2. The first code cell clones this repository if `lstm_*.py` are not next to the notebook. Set `REPO_BRANCH` to the branch that contains them.
3. Optional smoke run: set `SMOKE_TEST = True`. This uses 5% of users and one epoch, and the run ID is tagged `_smoke`.
4. Full run: keep the defaults and choose *Runtime → Run all*.
5. Load-only export: set `TRAIN_MODEL = False` and `LOAD_RUN_ID = "<run id>"`.
6. Resume after a disconnect: set `RESUME_RUN_ID = "<run id>"`. It resumes from the Drive recovery checkpoint and is not guaranteed to be bitwise exact.

## Outputs

Local outputs go to `/content/bt4222_outputs/2.1_lstm/<run_id>/` in Colab, or `outputs/` locally. Both are ignored by Git. Each run is copied to `<DATA_ROOT>/Experiments/2.1_lstm/<run_id>/` and verified there with sha256 hashes.

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
- `metrics.json`, `run_config.json`, `training_history.{json,csv}`, `requirements.txt`, `reload_check.json`, `manifest.json`
- `ablations/<variant>/summary.json`
- `logs/notebook_log.txt`
- `review_<run_id>.zip`

Optuna, when enabled, writes `<DATA_ROOT>/Experiments/2.1_lstm/tuning/<study>/` containing a SQLite snapshot and `trials.csv`.

Validation metrics are **development** results, because validation is used for early stopping and selection. Test labels are unavailable, so no test metric is produced.
