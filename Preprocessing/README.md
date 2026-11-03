# Using the engineered feature files

Run [step3_feature_engineering.ipynb](step3_feature_engineering.ipynb) in Colab to build the candidate-level Parquet files. They are saved under:

```text
MyDrive/Colab Notebooks/BT4222 Shared Final Folder/feature_engineering/
```

Each row is one candidate article shown in an impression. Train and validation have `clicked` (1 for a clicked candidate, 0 otherwise). Test has a null `clicked` column. Every feature file has the same column names for its case. The history features are already joined into these files; `*_history_profiles.parquet` is for inspecting the intermediate per-user profiles.

## Four input cases

Choose **one case** for a model experiment. In the table, `{split}` is `train`, `validation`, or `test`.

| Case | File relative to `feature_engineering/` | Predictor choice |
| --- | --- | --- |
| `raw` | `{split}_features.parquet` | Original fields; no log variants or scaling. |
| `log` | `log_variants/{split}_features.parquet` | Use train-selected `_log1p` fields in place of their raw counterparts. Other fields remain raw. |
| `scaled` | `scaled_log_variants/{split}_features.parquet` | Use the scaled raw fields; leave out `_log1p` fields. |
| `log_scaled` | `scaled_log_variants/{split}_features.parquet` | Use scaled `_log1p` fields in place of their scaled raw counterparts. |

There is no separate scaled-only Parquet file: the scaled log-variant file contains **both** versions, so `scaled` and `log_scaled` differ in which columns you select. The log selection is recorded in `log_variants/log_variant_selection.json`. The fitted train imputation and scaling values are recorded in `scaled_log_variants/scaling_state.json`.

The `log` and `log_scaled` cases require Step 15 of the notebook to have selected and written log variants. The scaled cases require Step 16. If no field meets the log-selection rule, Step 15 does not write log-variant files.

## Columns in the generated datasets

The last executed Step 3 schema in `step3_feature_engineering.ipynb` contains **209 raw columns**: 43 fixed columns, 116 history-preference columns, and 50 one-hot columns. The train, validation, and test feature files use this schema; `clicked` is null in test. Of the 209 columns, 202 are predictors selected by `load_xy()` and seven are identifiers, time, split, or target fields. The vocabulary comes from train, so regenerate this list from Parquet metadata if you rebuild with different source data.

**Identifiers, time, and target (7):**

```text
split, user_id, impression_id, candidate_article_id
current_page_article_id, impression_time, clicked
```

**Behavior and article features (15):**

```text
is_sso_user, is_subscriber, is_front_page, hour
day_of_week, slate_size, is_weekend, premium
body_word_count, topic_count, has_topics, article_age_h
article_age_missing, is_24h_eligible, slate_relative_age_h
```

**History and candidate-history features (21):**

```text
history_size, median_history_body_word_count, median_history_read_time, mean_history_scroll_percentage
history_body_word_count_missing, history_read_time_missing, history_scroll_missing, history_clicks_7d
candidate_category_preference, candidate_category_preference_missing, candidate_article_type_preference, candidate_article_type_preference_missing
candidate_topic_preference, candidate_topic_preference_missing, category_affinity_72h, category_affinity_missing
last5_mean_topic_jaccard, topic_affinity_missing, has_any_topic_overlap, same_category_mean_read_time
same_category_read_depth_missing
```

**Historical category fractions (24):**

```text
hist_category_frac_000, hist_category_frac_001, hist_category_frac_002, hist_category_frac_003
hist_category_frac_004, hist_category_frac_005, hist_category_frac_006, hist_category_frac_007
hist_category_frac_008, hist_category_frac_009, hist_category_frac_010, hist_category_frac_011
hist_category_frac_012, hist_category_frac_013, hist_category_frac_014, hist_category_frac_015
hist_category_frac_016, hist_category_frac_017, hist_category_frac_018, hist_category_frac_019
hist_category_frac_020, hist_category_frac_021, hist_category_frac_022, hist_category_frac_other
```

**Historical topic fractions (79):**

```text
hist_topic_frac_000, hist_topic_frac_001, hist_topic_frac_002, hist_topic_frac_003
hist_topic_frac_004, hist_topic_frac_005, hist_topic_frac_006, hist_topic_frac_007
hist_topic_frac_008, hist_topic_frac_009, hist_topic_frac_010, hist_topic_frac_011
hist_topic_frac_012, hist_topic_frac_013, hist_topic_frac_014, hist_topic_frac_015
hist_topic_frac_016, hist_topic_frac_017, hist_topic_frac_018, hist_topic_frac_019
hist_topic_frac_020, hist_topic_frac_021, hist_topic_frac_022, hist_topic_frac_023
hist_topic_frac_024, hist_topic_frac_025, hist_topic_frac_026, hist_topic_frac_027
hist_topic_frac_028, hist_topic_frac_029, hist_topic_frac_030, hist_topic_frac_031
hist_topic_frac_032, hist_topic_frac_033, hist_topic_frac_034, hist_topic_frac_035
hist_topic_frac_036, hist_topic_frac_037, hist_topic_frac_038, hist_topic_frac_039
hist_topic_frac_040, hist_topic_frac_041, hist_topic_frac_042, hist_topic_frac_043
hist_topic_frac_044, hist_topic_frac_045, hist_topic_frac_046, hist_topic_frac_047
hist_topic_frac_048, hist_topic_frac_049, hist_topic_frac_050, hist_topic_frac_051
hist_topic_frac_052, hist_topic_frac_053, hist_topic_frac_054, hist_topic_frac_055
hist_topic_frac_056, hist_topic_frac_057, hist_topic_frac_058, hist_topic_frac_059
hist_topic_frac_060, hist_topic_frac_061, hist_topic_frac_062, hist_topic_frac_063
hist_topic_frac_064, hist_topic_frac_065, hist_topic_frac_066, hist_topic_frac_067
hist_topic_frac_068, hist_topic_frac_069, hist_topic_frac_070, hist_topic_frac_071
hist_topic_frac_072, hist_topic_frac_073, hist_topic_frac_074, hist_topic_frac_075
hist_topic_frac_076, hist_topic_frac_077, hist_topic_frac_other
```

**Historical article-type fractions (13):**

```text
hist_article_type_frac_000, hist_article_type_frac_001, hist_article_type_frac_002, hist_article_type_frac_003
hist_article_type_frac_004, hist_article_type_frac_005, hist_article_type_frac_006, hist_article_type_frac_007
hist_article_type_frac_008, hist_article_type_frac_009, hist_article_type_frac_010, hist_article_type_frac_011
hist_article_type_frac_other
```

**One-hot encoded device (5):**

```text
ohe_device_type_000, ohe_device_type_001, ohe_device_type_002, ohe_device_type_missing
ohe_device_type_unknown
```

**One-hot encoded article category (25):**

```text
ohe_category_000, ohe_category_001, ohe_category_002, ohe_category_003
ohe_category_004, ohe_category_005, ohe_category_006, ohe_category_007
ohe_category_008, ohe_category_009, ohe_category_010, ohe_category_011
ohe_category_012, ohe_category_013, ohe_category_014, ohe_category_015
ohe_category_016, ohe_category_017, ohe_category_018, ohe_category_019
ohe_category_020, ohe_category_021, ohe_category_022, ohe_category_missing
ohe_category_unknown
```

**One-hot encoded article type (14):**

```text
ohe_article_type_000, ohe_article_type_001, ohe_article_type_002, ohe_article_type_003
ohe_article_type_004, ohe_article_type_005, ohe_article_type_006, ohe_article_type_007
ohe_article_type_008, ohe_article_type_009, ohe_article_type_010, ohe_article_type_011
ohe_article_type_missing, ohe_article_type_unknown
```

**One-hot encoded premium/subscriber combination (6):**

```text
ohe_premium_subscriber_000, ohe_premium_subscriber_001, ohe_premium_subscriber_002, ohe_premium_subscriber_003
ohe_premium_subscriber_missing, ohe_premium_subscriber_unknown
```

Numbered history and one-hot columns are mapped to their category, topic, article type, or device values in `feature_vocabulary.json`. The `_other`, `_missing`, and `_unknown` columns are included above.

The log-variant and scaled log-variant Parquet files retain the 209 columns above and add these **seven** columns from the last train run (216 physical columns total):

```text
median_history_body_word_count_log1p, article_age_h_log1p,
same_category_mean_read_time_log1p, median_history_read_time_log1p,
body_word_count_log1p, history_clicks_7d_log1p, slate_size_log1p
```

`load_xy()` selects either each original column or its `_log1p` replacement, so every input case still has 202 predictor columns for this run. To list the **actual** column names and types in any saved file without loading its rows, run this in Colab:

```python
from pathlib import Path
import pyarrow.parquet as pq
from google.colab import drive

drive.mount("/content/drive")
root = Path(
    "/content/drive/MyDrive/Colab Notebooks/"
    "BT4222 Shared Final Folder/feature_engineering"
)
case_dir = {"raw": root, "log": root / "log_variants",
            "scaled": root / "scaled_log_variants",
            "log_scaled": root / "scaled_log_variants"}
CASE = "raw"       # raw, log, scaled, or log_scaled
SPLIT = "train"    # train, validation, or test
path = case_dir[CASE] / f"{SPLIT}_features.parquet"
parquet = pq.ParquetFile(path)
print(f"{parquet.metadata.num_rows:,} rows; {len(parquet.schema_arrow)} columns")
for number, field in enumerate(parquet.schema_arrow, start=1):
    print(f"{number:3d}. {field.name}: {field.type}")
```

## Preview before loading

Run [step3b_demo_call_engineered_features.ipynb](step3b_demo_call_engineered_features.ipynb) in Colab to see the row count, predictor count, and first five rows of each split. It reads Parquet metadata and only a small batch of selected columns. It does **not** load all candidates into pandas or compute a full-dataset label distribution. Change `CASE` in that notebook to preview `raw`, `log`, `scaled`, or `log_scaled`.

## Load `X`, `y`, and impression keys

The code below defines a loader in Colab; defining it does not read the feature rows. Change `FEATURE_ROOT` if you downloaded the files elsewhere. Load only the splits you need at a time, since a full train table can use substantial RAM.

```python
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from google.colab import drive

drive.mount("/content/drive")
FEATURE_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/"
    "BT4222 Shared Final Folder/feature_engineering"
)

CASE_DIRS = {
    "raw": FEATURE_ROOT,
    "log": FEATURE_ROOT / "log_variants",
    "scaled": FEATURE_ROOT / "scaled_log_variants",
    "log_scaled": FEATURE_ROOT / "scaled_log_variants",
}
NON_PREDICTORS = {
    "split", "user_id", "impression_id", "candidate_article_id",
    "current_page_article_id", "impression_time", "clicked",
}

def load_xy(split, case):
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if case not in CASE_DIRS:
        raise ValueError(f"Unknown case: {case}")

    path = CASE_DIRS[case] / f"{split}_features.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    available = pq.ParquetFile(path).schema_arrow.names

    if case in {"log", "log_scaled"}:
        selection_path = FEATURE_ROOT / "log_variants/log_variant_selection.json"
        source_fields = json.loads(selection_path.read_text())["source_fields"]
        log_fields = {f"{field}_log1p" for field in source_fields}
        missing = log_fields - set(available)
        if missing:
            raise ValueError(f"Log fields missing from {path}: {sorted(missing)}")
        excluded = set(source_fields)  # choose the log version of each selected field
    else:
        excluded = {field for field in available if field.endswith("_log1p")}

    feature_columns = [
        field for field in available
        if field not in NON_PREDICTORS and field not in excluded
    ]
    key_columns = ["user_id", "impression_id", "candidate_article_id"]
    columns_to_read = key_columns + feature_columns
    if split != "test":
        columns_to_read.append("clicked")
    data = pd.read_parquet(path, columns=columns_to_read)

    X = data[feature_columns].copy()
    keys = data[key_columns].copy()
    if split == "test":
        y = None
    else:
        if data["clicked"].isna().any():
            raise ValueError(f"{split} has missing click labels")
        y = data["clicked"].astype("int8")
    return X, y, keys

```

When you are ready to train and have enough RAM, run this separately. You can defer loading test until prediction time:

```python
CASE = "raw"  # change to "log", "scaled", or "log_scaled"
X_train, y_train, train_keys = load_xy("train", CASE)
X_validation, y_validation, validation_keys = load_xy("validation", CASE)
assert X_train.columns.equals(X_validation.columns)
print(CASE, X_train.shape, X_validation.shape)
print("Train target:", y_train.value_counts().to_dict())

# Later, for predictions:
# X_test, y_test, test_keys = load_xy("test", CASE)
# assert X_train.columns.equals(X_test.columns) and y_test is None
```

`X_train`, `X_validation`, and `X_test` are the model input tables. `y_train` and `y_validation` are the click targets. Keep `train_keys`, `validation_keys`, and `test_keys` beside the predictions so you can group candidates by **`(user_id, impression_id)`** and identify each article. In the test data, `impression_id` alone is not unique across users.

Fit the model with `X_train, y_train`, evaluate it with `X_validation, y_validation`, and predict on `X_test`. The numbered one-hot and history-preference columns are mapped to their original values in `feature_vocabulary.json`.

The saved files retain both raw and log alternatives for traceability; the loader selects one version per transformed field. The raw and log cases can still contain null predictors. The scaled files impute their selected continuous columns using train values, but other nullable fields may remain, so use model-appropriate missing-value handling. Fit any additional imputer or scaler on **train only**, then apply it to validation and test.

## Article text and image embeddings

[`embedding_pipeline.py`](embedding_pipeline.py) is a standalone regular-API example.
It combines each article's Danish title, subtitle, and body into one text document,
requests one Gemini Embedding 2 vector, and joins the supplied image vectors by
`article_id`. Text and image vectors are saved separately.

### Setup and run

Use Python 3.11 or later. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r "Preprocessing/requirements-embeddings.txt"
cp .env.example .env
```

Add your `GEMINI_API_KEY` to `.env`. This file is ignored by Git. The script reads
it from the repository root; existing environment variables take precedence.
The defaults are `gemini-embedding-2` and 768 text dimensions.

The current data remains in the sibling `Project` directory. Adjust these paths
if your files are elsewhere:

```sh
EMBEDDING_DATA=../Project/data/ebnerd_small
EMBEDDING_IMAGES=../Project/data/ebnerd_large/Ekstra_Bladet_image_embeddings/image_embeddings.parquet

# Local input check: no API calls and no output writes.
python "Preprocessing/embedding_pipeline.py" \
  --articles "$EMBEDDING_DATA/articles.parquet" \
  --image-embeddings "$EMBEDDING_IMAGES" --dry-run

# Generate a small sample in its own directory.
python "Preprocessing/embedding_pipeline.py" \
  --articles "$EMBEDDING_DATA/articles.parquet" \
  --image-embeddings "$EMBEDDING_IMAGES" --limit 3 \
  --output-dir data/embeddings/ebnerd_small-smoke
```

For the full catalogue, omit `--limit 3` and the smoke-test `--output-dir`.

The feature-engineering notebook combines the train/validation and official test
article catalogues. To prepare vectors for both, pass them in the same order:

```sh
python "Preprocessing/embedding_pipeline.py" \
  --articles /path/to/train_val/articles.parquet /path/to/ebnerd_testset/articles.parquet \
  --image-embeddings "$EMBEDDING_IMAGES" --output-dir data/embeddings/combined
```

Overlapping article IDs keep the first catalogue's snapshot, matching Step 3.
The current local embedding files cover the 20,738-article SMALL catalogue; extra
articles from the official test catalogue need vectors generated from that file.
Use a left join when adding vectors to candidates so missing vectors never remove
candidate rows.

The script uses the regular API, one document per request. Successful text vectors
are checkpointed every 100 rows and on interruption. A separate local
`text_cache.parquet` keeps all successful vectors even after a limited or interrupted
rerun. Repeating the command reuses
vectors with the same prepared text, model and dimension, including identical text
on different article IDs. Documents longer than 8,192 tokens receive an explicit
`input_too_long` status. Temporary rate-limit/server errors are retried.

### Files and model inputs

Default outputs are in `data/embeddings/ebnerd_small/`:

| File | Main fields |
| --- | --- |
| `text_embeddings.parquet` | `article_id`, `text_embedding` (768 float32 values), `has_text_embedding`, `status`, `error_code`, `input_hash` |
| `image_embeddings.parquet` | `article_id`, `image_embedding` (1,024 float32 values), `has_image_embedding`, `status`, `error_code` |

Missing inputs remain null; no article is dropped and no zero vector is invented.
The inspected SMALL catalogue has 20,738 articles, 18 without usable text. The
shared image artifact covers 16,995 articles (81.95%); 3,743 have no image vector.
Extra IDs in the shared image artifact are excluded by the article-ID join.

Use `candidate_article_id = article_id` for candidate feature lookups and the same
article-ID mapping for historical clicks. Keep dense vectors in lookup tables
until model batching rather than duplicating them across the full impression
file. Text and image vectors come from different model spaces; do not compute
text-to-image cosine similarity between them. Fit downstream scaling, PCA and
missing-value handling on training data only. The text is a supplied catalogue
snapshot, not a reconstruction of article text at each impression time.
