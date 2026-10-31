# Feature pipeline

The pipeline expands each impression into **one row per shown candidate article**. It takes that split's behaviors and fixed history plus the shared article catalogue, and writes one encoded Parquet file per split. Fit the one-hot mapping on train and reuse `onehot_state.json` for validation and test. See [pipeline_model_smoke_test.ipynb](pipeline_model_smoke_test.ipynb) for a runnable Python example.

## Feature fields

Types are Parquet types; some fields may be null. The fixed history is the snapshot supplied with the corresponding split. The four nominal fields marked below are present only in the temporary raw table; the final Parquet replaces them with one-hot columns.

| Field | Type | Role | Processing |
| --- | --- | --- | --- |
| split | string | Metadata | Set from the `split` argument; identifies the source period. |
| impression_id | uint32 | Group key | Groups candidates from the same impression. |
| user_id | uint32 | Join key | Joins the user’s fixed history. |
| candidate_article_id | int32 | Candidate key | Exploded from article_ids_inview; one row per candidate. |
| current_page_article_id | int32 | Context key | Source behavior article_id; used to derive is_front_page. |
| impression_time | timestamp[us] | Source time | Source timestamp for time features and history cutoffs. |
| clicked | int8 | Target | 1 if candidate is in article_ids_clicked, else 0; null for unlabeled test. |
| device_type | string | Predictor; encoded in final file | Device code converted to a nominal category. |
| is_sso_user | bool | Predictor | Source login-status flag. |
| is_subscriber | bool | Predictor | Source subscription snapshot. |
| is_front_page | bool | Predictor | True when current_page_article_id is null. |
| hour | int8 | Predictor | Hour extracted from impression_time, 0–23. |
| day_of_week | int8 | Predictor | Weekday from impression_time, Monday 0 through Sunday 6. |
| slate_size | int16 | Predictor | Number of in-view candidates in the impression. |
| has_history | bool | Predictor | Whether the user has a history row. |
| history_length | int32 | Predictor | Number of fixed-history clicks with valid timestamps. |
| history_length_log1p | float32 | Predictor | log(1 + history_length). |
| median_read_time | float32 | Predictor | Median finite, nonnegative historical read time in seconds. |
| median_read_time_log1p | float32 | Predictor | log(1 + median_read_time). |
| mean_scroll_percentage | float32 | Predictor | Mean valid historical scroll percentage, 0–100. |
| history_clicks_7d | int32 | Predictor | Fixed-history clicks in the seven days before this impression. |
| history_clicks_7d_log1p | float32 | Predictor | log(1 + history_clicks_7d). |
| category | string | Predictor; encoded in final file | Candidate article's category code as a nominal string. |
| article_type | string | Predictor; encoded in final file | Candidate article's type label. |
| premium | bool | Predictor | Candidate article's paywall flag. |
| article_age_h | float32 | Predictor | Hours from publication to impression; null if unknown or published later. |
| article_age_log1p | float32 | Predictor | log(1 + article_age_h) when age is valid. |
| article_age_missing | bool | Predictor | True when valid article age is unavailable. |
| is_24h_eligible | bool | Predictor | True when candidate was published 0–24 hours before its impression. |
| slate_relative_age_h | float32 | Predictor | Candidate age minus median age of 24-hour-eligible candidates in the same slate; null outside that group. |
| category_affinity_72h | float32 | Predictor | Candidate-category share of fixed-history clicks, exponentially weighted on a 72-hour scale. |
| category_affinity_missing | bool | Predictor | True when category affinity cannot be calculated. |
| last5_mean_topic_jaccard | float32 | Predictor | Mean topic overlap with the user's last five fixed-history clicks, skipping unusable comparisons. |
| topic_affinity_missing | bool | Predictor | True when topic affinity cannot be calculated. |
| has_any_topic_overlap | bool | Predictor | True if topic affinity is above zero; null if affinity is missing. |
| same_category_mean_log1p_read_time | float32 | Predictor | Mean log(1 + read seconds) for valid historical clicks in the candidate's category. |
| same_category_read_depth_missing | bool | Predictor | True when no usable same-category reading time exists. |
| premium_subscriber | string | Predictor; encoded in final file | Joint premium/subscriber group; missing group if either status is unavailable. |

All shown candidates remain in the file. Predictor marks a feature available to a model; the team can select among raw and log alternatives. Null values stay distinct from observed zeros.

## Final encoded file (*_encoded.parquet)

The final file contains every field above except `article_type`, `device_type`, `category`, and `premium_subscriber`. Those four are replaced with train-fitted 0/1 columns:

| Field pattern | Type | Role | Processing |
| --- | --- | --- | --- |
| ohe_FIELD_NNN | int8 | Predictor | 1 for the training category at index NNN in onehot_state.json. |
| ohe_FIELD_missing | int8 | Predictor | 1 when the raw value is null. |
| ohe_FIELD_unknown | int8 | Predictor | 1 for a category absent from the fitted training vocabulary. |

FIELD stands for device_type, category, article_type, or premium_subscriber. **NNN is a zero-based, zero-padded category index**: for example, ohe_category_000 refers to the first category listed under category in onehot_state.json. Validation and test use that same mapping. Numeric imputation and standardisation are not applied.

## Choosing model inputs

The encoded Parquet keeps all generated features. Select a model's columns after loading it:

~~~python
from feature_pipeline import encoded_baseline_columns, load_onehot_state

state = load_onehot_state("data/features/onehot_state.json")
columns = list(encoded_baseline_columns(
    state, include_engineered=False, log_transform=False
))
X_train = train[columns]
~~~

The default selects all EDA-retained original predictors, with raw numeric values. `include_engineered=True` adds the engineered predictors, including the premium/subscriber interaction. `log_transform=True` substitutes available `log1p` variants for history length, median read time, article age, and (when engineered features are included) seven-day click count. Each raw/log pair contributes one column. The same-category reading-depth feature is defined using log read times and does not change with this switch. Age, gender, and postcode remain excluded because the EDA found over 95% missingness. These switches choose model inputs; they do not change the generated Parquet files. The smoke-test notebook fits imputation and scaling on train after selecting columns.

