"""Memory-bounded, chronological EB-NeRD click-ranking baseline.

Run from the project root: python "Baseline Testing/train_baseline.py" --root PATH_TO_DATA
"""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import resource
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BEHAVIOR_COLUMNS = [
    'impression_id', 'user_id', 'impression_time', 'device_type',
    'article_ids_inview', 'article_ids_clicked',
]
NUMERIC = [
    'log_age_hours', 'age_missing', 'future_publication', 'premium',
    'article_missing', 'log_history_length', 'category_affinity',
    'log_category_clicks', 'previously_clicked', 'hour_sin', 'hour_cos',
]
CATEGORICAL = ['category', 'article_type', 'device_type', 'weekday', 'age_bucket']


def sample_impressions(path, count, seed, batch_size=32768):
    """Uniform sample without replacement, retaining complete impression slates.

    Independent random priorities avoid assuming Parquet rows are chronological.
    count=0 reads all impressions and should only be used with sufficient RAM.
    """
    rng = np.random.default_rng(seed)
    selected, priorities = None, np.empty(0)
    total = 0
    for batch in pq.ParquetFile(path).iter_batches(
        columns=BEHAVIOR_COLUMNS, batch_size=batch_size
    ):
        total += len(batch)
        values = rng.random(len(batch))
        indices = np.arange(len(batch))
        if count and len(indices) > count:
            indices = np.argpartition(values, count - 1)[:count]
        table = pa.Table.from_batches([batch]).take(pa.array(indices))
        selected = table if selected is None else pa.concat_tables([selected, table])
        priorities = np.concatenate([priorities, values[indices]])
        if count and len(priorities) > count:
            keep = np.argpartition(priorities, count - 1)[:count]
            selected = selected.take(pa.array(keep))
            priorities = priorities[keep]
    if selected is None or not len(selected):
        raise ValueError(f'No impressions in {path}')
    result = selected.to_pandas().sort_values(
        ['impression_time', 'impression_id'], kind='stable'
    ).reset_index(drop=True)
    if result.impression_id.duplicated().any():
        raise ValueError('Duplicate impression IDs in sample')
    if result.impression_time.isna().any():
        raise ValueError('Missing impression timestamps')
    print(f'{path}: sampled {len(result):,} of {total:,} impressions', flush=True)
    return result, total


def load_profiles(path, users, cutoff, article_categories):
    """Fixed history snapshot strictly before the first evaluated impression.

    Read only selected users, and exclude any history at or after cutoff.
    No behavior labels, current outcomes, or catalog popularity totals enter here.
    """
    profiles, excluded = {}, 0
    user_values = pa.array(sorted(map(int, users)), type=pa.uint32())
    for batch in pq.ParquetFile(path).iter_batches(
        columns=['user_id', 'article_id_fixed', 'impression_time_fixed'],
        batch_size=4096,
    ):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.is_in(table['user_id'], value_set=user_values))
        if not len(table):
            continue
        ids = table['article_id_fixed'].combine_chunks()
        timestamps = table['impression_time_fixed'].combine_chunks()
        lengths = pc.list_value_length(ids)
        if (ids.null_count or timestamps.null_count or
                not pc.all(pc.equal(lengths, pc.list_value_length(timestamps))).as_py()):
            raise ValueError('History ID and timestamp lists must align')
        flat_ids = pc.list_flatten(ids)
        flat_times = pc.list_flatten(timestamps)
        valid = pc.fill_null(pc.less(flat_times, pa.scalar(cutoff, type=flat_times.type)), False)
        parents = pc.filter(pc.list_parent_indices(ids), valid).to_numpy()
        past = pc.filter(flat_ids, valid).to_numpy()
        excluded += len(flat_ids) - len(past)
        counts = np.bincount(parents, minlength=len(table))
        offsets = np.r_[0, np.cumsum(counts)]
        for index, user in enumerate(table['user_id'].to_numpy()):
            user_past = past[offsets[index]:offsets[index + 1]]
            categories = Counter(article_categories.get(int(a), 'unknown') for a in user_past)
            profiles[int(user)] = (len(user_past), categories, np.unique(user_past))
    return profiles, excluded


def build_features(behaviors, articles, profiles):
    rows, labels, group_sizes = [], [], []
    for row in behaviors.itertuples(index=False):
        candidates = list(row.article_ids_inview)
        clicked = set(row.article_ids_clicked)
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError(f'Empty or duplicate candidates: {row.impression_id}')
        if not clicked.issubset(set(candidates)):
            raise ValueError(f'Clicked article outside slate: {row.impression_id}')
        group_sizes.append(len(candidates))
        n, category_counts, past_ids = profiles.get(int(row.user_id), (0, {}, np.empty(0, dtype=np.int32)))
        seen = set(np.intersect1d(candidates, past_ids).tolist())
        for candidate in candidates:
            category = articles['category'].get(candidate, 'unknown')
            category_count = category_counts.get(category, 0)
            rows.append((
                int(row.impression_id), int(candidate), row.impression_time,
                str(row.device_type), category,
                articles['article_type'].get(candidate, 'unknown'),
                articles['published_time'].get(candidate, pd.NaT),
                articles['premium'].get(candidate, False),
                candidate not in articles['category'],
                np.log1p(n), category_count / max(n, 1),
                np.log1p(category_count), candidate in seen,
            ))
            labels.append(int(candidate in clicked))
    columns = [
        'impression_id', 'candidate_article_id', 'impression_time', 'device_type',
        'category', 'article_type', 'published_time', 'premium', 'article_missing',
        'log_history_length', 'category_affinity', 'log_category_clicks',
        'previously_clicked',
    ]
    dtypes = ['int64', 'int64', 'datetime64[us]', 'object', 'object',
              'object', 'datetime64[us]', 'bool', 'bool',
              'float32', 'float32', 'float32', 'bool']
    frame = pd.DataFrame({name: pd.Series(values, dtype=dtype)
                          for name, values, dtype in zip(columns, zip(*rows), dtypes)})
    del rows
    age = (frame.impression_time - pd.to_datetime(frame.published_time)).dt.total_seconds() / 3600
    frame['article_age_hours'] = age
    frame['age_missing'] = age.isna()
    frame['future_publication'] = age.lt(0)
    # Keep unusual and missing metadata candidates; expose flags instead of dropping rows.
    frame['log_age_hours'] = np.log1p(age.clip(lower=0).fillna(0))
    frame['age_bucket'] = pd.cut(
        age, [-np.inf, 0, 6, 24, 72, 168, np.inf],
        labels=['nonpositive', '0-6h', '6-24h', '1-3d', '3-7d', '7d+'],
    ).astype('string').fillna('unknown')
    hour = frame.impression_time.dt.hour + frame.impression_time.dt.minute / 60
    frame['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    frame['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    frame['weekday'] = frame.impression_time.dt.weekday.astype(str)
    for col in NUMERIC:
        frame[col] = frame[col].astype('float32')
    for col in CATEGORICAL:
        frame[col] = frame[col].astype(str)
    y = np.asarray(labels, dtype=np.int8)
    sizes = np.asarray(group_sizes, dtype=np.int32)
    assert sizes.sum() == len(frame) == len(y)
    assert np.isfinite(frame[NUMERIC].to_numpy()).all()
    return frame, y, sizes


def evaluate(frame, y, sizes, scores):
    """Macro impression AUC and nDCG, with expected credit for tied scores.

    AUC gives 0.5 credit for ties. nDCG uses mean relevance within each score
    tie, so a category/freshness tie cannot benefit from the stored list order.
    """
    group = np.repeat(np.arange(len(sizes)), sizes)
    data = pd.DataFrame({'g': group, 'y': y, 'score': scores,
                         'old': frame.article_age_hours.gt(24).to_numpy(),
                         'age': frame.article_age_hours.to_numpy()})
    positives = np.bincount(group, weights=y, minlength=len(sizes))
    negatives = sizes - positives
    valid_auc = (positives > 0) & (negatives > 0)
    ranks = data.groupby('g')['score'].rank(method='average').to_numpy()
    rank_sums = np.bincount(group, weights=ranks * y, minlength=len(sizes))
    auc_numerator = rank_sums - positives * (positives + 1) / 2
    auc = np.mean(auc_numerator[valid_auc] /
                  (positives[valid_auc] * negatives[valid_auc])) if valid_auc.any() else None
    for name in ['y', 'old']:
        data[f'tie_{name}'] = data.groupby(['g', 'score'])[name].transform('mean')
    ranked = data.sort_values(['g', 'score'], ascending=[True, False], kind='stable')
    position = ranked.groupby('g').cumcount().to_numpy() + 1
    discount = 1 / np.log2(position + 1)
    result = {'mean_impression_auc': None if auc is None else float(auc),
              'auc_evaluable_impressions': int(valid_auc.sum()),
              'impressions': len(sizes), 'candidate_pairs': len(y)}
    valid_ndcg = positives > 0
    for k in [5, 10]:
        top = position <= k
        dcg = np.bincount(ranked.g.to_numpy(), weights=(
            ranked.tie_y.to_numpy() * discount * top), minlength=len(sizes))
        ideal_discounts = np.r_[0, np.cumsum(1 / np.log2(np.arange(1, k + 1) + 1))]
        ideal = ideal_discounts[np.minimum(positives, k).astype(int)]
        result[f'ndcg@{k}'] = float(np.mean(dcg[valid_ndcg] / ideal[valid_ndcg]))
        result[f'older_than_24h_share@{k}'] = float(ranked.tie_old.to_numpy()[top].mean())
    old = data.old.to_numpy()
    # These diagnostics concern observed clicks in the logged slates, not causal uplift.
    result['older_than_24h_candidate_share'] = float(old.mean())
    result['older_than_24h_click_share'] = float(y[old].sum() / max(y.sum(), 1))
    return result


def make_model():
    features = ColumnTransformer([
        ('numeric', StandardScaler(), NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='ignore', dtype=np.float32), CATEGORICAL),
    ], sparse_threshold=1.0)
    return Pipeline([
        ('features', features),
        ('model', LogisticRegression(C=1.0, solver='lbfgs', max_iter=300)),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    data_candidates = [project_root / 'data', project_root, project_root.parent / 'project']
    default_data = next((p for p in data_candidates if (p / 'articles.parquet').exists()), project_root / 'data')
    parser.add_argument('--root', type=Path, default=default_data)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--train-impressions', type=int, default=50000)
    parser.add_argument('--validation-impressions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.train_impressions < 0 or args.validation_impressions < 0:
        parser.error('Sample sizes must be nonnegative; 0 means all impressions')
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = {'python': sys.version, 'packages': {
        name: version(name) for name in
        ['numpy', 'pandas', 'pyarrow', 'scikit-learn', 'scipy', 'joblib']}}
    (args.output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    metadata = pd.read_parquet(args.root / 'articles.parquet', columns=[
        'article_id', 'category', 'article_type', 'published_time', 'premium',
    ]).set_index('article_id')
    if metadata.index.duplicated().any():
        raise ValueError('Duplicate article metadata IDs')
    metadata['category'] = metadata.category.astype('string').fillna('unknown')
    metadata['article_type'] = metadata.article_type.fillna('unknown')
    metadata['premium'] = metadata.premium.fillna(False)
    articles = metadata.to_dict()
    train, train_total = sample_impressions(
        args.root / 'train/behaviors.parquet', args.train_impressions, args.seed)
    val, val_total = sample_impressions(
        args.root / 'validation/behaviors.parquet', args.validation_impressions, args.seed + 1)
    if train.impression_time.max() >= val.impression_time.min():
        raise ValueError('Training and validation periods overlap')
    if np.intersect1d(train.impression_id, val.impression_id).size:
        raise ValueError('Training and validation impression IDs overlap')
    report = {'config': vars(args).copy(), 'sampling': {
        'train_population_impressions': train_total,
        'validation_population_impressions': val_total,
        'method': 'Uniform random-priority impression sampling; no candidate sampling',
    }, 'splits': {}}
    matrices = {}
    for name, behaviors in [('train', train), ('validation', val)]:
        cutoff = behaviors.impression_time.min()
        profiles, excluded = load_profiles(
            args.root / name / 'history.parquet', behaviors.user_id.unique(),
            cutoff, articles['category'])
        print(f'{name}: loaded {len(profiles):,} compact history profiles', flush=True)
        frame, y, sizes = build_features(behaviors, articles, profiles)
        del profiles
        matrices[name] = (frame, y, sizes)
        report['splits'][name] = {
            'impressions': len(behaviors), 'candidate_pairs': len(y),
            'time_min': cutoff, 'time_max': behaviors.impression_time.max(),
            'click_rate_in_sampled_slates': float(y.mean()),
            'history_cutoff_exclusive': cutoff,
            'history_events_excluded_at_or_after_cutoff_or_missing_time': excluded,
            'missing_article_pairs': int(frame.article_missing.sum()),
            'missing_age_pairs': int(frame.age_missing.sum()),
            'future_publication_pairs': int(frame.future_publication.sum()),
            'feature_table_megabytes': float(frame.memory_usage(deep=True).sum() / 1e6),
        }
        print(f'{name}: built {len(y):,} pairs', flush=True)
    train_x, train_y, _ = matrices['train']
    val_x, val_y, val_sizes = matrices['validation']
    if len(np.unique(train_y)) != 2:
        raise ValueError('Training requires both click labels')
    model = make_model()
    fit_started = time.perf_counter()
    model.fit(train_x, train_y)
    report['fit_seconds'] = time.perf_counter() - fit_started
    report['solver_iterations'] = int(model['model'].n_iter_[0])
    probabilities = model.predict_proba(val_x)[:, 1]
    methods = {
        # All scores tied = exact expected metrics under uniform random ordering.
        'random_expected': np.zeros(len(val_y)),
        'freshness': -val_x.article_age_hours.clip(lower=0).fillna(np.inf).to_numpy(),
        'category_affinity': val_x.category_affinity.to_numpy(),
        'logistic_regression': probabilities,
    }
    report['metrics'] = {}
    for name, scores in methods.items():
        report['metrics'][name] = evaluate(val_x, val_y, val_sizes, scores)
        print(name, json.dumps(report['metrics'][name]), flush=True)
    report['total_seconds'] = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_megabytes'] = peak / (1e6 if sys.platform == 'darwin' else 1024)
    report['notes'] = [
        'Evaluation uses whole sampled validation slates, not the full validation population unless sample size is 0.',
        'No model selection or hyperparameter tuning on the supplied validation period.',
        'Probabilities refer to the supplied logged candidate population; they are not site-wide CTR estimates.',
        'History snapshots are fixed before the earliest sampled impression of each split.',
        'No candidate age cutoff, post-impression features, catalog totals, or candidate-list position features.',
        'Session continuity and diversity are deferred until a click-ranking baseline is established.',
    ]
    with (args.output / 'report.json').open('w') as handle:
        json.dump(report, handle, indent=2, default=str)
    pd.DataFrame(report['metrics']).T.to_csv(args.output / 'metrics.csv')
    predictions = val_x[['impression_id', 'candidate_article_id', 'article_age_hours']].copy()
    predictions['clicked'] = val_y
    predictions['click_probability'] = probabilities
    predictions['rank'] = predictions.groupby('impression_id').click_probability.rank(
        method='min', ascending=False).astype('int32')
    predictions.to_parquet(args.output / 'validation_predictions.parquet', index=False)
    joblib.dump({'pipeline': model, 'numeric_features': NUMERIC,
                 'categorical_features': CATEGORICAL}, args.output / 'model.joblib')
    pd.DataFrame({
        'feature': model['features'].get_feature_names_out(),
        'coefficient': model['model'].coef_[0],
    }).to_csv(args.output / 'logistic_regression_coefficients.csv', index=False)
    print(f'Saved model and results to {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
