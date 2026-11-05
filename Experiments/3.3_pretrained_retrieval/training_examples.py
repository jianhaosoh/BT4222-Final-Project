"""Prepare sampled, observed-slate training examples for a frozen-embedding retriever.

No model is fitted. Validation/test behaviour labels are never read.
Pass the shared catalogue_pipeline module as `cp` to reuse exact ID/history rules.
"""
from contextlib import closing
import hashlib
import heapq
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = 'observed_slate_examples_v1'
ARTIFACTS = ['training_examples.parquet', 'user_profiles.parquet', 'article_vectors.parquet',
             'training_config.json', 'training_sources.json', 'training_summary.json']


def _path(config, name, choices):
    root = Path(config['drive_root'])
    override = config.get('paths', {}).get(name)
    candidates = ([Path(override)] if override else [root / p for p in choices])
    if override and not candidates[0].is_absolute():
        candidates = [root / candidates[0]]
    found = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(found) != 1:
        raise FileNotFoundError(f'{name}: expected one source, found {found}; set CONFIG["paths"]["{name}"]')
    return found[0]


def resolve_sources(config):
    """Resolve only training behaviour/history plus the declared article lookups."""
    sources = {
        'features_train': _path(config, 'features_train', ['feature_engineering/train_features.parquet']),
        'history_train': _path(config, 'history_train', ['train_val/train/history.parquet']),
    }
    for part in ['train_val', 'test']:
        sources[f'catalogue_{part}'] = _path(config, f'catalogue_{part}', [f'{part}/articles.parquet'])
        sources[f'embeddings_{part}'] = _path(config, f'embeddings_{part}', [
            f'{part}/article embeddings/text_embeddings.parquet',
            f'embeddings/{part}/text_embeddings.parquet'])
    return sources


def sample_negatives(candidates, *, seed, behaviour_id, positive_id, maximum):
    """Per-positive deterministic sampling without replacement; no global RNG state."""
    def priority(aid):
        payload = f'{seed}:negative:{behaviour_id}:{positive_id}:{aid}'.encode()
        return hashlib.blake2b(payload, digest_size=8).digest(), aid
    return heapq.nsmallest(maximum, set(candidates), key=priority)


def build(config, work, cp):
    if config.get('train_requests', 50_000) is not None and (
        type(config.get('train_requests', 50_000)) is not int or config.get('train_requests', 50_000) < 1
    ):
        raise ValueError('train_requests must be positive or None')
    for name, default in [('negatives_per_positive', 4), ('history_limit', 10)]:
        if type(config.get(name, default)) is not int or config.get(name, default) < 1:
            raise ValueError(f'{name} must be a positive integer')
    config = json.loads(json.dumps(config, default=str))
    seed = int(config.get('seed', 42))
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    sources = resolve_sources(config)
    fingerprints = {key: cp.fingerprint(path, config.get('hash_inputs', False)) for key, path in sources.items()}
    signature = {'version': VERSION, 'pipeline_version': cp.PIPELINE_VERSION, 'config': config,
                 'sources': fingerprints}
    state = work / 'training_sources.json'
    if state.exists() and json.loads(state.read_text()) != signature:
        raise ValueError('Training inputs/settings changed; use a new run ID')
    if (work / 'training_summary.json').exists():
        if not all((work / name).is_file() for name in ARTIFACTS):
            raise ValueError('Completed dataset bundle is incomplete; use a fresh run ID')
        summary = json.loads((work / 'training_summary.json').read_text())
        print('Reusing completed training dataset:', work, flush=True)
        return summary
    estimated_index_bytes = fingerprints['features_train']['rows'] * 320
    free = shutil.disk_usage(work).free
    if estimated_index_bytes > config.get('max_index_gb', 20) * 1e9 or estimated_index_bytes > free * 0.7:
        raise RuntimeError('Training-only disk index estimate exceeds budget/free space')
    cp.save_json(state, signature)
    index_dir = work / '_indexes'; index_dir.mkdir(exist_ok=True)
    db_path = index_dir / 'train.sqlite'; indexed = index_dir / 'train.json'
    if indexed.exists():
        index_info = json.loads(indexed.read_text())
    else:
        if db_path.exists(): db_path.unlink()  # Only this incomplete generated index.
        print('Indexing existing training slates; no catalogue-wide pair expansion.', flush=True)
        index_info = cp.index_shared(sources['features_train'], 'train', db_path, seed)
        cp.save_json(indexed, index_info)
    selected = list(cp.request_iter(db_path, config.get('train_requests', 50_000)))
    users = {int(req[1]) for req in selected}
    vectors, embedding_info = cp.load_embeddings(
        [sources['embeddings_train_val'], sources['embeddings_test']],
        config.get('embedding_column', 'text_embedding'), config.get('embedding_dimension'))
    catalogue, overlap = cp.load_catalogue([sources['catalogue_train_val'], sources['catalogue_test']])
    dimension = embedding_info['dimension']
    profile_path = index_dir / 'training_profiles.sqlite'
    if profile_path.exists(): profile_path.unlink()
    profiles, history_info = cp.history_profiles(
        sources['history_train'], users, index_info['split_start'], vectors,
        output_db=profile_path, history_limit=config.get('history_limit', 10), aggregation='mean_profile')
    # Do not use sampled-request timestamps as a looser history cutoff.
    # The source history is required to precede the FIRST request of the full train split.
    schema = pa.schema([
        ('example_id', pa.int64()), ('behaviour_id', pa.string()),
        ('user_id', pa.int64()), ('impression_id', pa.int64()), ('impression_time', pa.string()),
        ('positive_article_id', pa.int64()), ('negative_article_ids', pa.list_(pa.int64())),
        ('negative_count', pa.int32()), ('profile_coverage', pa.float32()),
    ])
    counts = dict(selected_requests=len(selected), requests_with_examples=0,
                  requests_without_positive_clicks=0, requests_without_usable_profile=0,
                  requests_without_usable_negative=0, positive_clicks_seen=0,
                  positive_clicks_excluded_missing_vector=0, positive_clicks_excluded_by_eligibility=0,
                  negative_candidates_excluded_missing_vector=0, negative_candidates_excluded_by_eligibility=0,
                  examples=0, sampled_negative_pairs=0)
    used_users = set(); used_articles = set(); request_ids = []; started = time.perf_counter()
    # Precompute metadata once rather than rebuilding a 124k-article pool for each slate.
    published = dict(zip(catalogue.index.astype(int), catalogue.published_time))
    import pandas as pd
    def eligible(aid, when):
        value = published.get(aid)
        return value is not None and not pd.isna(value) and value <= pd.Timestamp(when)
    example_tmp = work / 'training_examples.tmp.parquet'
    pending = []
    try:
        with closing(sqlite3.connect(db_path)) as db, pq.ParquetWriter(example_tmp, schema) as writer:
            for number, req in enumerate(selected, 1):
                key, user, impression, when, _ = req
                request_ids.append(key)
                rows = db.execute('SELECT article_id,clicked FROM candidates WHERE behaviour_id=? ORDER BY article_id', (key,)).fetchall()
                all_positive = {int(a) for a, clicked in rows if clicked == 1}
                counts['positive_clicks_seen'] += len(all_positive)
                if not all_positive:
                    counts['requests_without_positive_clicks'] += 1
                    continue
                profile, coverage = profiles.get(int(user), (None, 0))
                if profile is None:
                    counts['requests_without_usable_profile'] += 1
                    continue
                negative_ids = []
                for aid, label in rows:
                    aid = int(aid)
                    if label != 0 or aid in all_positive: continue
                    if not eligible(aid, when):
                        counts['negative_candidates_excluded_by_eligibility'] += 1
                    elif aid not in vectors:
                        counts['negative_candidates_excluded_missing_vector'] += 1
                    else: negative_ids.append(aid)
                positive_ids = []
                for aid in sorted(all_positive):
                    if not eligible(aid, when):
                        counts['positive_clicks_excluded_by_eligibility'] += 1
                    elif aid not in vectors:
                        counts['positive_clicks_excluded_missing_vector'] += 1
                    else: positive_ids.append(aid)
                if not negative_ids:
                    counts['requests_without_usable_negative'] += 1
                    continue
                made = False
                for positive in positive_ids:
                    negatives = sample_negatives(negative_ids, seed=seed, behaviour_id=key,
                        positive_id=positive, maximum=config.get('negatives_per_positive', 4))
                    if all_positive.intersection(negatives): raise AssertionError('Clicked article selected as negative')
                    pending.append(dict(example_id=counts['examples'], behaviour_id=key, user_id=int(user),
                        impression_id=int(impression), impression_time=when, positive_article_id=positive,
                        negative_article_ids=negatives, negative_count=len(negatives), profile_coverage=coverage))
                    counts['examples'] += 1; counts['sampled_negative_pairs'] += len(negatives)
                    used_articles.add(positive); used_articles.update(negatives); used_users.add(int(user)); made = True
                    if len(pending) >= 1024:
                        writer.write_table(pa.Table.from_pylist(pending, schema=schema)); pending.clear()
                counts['requests_with_examples'] += made
                if number % 1000 == 0:
                    print(f'Requests processed: {number:,}/{len(selected):,}; examples: {counts["examples"]:,}', flush=True)
            if pending: writer.write_table(pa.Table.from_pylist(pending, schema=schema))
        if not counts['examples']:
            raise ValueError('No usable examples. Inspect profiles, positive labels, article/vector coverage and displayed negatives.')
        example_tmp.replace(work / 'training_examples.parquet')
        profile_schema = pa.schema([('user_id', pa.int64()), ('profile_embedding', pa.list_(pa.float32())),
                                    ('history_vector_coverage', pa.float32())])
        with pq.ParquetWriter(work / 'user_profiles.tmp.parquet', profile_schema) as writer:
            ids = sorted(used_users)
            for start in range(0, len(ids), 512):
                block = [{'user_id': user, 'profile_embedding': profiles.get(user)[0].tolist(),
                          'history_vector_coverage': profiles.get(user)[1]} for user in ids[start:start+512]]
                writer.write_table(pa.Table.from_pylist(block, schema=profile_schema))
        (work / 'user_profiles.tmp.parquet').replace(work / 'user_profiles.parquet')
    finally:
        profiles.close()
    article_schema = pa.schema([('article_id', pa.int64()), ('text_embedding', pa.list_(pa.float32()))])
    with pq.ParquetWriter(work / 'article_vectors.tmp.parquet', article_schema) as writer:
        ids = sorted(used_articles)
        for start in range(0, len(ids), 512):
            writer.write_table(pa.Table.from_pylist([
                {'article_id': aid, 'text_embedding': vectors[aid].tolist()} for aid in ids[start:start+512]], schema=article_schema))
    (work / 'article_vectors.tmp.parquet').replace(work / 'article_vectors.parquet')
    recipe = dict(signature=signature, dataset_kind='sampled_observed_slate_training',
        user_profile='normalized equal mean of available latest-N train-history vectors',
        history_limit=config.get('history_limit', 10), history_cutoff=index_info['split_start'],
        selection='smallest BLAKE2b(seed:train:composite request ID); complete recorded slates',
        negatives='up to N displayed-but-unclicked articles from the SAME request; no clicked article may be negative',
        multiple_clicks='one example per usable positive; all request positives excluded from negatives',
        negative_sampling='deterministic BLAKE2b priority per seed/request/positive/negative; without replacement',
        fewer_negatives='keep fewer; downstream loss must mask padding, not invent negatives',
        training_coverage='exclude missing profiles, unavailable vectors and publication-ineligible training candidates; counts reported',
        inference_coverage='this training filter does not change full-catalogue inference eligibility or missing-vector fallbacks',
        snapshot_policy='train_val first, test second; shared frozen article lookups only; no validation/test behaviour labels',
        outcome_caution='unclicked displayed articles are imperfect training signals; unshown outcomes are unknown',
        learned_parameters=False, transformer_fine_tuned=False, model_training_performed=False)
    cp.save_json(work / 'training_config.json', recipe)
    summary = dict(version=VERSION, status='complete', counts=counts,
        embedding_dimension=dimension, saved_users=len(used_users), saved_articles=len(used_articles),
        history_coverage=history_info, catalogue_overlap_first_wins=overlap,
        selected_request_ids_sha256=hashlib.sha256(json.dumps(request_ids).encode()).hexdigest(),
        selected_request_ids=request_ids, seconds=round(time.perf_counter()-started, 3),
        training_candidate_rows_indexed=index_info['candidate_rows'], estimated_index_bytes=estimated_index_bytes,
        model_training_performed=False)
    cp.save_json(work / 'training_summary.json', summary)
    print('Training dataset ready:', work, flush=True)
    return summary


def persist(work, destination):
    """Copy the portable training dataset, not the large generated SQLite indexes."""
    work = Path(work); destination = Path(destination)
    if destination.resolve() == work.resolve(): return destination
    if not all((work / name).is_file() for name in ARTIFACTS):
        raise ValueError('Prepare the complete training dataset before copying')
    if destination.exists():
        raise FileExistsError('Persistent destination exists; choose a new destination/run ID')
    destination.mkdir(parents=True)
    for name in ARTIFACTS:
        shutil.copy2(work / name, destination / name)
        if (destination / name).stat().st_size != (work / name).stat().st_size:
            raise RuntimeError(f'Incomplete training artifact copy: {name}')
    return destination
