"""Publish existing full-catalogue SMALL vectors to the shared project folders."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def article_ids(path: Path) -> set[int]:
    ids = pq.read_table(path, columns=['article_id']).column('article_id').to_pylist()
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError(f'Expected unique, non-null article IDs: {path}')
    return set(ids)


def publish_embeddings(shared_root: Path, embeddings_dir: Path) -> dict:
    """Copy vectors without API calls, retaining missing rows and encoder provenance."""
    catalogue = shared_root / 'train_val/articles.parquet'
    expected = article_ids(catalogue)
    if article_ids(shared_root / 'test/articles.parquet') != expected:
        raise ValueError('Both project folders must contain the same SMALL catalogue; run Step 1 first.')
    metadata = json.loads((embeddings_dir / 'metadata.json').read_text())
    source_hash = metadata.get('catalogue_sha256')
    catalogue_hash = sha256(catalogue)
    if sha256(shared_root / 'test/articles.parquet') != catalogue_hash:
        raise ValueError('Both project folders must contain the same SMALL catalogue snapshot; run Step 1 first.')
    if source_hash and source_hash != catalogue_hash:
        raise ValueError('Embedding metadata refers to a different article catalogue.')
    counts, present = {}, {}
    for kind in ('text', 'image'):
        path = embeddings_dir / f'{kind}_embeddings.parquet'
        if article_ids(path) != expected:
            raise ValueError(f'{kind} embedding article IDs must cover the full SMALL catalogue, including missing vectors.')
        # Read scalar columns only: nullable fixed-size vector arrays need not be decoded.
        frame = pq.read_table(path, columns=['status', f'has_{kind}_embedding']).to_pydict()
        counts[kind] = dict(Counter(frame['status']))
        present[kind] = sum(frame[f'has_{kind}_embedding'])
    metadata.update(articles=len(expected), counts=counts,
                    catalogue_sha256=catalogue_hash,
                    scope='Complete EB-NeRD SMALL article catalogue shared by all project splits')
    for folder in ('train_val', 'test'):
        destination = shared_root / folder / 'article embeddings'
        destination.mkdir(parents=True, exist_ok=True)
        for kind in ('text', 'image'):
            source = embeddings_dir / f'{kind}_embeddings.parquet'
            target = destination / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        published = {**metadata, 'catalogue': f'{folder}/articles.parquet'}
        if folder == 'test':
            published.update(
                source_embeddings='train_val/article embeddings',
                new_text_vectors_generated=0,
                reused_text_vectors=present['text'], reused_image_vectors=present['image'],
                reuse_reason='Both article catalogues contain the same SMALL article IDs and text; '
                             'reuse their existing article-ID-keyed vectors, including nulls and flags.')
        (destination / 'metadata.json').write_text(json.dumps(published, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-root', type=Path, required=True)
    parser.add_argument('--embeddings-dir', type=Path, required=True)
    args = parser.parse_args()
    result = publish_embeddings(args.shared_root, args.embeddings_dir)
    print(f"Published {result['articles']:,} articles to train_val and test; no vectors generated.")
