"""Frozen-HUIM freshness scoring and bounded columnar handoff exports.

The neural forward pass is the original saved HUIM implementation. Only the
ranking recipe changes. Float-bit top-k ordering follows the improved LSTM
helper: exact float32 score descending, original article ID ascending.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from huim.catalogue import CatalogueScorer
from huim.metrics import request_retrieval

FRESHNESS_GRID = (0., .25, .5, 1., 2., 4., 8., 16., 64.)
SCORE_SCHEMA = pa.schema([('behaviour_id', pa.string()), ('article_id', pa.int64()),
                          ('rank', pa.int32()), ('score', pa.float32())])
RANKING_SCHEMA = pa.schema([('behaviour_id', pa.string()), ('article_ids', pa.list_(pa.int64()))])
MANIFEST_SCHEMA = pa.schema([('behaviour_id', pa.string()), ('user_id', pa.int64()),
    ('impression_id', pa.int64()), ('impression_time', pa.string()),
    ('eligible_count', pa.int64()), ('shortlist_length', pa.int32())])


def exact_topk(scores, article_ids, k, id_rank=None, scale=None):
    """Indices of exact top-k; masked -inf values sort behind finite scores."""
    n = scores.shape[-1]
    if id_rank is None:
        id_rank = torch.argsort(torch.argsort(article_ids))
    if scale is None:
        scale = 1 << max(1, n.bit_length())
    bits = scores.contiguous().view(torch.int32).to(torch.int64)
    ordered = torch.where(bits < 0, -(bits & 0x7fffffff), bits)
    keys = ordered * scale + (scale - 1 - id_rank)
    return torch.topk(keys, min(k, n), dim=-1, sorted=True).indices


def blend(logits, age_hours, weight):
    return logits.float() - float(weight) * torch.log1p(age_hours.clamp_min(0)).float()


class FreshnessScorer(CatalogueScorer):
    """Reuse cached HUIM states; score once, rank every requested coefficient.

    The original HUIM pair chunks bound attention memory. The [request, article]
    logit matrix is only ~8 MB for 16 requests x 125,541 articles. Full logits
    stay on the GPU and are discarded after each request block.
    """
    def __init__(self, model, ctx, art, device, pair_chunk=131072):
        super().__init__(model, ctx, art, device, pair_chunk=pair_chunk, amp=False)
        self.ids = torch.as_tensor(art.ids[ctx.catalogue_idx], device=device)
        self.id_rank = torch.argsort(torch.argsort(self.ids))
        self.scale = 1 << max(1, len(self.ids).bit_length())

    @torch.inference_mode()
    def age_and_mask(self, positions):
        rp = torch.as_tensor(positions, dtype=torch.int64, device=self.device)
        request_time = self.ctx.tensors['req_s'][rp]
        eligible = self.pub_ok[None, :] & (self.pub[None, :] <= request_time[:, None])
        age = ((request_time[:, None] - self.pub[None, :]).double() / 3600.).clamp_min(0)
        age = age.masked_fill(~self.pub_ok[None, :], 0.)
        return age, eligible

    @torch.inference_mode()
    def rank_many(self, positions, weights, k=200):
        logits = self.logits(np.asarray(positions, np.int64))
        age, eligible = self.age_and_mask(positions)
        counts = eligible.sum(1)
        results = {}
        for weight in weights:
            score = blend(logits, age, weight).masked_fill(~eligible, float('-inf'))
            order = exact_topk(score, self.ids, k, self.id_rank, self.scale)
            results[float(weight)] = {
                'article_ids': self.ids[order].cpu().numpy(),
                'scores': score.gather(1, order).cpu().numpy(),
                'lengths': counts.clamp_max(k).cpu().numpy(),
                'eligible_count': counts.cpu().numpy(),
            }
        return results

    @torch.inference_mode()
    def newest(self, positions, k=200):
        _, eligible = self.age_and_mask(positions)
        keys = self.pub[None, :] * self.scale + (self.scale - 1 - self.id_rank)
        keys = keys.expand(len(positions), -1).masked_fill(~eligible, torch.iinfo(torch.int64).min)
        order = torch.topk(keys, min(k, len(self.ids)), dim=1, sorted=True).indices
        return self.ids[order].cpu().numpy(), eligible.sum(1).clamp_max(k).cpu().numpy()


def retrieval(requests, positions, ranked_ids, lengths=None):
    """Macro known-click recovery at 100; all original clicks stay in ideals."""
    rows = []
    for j, pos in enumerate(positions):
        r = requests.iloc[int(pos)]
        ids = ranked_ids[j] if lengths is None else ranked_ids[j, :int(lengths[j])]
        n, hits, dcg, ideal = request_retrieval(ids, r.article_ids_clicked, 100)
        rows.append({'behaviour_id': str(r.behaviour_id), 'n_clicked': n,
                     'hits@100': hits, 'known_click_recall@100': hits/n if n else np.nan,
                     'observed_click_ndcg@100': dcg/ideal if ideal else np.nan})
    frame = pd.DataFrame(rows)
    return summarise_retrieval(frame), frame


def summarise_retrieval(frame):
    return {'known_click_recall@100': float(frame['known_click_recall@100'].mean()),
            'observed_click_ndcg@100': float(frame['observed_click_ndcg@100'].mean()),
            'requests_evaluated': int(frame.n_clicked.gt(0).sum()),
            'requests_without_known_clicks': int(frame.n_clicked.eq(0).sum()),
            'clicked_articles_total': int(frame.n_clicked.sum())}


class ShortlistWriter:
    """Same generic score/list handoff as LSTM, with no pseudo-probabilities."""
    def __init__(self, root, split):
        root = Path(root)
        self.paths = {
            'scores': root/'predictions'/split/'submission_article_scores.parquet',
            'rankings': root/'rankings'/split/'submission_top_200.parquet',
            'manifest': root/'diagnostics'/'catalogue'/f'{split}_requests.parquet',
        }
        schemas = dict(scores=SCORE_SCHEMA, rankings=RANKING_SCHEMA, manifest=MANIFEST_SCHEMA)
        self.writers = {}
        for name, path in self.paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            self.writers[name] = pq.ParquetWriter(path, schemas[name], compression='zstd')
        self.pending = []
        self.n_requests = self.n_scores = self.eligible_pairs = 0

    def add(self, requests, result):
        ids, scores = np.asarray(result['article_ids']), np.asarray(result['scores'], np.float32)
        lens = np.asarray(result['lengths'], np.int64)
        counts = np.asarray(result['eligible_count'], np.int64)
        width = ids.shape[1]
        keep = np.arange(width)[None, :] < lens[:, None]
        assert np.isfinite(scores[keep]).all()
        assert np.array_equal(lens, np.minimum(200, counts))
        # A single request's prefix cannot be longer than the available catalogue.
        assert np.all(lens <= width)
        bids = requests.behaviour_id.astype(str).to_numpy()
        score_table = pa.Table.from_pydict({
            'behaviour_id': np.repeat(bids, lens), 'article_id': ids[keep],
            'rank': np.broadcast_to(np.arange(1, width+1), ids.shape)[keep].astype(np.int32),
            'score': scores[keep]}, schema=SCORE_SCHEMA)
        offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
        lists = pa.ListArray.from_arrays(pa.array(offsets), pa.array(ids[keep], type=pa.int64()))
        ranking_table = pa.Table.from_arrays([pa.array(bids), lists], schema=RANKING_SCHEMA)
        manifest_table = pa.Table.from_pydict({
            'behaviour_id': bids, 'user_id': requests.user_id.to_numpy(np.int64),
            'impression_id': requests.impression_id.to_numpy(np.int64),
            'impression_time': requests.impression_time.astype(str).tolist(),
            'eligible_count': counts, 'shortlist_length': lens}, schema=MANIFEST_SCHEMA)
        self.pending.append(dict(scores=score_table, rankings=ranking_table, manifest=manifest_table))
        self.n_requests += len(requests)
        self.n_scores += int(lens.sum())
        self.eligible_pairs += int(counts.sum())
        if sum(len(x['rankings']) for x in self.pending) >= 1024:
            self.flush()

    def flush(self):
        if self.pending:
            for name, writer in self.writers.items():
                writer.write_table(pa.concat_tables([part[name] for part in self.pending]))
            self.pending.clear()

    def close(self):
        self.flush()
        for writer in self.writers.values():
            writer.close()
        return self.paths


def recorded_age(ctx, art):
    rec = ctx.recorded
    t = ctx.tensors['req_s'].detach().cpu().numpy()[rec['req_pos']]
    pub = art.tensors['pub_s'].detach().cpu().numpy()[rec['art_idx']]
    missing = art.tensors['pub_missing'].detach().cpu().numpy()[rec['art_idx']]
    age = np.maximum((t-pub).astype(np.float64)/3600., 0)
    age[missing] = 0
    return age


def write_recorded(path, ctx, raw_logits, age, weight):
    scores = blend(torch.as_tensor(raw_logits), torch.as_tensor(age), weight).numpy()
    rec = ctx.recorded
    schema = pa.schema([('behaviour_id', pa.string()), ('article_id', pa.int64()), ('score', pa.float32())])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, schema, compression='zstd') as writer:
        for start in range(0, len(scores), 100_000):
            end = min(start+100_000, len(scores))
            writer.write_table(pa.Table.from_pydict({
                'behaviour_id': ctx.requests.behaviour_id.to_numpy()[rec['req_pos'][start:end]],
                'article_id': rec['article_id'][start:end], 'score': scores[start:end]}, schema=schema))
    return scores
