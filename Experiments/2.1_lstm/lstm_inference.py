"""Batched float32 catalogue inference for the single LSTM freshness model.

Uses the original model's article encoder, cached histories and prediction head.
No labels enter scoring. Catalogue eligibility comes from the shared index's
microsecond bounds. Only small retained top-k blocks are transferred to CPU.
"""
import math
import time

import numpy as np
import pyarrow as pa
import torch

import lstm_data as D
import lstm_model as M


def ranking_keys(scores, id_rank, scale):
    """Exact descending float32 score, then ascending original article ID.

    IEEE float32 bits give an integer order; a lower-order ID rank breaks ties.
    No epsilon is added to scores. Signed zeros compare equally. ``scale`` is
    larger than every ID rank, so even adjacent float scores retain their order.
    """
    bits = scores.contiguous().view(torch.int32).to(torch.int64)
    ordered = torch.where(bits < 0, -(bits & 0x7fffffff), bits)
    return ordered * scale + (scale - 1 - id_rank)


def blend_scores(logits, age_hours, weight):
    return logits - float(weight) * torch.log1p(age_hours.clamp(min=0)).float()


class BatchedCatalogueScorer(M.CatalogueScorer):
    """Score request blocks against candidate tiles, retaining exact top-k.

    Defaults bound a tile to 16 requests x 8,192 articles (131,072 pairs).
    Histories and article states remain cached as in the reference scorer.
    Inference stays float32; batching does not introduce mixed precision.
    """

    def __init__(self, model, data, index, age_stats, pair_chunk=131072,
                 request_batch=16, requests=None):
        if model.cfg.variant != 'attention':
            raise ValueError('The improved run uses the attention LSTM variant.')
        super().__init__(model, data, index, age_stats, chunk=32768, requests=requests)
        self.pair_chunk, self.request_batch = int(pair_chunk), int(request_batch)
        self.request_users = data.imp_user_row.cpu().numpy()
        self.host_slots = self.user_slot.cpu().numpy()
        self.id_rank = torch.argsort(torch.argsort(self.ids))
        self.rank_scale = 1 << max(1, len(self.ids).bit_length())

    @torch.inference_mode()
    def _tiles(self, requests):
        requests = np.asarray(requests, dtype=np.int64)
        d, b = self.data, len(requests)
        rp = torch.as_tensor(requests, dtype=torch.long, device=self.device)
        users = self.request_users[requests]
        u = torch.as_tensor(users, dtype=torch.long, device=self.device)
        if self.precomputed:
            slots = torch.as_tensor(self.host_slots[users], dtype=torch.long, device=self.device)
            states, keys = self.states[slots], self.keys[slots]
        else:
            states = self.model.encode_history(d.hist_rows[u], d.hist_feats[u], d.hist_len[u])
            keys = self.model.key(states)
        lengths = d.hist_len[u]
        has_history = lengths > 0
        history_mask = torch.arange(states.shape[1], device=self.device)[None, :] < lengths[:, None]
        request_time = d.imp_time[rp]
        gap = ((torch.log1p(((request_time[:, None] - d.hist_time[u]).double() / 3600.)
                           .clamp(min=0)) - self.model.cfg.gap_mean) / self.model.cfg.gap_std).float()
        low = torch.as_tensor(self.lo[requests], device=self.device)
        high = torch.as_tensor(self.hi[requests], device=self.device)
        width = max(1, self.pair_chunk // b)
        for start in range(int(self.lo[requests].min()), int(self.hi[requests].max()), width):
            end = min(start + width, int(self.hi[requests].max()))
            c = end - start
            positions = torch.arange(start, end, device=self.device)
            age = ((request_time[:, None] - self.pub[None, start:end]).double() / 3600.).clamp(min=0)
            age_z = ((torch.log1p(age) - self.age_mean) / self.age_std).float()
            attention = torch.matmul(self.q[start:end].float()[None, :, :],
                                     keys.float().transpose(1, 2)) / math.sqrt(self.model.cfg.attention_dim)
            attention = attention + self.model.recency_weight * gap[:, None, :]
            attention = attention.masked_fill(~history_mask[:, None, :], -float('inf'))
            attention = torch.where(has_history[:, None, None], attention, torch.zeros_like(attention))
            weights = torch.softmax(attention, dim=-1) * history_mask[:, None, :]
            context = torch.matmul(weights, states.float())
            candidates = self.enc[None, start:end, :].expand(b, -1, -1).reshape(b * c, -1)
            features = torch.stack((age_z, torch.zeros_like(age_z)), dim=-1).reshape(b * c, 2)
            logits = self.model._score(
                candidates, context.reshape(b * c, -1),
                has_history[:, None].expand(-1, c).reshape(-1), features,
                d.imp_ctx[rp, None, :].expand(-1, c, -1).reshape(b * c, -1),
            ).float().reshape(b, c)
            eligible = (positions[None, :] >= low[:, None]) & (positions[None, :] < high[:, None])
            yield positions, logits, age, eligible

    @torch.inference_mode()
    def batches(self, requests, k=200, freshness_weights=(0.,)):
        requests = np.asarray(requests, dtype=np.int64)
        weights = tuple(dict.fromkeys(float(x) for x in freshness_weights))
        for offset in range(0, len(requests), self.request_batch):
            block = requests[offset:offset + self.request_batch]
            b = len(block)
            best = {w: (torch.empty((b, 0), dtype=torch.long, device=self.device),
                        torch.empty((b, 0), device=self.device)) for w in weights}
            finite = torch.ones((), dtype=torch.bool, device=self.device)
            for positions, logits, age, eligible in self._tiles(block):
                finite &= torch.isfinite(logits).all()
                for weight in weights:
                    old_pos, old_score = best[weight]
                    scores = blend_scores(logits, age, weight).masked_fill(~eligible, -float('inf'))
                    merged_pos = torch.cat((old_pos, positions[None, :].expand(b, -1)), dim=1)
                    merged_score = torch.cat((old_score, scores), dim=1)
                    keys = ranking_keys(merged_score, self.id_rank[merged_pos], self.rank_scale)
                    choose = torch.topk(keys, min(k + 1, keys.shape[1]), dim=1, sorted=True).indices
                    best[weight] = (merged_pos.gather(1, choose), merged_score.gather(1, choose))
            if not bool(finite):
                raise FloatingPointError('Nonfinite catalogue logits in a request block.')
            counts = (self.hi[block] - self.lo[block]).astype(np.int64)
            result = {}
            for weight, (positions, scores) in best.items():
                result[weight] = {'article_ids': self.ids[positions].cpu().numpy(),
                                  'scores': scores.cpu().numpy()}
            yield block, counts, result

    @torch.inference_mode()
    def full_scores(self, requests):
        """Small numerical benchmark only; not used by training or exports."""
        requests = np.asarray(requests, dtype=np.int64)
        out = torch.full((len(requests), len(self.ids)), -float('inf'), device=self.device)
        for pos, logits, _, eligible in self._tiles(requests):
            out[:, pos] = logits.masked_fill(~eligible, -float('inf'))
        return out


def recorded_scores(neural_logits, arrays, articles, freshness_weight):
    """The identical final score on original slates, including shown non-clicks."""
    request_time = np.repeat(arrays.imp_time_s, np.diff(arrays.offsets))
    age_hours = np.maximum((request_time - articles.published_s[arrays.cand_article_row]) / 3600., 0.)
    return (np.asarray(neural_logits, dtype=np.float32)
            - float(freshness_weight) * np.log1p(age_hours).astype(np.float32))


class ColumnarShortlistWriter(D.ShortlistWriter):
    """The same three Parquet schemas, using column arrays and larger row groups.

    Avoids creating one Python dictionary per retained article. Buffers at most
    1,024 requests (204,800 retained pairs at k=200) before a local-disk flush.
    """

    def __init__(self, run_dir, split, shortlist_size=200):
        super().__init__(run_dir, split, shortlist_size)
        self.pending = {key: [] for key in self.schemas}
        self.pending_requests = 0

    def add_batch(self, behaviour_ids, request_times_us, eligible_counts, ids, scores):
        lengths = np.minimum(self.k, np.asarray(eligible_counts, dtype=np.int64))
        ids, scores = np.asarray(ids)[:, :self.k], np.asarray(scores)[:, :self.k]
        valid = np.arange(ids.shape[1])[None, :] < lengths[:, None]
        offsets = np.r_[0, np.cumsum(lengths)].astype(np.int32)
        bids = pa.array(behaviour_ids, type=pa.string())
        flat_ids = pa.array(ids[valid], type=pa.int64())
        self.pending['ranks'].append(pa.Table.from_arrays([
            bids, pa.ListArray.from_arrays(pa.array(offsets), flat_ids),
        ], schema=self.schemas['ranks']))
        self.pending['scores'].append(pa.Table.from_arrays([
            bids.take(pa.array(np.repeat(np.arange(len(lengths)), lengths), type=pa.int64())),
            flat_ids,
            pa.array(np.broadcast_to(np.arange(1, ids.shape[1] + 1), ids.shape)[valid], type=pa.int32()),
            pa.array(scores[valid], type=pa.float32()),
        ], schema=self.schemas['scores']))
        self.pending['manifest'].append(pa.Table.from_arrays([
            bids, pa.array(request_times_us, type=pa.int64()),
            pa.array(eligible_counts, type=pa.int64()), pa.array(lengths, type=pa.int32()),
        ], schema=self.schemas['manifest']))
        self.requests += len(lengths)
        self.score_rows += int(lengths.sum())
        self.pending_requests += len(lengths)
        if self.pending_requests >= 1024:
            self.flush()

    def flush(self):
        for key, tables in self.pending.items():
            if tables:
                self.writers[key].write_table(pa.concat_tables(tables))
                tables.clear()
        self.pending_requests = 0


def synchronise(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_equivalence(reference, batched, requests, tolerance=1e-4):
    """Compare full eligible scores and top-100/200 on a small declared sample.

    The notebook runs this on Colab using the previous saved weights. It is not
    executed during notebook preparation. Timings exclude cache construction.
    """
    requests = np.asarray(requests, dtype=np.int64)
    # Warm the two paths before timing; weights and all dropout remain frozen.
    reference.topk(int(requests[0]), 200)
    next(batched.batches(requests[:batched.request_batch], 200, (0.,)))
    synchronise(reference.device)
    start = time.perf_counter()
    old = {}
    for i in requests:
        ids, scores = reference.score_request(int(i))
        old[int(i)] = (ids, scores)
    synchronise(reference.device)
    old_seconds = time.perf_counter() - start
    start = time.perf_counter()
    new = batched.full_scores(requests)
    synchronise(reference.device)
    new_seconds = time.perf_counter() - start
    difference, same = 0., True
    for row, i in enumerate(requests):
        ids, scores = old[int(i)]
        lo, hi = int(batched.lo[i]), int(batched.hi[i])
        actual = new[row, lo:hi]
        if len(scores):
            difference = max(difference, float((scores - actual).abs().max()))
        age = ((batched.data.imp_time[int(i)] - batched.pub[lo:hi]).double() / 3600.).clamp(min=0)
        for weight in (0., 2.):
            for k in (100, 200):
                a, _ = M.select_top_k(blend_scores(scores, age, weight), ids, k)
                b, _ = M.select_top_k(blend_scores(actual, age, weight), ids, k)
                same = same and torch.equal(a, b)
    synchronise(reference.device)
    start = time.perf_counter()
    old_top = {int(i): reference.topk(int(i), 200)[0].cpu().numpy() for i in requests}
    synchronise(reference.device)
    old_top_seconds = time.perf_counter() - start
    start = time.perf_counter()
    fast_top = {}
    for block, counts, result in batched.batches(requests, 200, (0.,)):
        for row, i in enumerate(block):
            fast_top[int(i)] = result[0.]['article_ids'][row, :min(200, counts[row])]
    synchronise(reference.device)
    fast_top_seconds = time.perf_counter() - start
    same = same and all(np.array_equal(old_top[int(i)], fast_top[int(i)]) for i in requests)
    result = {'requests': len(requests), 'max_abs_logit_difference': difference,
              'tolerance': tolerance, 'top100_top200_match_at_weights_0_and_2': same,
              'reference_score_seconds': old_seconds, 'batched_score_seconds': new_seconds,
              'score_only_speedup': old_seconds / new_seconds if new_seconds else None,
              'reference_top200_seconds': old_top_seconds,
              'batched_top200_seconds': fast_top_seconds,
              'top200_speedup': old_top_seconds / fast_top_seconds if fast_top_seconds else None,
              'passed': difference <= tolerance and same}
    if not result['passed']:
        raise AssertionError(f'Batched inference differs from reference: {result}')
    return result
