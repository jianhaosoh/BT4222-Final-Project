"""Small numerical checks; no dataset, fitted model or training is used."""
import unittest
import tempfile
from types import MethodType, SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch

import lstm_inference as F
import lstm_model as M


class RankingTests(unittest.TestCase):
    def test_exact_score_then_id_order(self):
        ids = torch.tensor([90, 11, 42, 7, 100, 3, 19, 6])
        rank = torch.argsort(torch.argsort(ids))
        scores = torch.tensor([
            [1., 1., -2., -0., 0., -float('inf'), 3., 1.],
            [-1., -1., -1., -1., -1., -1., -1., -1.],
        ])
        keys = F.ranking_keys(scores, rank, 16)
        order = torch.topk(keys, 5, dim=1, sorted=True).indices
        for row in range(len(scores)):
            expected = np.lexsort((ids.numpy(), -scores[row].numpy()))[:5]
            np.testing.assert_array_equal(order[row].numpy(), expected)

    def test_adjacent_float_scores_are_never_reversed_by_id(self):
        low = torch.tensor(1., dtype=torch.float32)
        high = torch.nextafter(low, torch.tensor(float('inf')))
        scores = torch.stack([low, high]).reshape(1, 2)
        keys = F.ranking_keys(scores, torch.tensor([0, 1]), 4)
        self.assertGreater(keys[0, 1].item(), keys[0, 0].item())

    def test_freshness_and_zero_weight(self):
        logits = torch.tensor([[2., 2., 3.]])
        age = torch.tensor([[0., 1., 24.]])
        torch.testing.assert_close(F.blend_scores(logits, age, 0.), logits)
        torch.testing.assert_close(F.blend_scores(logits, age, 2.),
                                   logits - 2. * torch.log1p(age))

    def test_columnar_exports_preserve_short_and_empty_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = F.ColumnarShortlistWriter(folder, 'validation', 200)
            writer.add_batch(['1:2', '3:4', '5:6'], np.array([1, 2, 3]), np.array([2, 1, 0]),
                             np.array([[8, 9], [7, 0], [0, 0]]),
                             np.array([[2., 1.], [3., -np.inf], [-np.inf, -np.inf]]))
            paths = writer.close()
            ranks = pq.read_table(paths['rankings']).to_pydict()
            self.assertEqual(ranks['article_ids'], [[8, 9], [7], []])
            pairs = pq.read_table(paths['scores']).to_pydict()
            self.assertEqual(pairs['behaviour_id'], ['1:2', '1:2', '3:4'])
            self.assertEqual(pairs['rank'], [1, 2, 1])
            self.assertEqual(pairs['score'], [2., 1., 3.])
            self.assertEqual(pq.read_table(paths['manifest']).num_rows, 3)

    def test_batched_tiles_match_reference_on_tiny_arithmetic_fixture(self):
        # Deterministic tensor functions, not a fitted neural model or dataset.
        enc = torch.arange(21, dtype=torch.float32).reshape(7, 3) / 13
        projection = torch.tensor([[.1, .2], [.3, -.1], [-.2, .1]])
        model = SimpleNamespace(cfg=SimpleNamespace(variant='attention', article_dim=3,
            attention_dim=2, lstm_hidden=2, gap_mean=0., gap_std=1.), recency_weight=torch.tensor(.3))
        model.eval = lambda: model
        model.articles = SimpleNamespace(encode_rows=lambda rows: enc[rows])
        model.query = lambda x: x @ projection
        model.key = lambda x: x * .7
        model.encode_history = lambda rows, features, lengths: enc[rows, :2].cumsum(1)
        model._score = lambda cand, ctx, has, features, request: (
            cand.sum(-1) + ctx.sum(-1) * .2 + features[:, 0] * .1 + request[:, 0])
        model.score_one_request = MethodType(M.TargetAwareLSTMRanker.score_one_request, model)
        request_us = np.array([7200000123, 18000000999, 3600000000, -1, 21600000000], dtype=np.int64)
        pub = np.arange(6, dtype=np.int64) * 3600
        index = SimpleNamespace(rows=np.arange(1, 7), article_ids=np.array([30, 20, 15, 25, 10, 50]),
                                published_s=pub)
        index.bounds = lambda times, max_age=None: (
            np.zeros(len(times), dtype=np.int64), np.searchsorted(pub * 1_000_000, times, side='right'))
        data = SimpleNamespace(device=torch.device('cpu'),
            imp_time_us=request_us, imp_time=torch.tensor(request_us // 1_000_000),
            imp_user_row=torch.tensor([0, 1, 0, 2, 2]), imp_ctx=torch.arange(5).float()[:, None],
            hist_rows=torch.tensor([[1, 2], [3, 0], [0, 0]]),
            hist_feats=torch.zeros(3, 2, 1), hist_len=torch.tensor([2, 1, 0]),
            hist_time=torch.tensor([[-7200, -3600], [-3600, 0], [0, 0]]))
        slow = M.CatalogueScorer(model, data, index, {'mean': 0., 'std': 1.}, chunk=2)
        fast = F.BatchedCatalogueScorer(model, data, index, {'mean': 0., 'std': 1.},
                                         pair_chunk=6, request_batch=2)
        for block, counts, output in fast.batches(np.arange(5), 3, (0., 2.)):
            for row, i in enumerate(block):
                ids, logits = slow.score_request(int(i))
                age = ((data.imp_time[i] - slow.pub[:len(ids)]).double() / 3600.).clamp(min=0)
                for weight in (0., 2.):
                    expected_ids, expected_scores = M.select_top_k(
                        F.blend_scores(logits, age, weight), ids, 3)
                    count = min(3, int(counts[row]))
                    np.testing.assert_array_equal(output[weight]['article_ids'][row, :count],
                                                  expected_ids.numpy())
                    np.testing.assert_allclose(output[weight]['scores'][row, :count],
                                               expected_scores.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
