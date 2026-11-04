"""Small CPU arithmetic checks; no project data, training or optimiser steps."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestFreshness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('huim_freshness') is not None:
            import huim_freshness
            cls.f = huim_freshness

    def setUp(self):
        self.assertTrue(hasattr(self, 'f'), 'HUIM freshness inference helper is required')

    def test_exact_float_and_id_order(self):
        x = torch.tensor([[0., -0., 1., 1., -2., float('-inf')]], dtype=torch.float32)
        x[0, 3] = torch.nextafter(x[0, 2], torch.tensor(float('inf')))
        ids = torch.tensor([4, 2, 8, 9, 1, 3])
        idx = self.f.exact_topk(x, ids, 6)
        self.assertEqual(ids[idx][0].tolist(), [9, 8, 2, 4, 1, 3])

    def test_zero_weight_and_age_penalty(self):
        x = torch.tensor([[1., 2., 3.]])
        age = torch.tensor([[0., 1., 24.]], dtype=torch.float64)
        self.assertTrue(torch.equal(self.f.blend(x, age, 0.), x))
        expected = x - 2 * torch.log1p(age).float()
        torch.testing.assert_close(self.f.blend(x, age, 2.), expected, rtol=0, atol=0)

    def test_scorer_matches_original_neural_and_blended_reference(self):
        from test_static import synthetic_world, make_model
        from huim.catalogue import CatalogueScorer
        _, _, _, _, pre, art, ctx = synthetic_world(n_articles=60, n_requests=4)
        model = make_model(pre, art)
        device = torch.device('cpu')
        reference = CatalogueScorer(model, ctx, art, device, pair_chunk=19, amp=False)
        scorer = self.f.FreshnessScorer(model, ctx, art, device, pair_chunk=31)
        requests = np.arange(4)
        neural = reference.logits(requests)
        ranks = scorer.rank_many(requests, [0., 2.], k=100)
        ids = art.ids[ctx.catalogue_idx]
        pub = art.tensors['pub_s'][ctx.catalogue_idx].numpy()
        ok = ~art.tensors['pub_missing'][ctx.catalogue_idx].numpy()
        for w, result in ranks.items():
            for row, pos in enumerate(requests):
                t = int(ctx.tensors['req_s'][pos])
                mask = ok & (pub <= t)
                age = np.maximum((t - pub).astype(np.float64) / 3600., 0)
                score = neural[row].numpy() - float(w) * np.log1p(age).astype(np.float32)
                order = np.lexsort((ids[mask], -score[mask]))
                expected_ids = ids[mask][order]
                n = len(expected_ids)
                self.assertEqual(int(result['lengths'][row]), n)
                np.testing.assert_array_equal(result['article_ids'][row, :n], expected_ids)
                np.testing.assert_allclose(result['scores'][row, :n], score[mask][order], atol=2e-6, rtol=0)

    def test_columnar_export_keeps_empty_requests_and_ranked_scores(self):
        import pandas as pd
        req = pd.DataFrame({'behaviour_id': ['4:1', '5:1'], 'user_id': [4, 5],
                            'impression_id': [1, 1], 'impression_time': pd.to_datetime(['2023-01-01']*2)})
        with tempfile.TemporaryDirectory() as tmp:
            writer = self.f.ShortlistWriter(Path(tmp), 'validation')
            result = {'article_ids': np.array([[9, 2], [2, 9]]),
                      'scores': np.array([[3., 2.], [-np.inf, -np.inf]], np.float32),
                      'lengths': np.array([2, 0]), 'eligible_count': np.array([2, 0])}
            writer.add(req, result)
            files = writer.close()
            rankings = pq.read_table(files['rankings']).to_pydict()
            self.assertEqual(rankings['article_ids'], [[9, 2], []])
            scores = pq.read_table(files['scores']).to_pydict()
            self.assertEqual(scores['article_id'], [9, 2])
            self.assertEqual(scores['rank'], [1, 2])
            self.assertNotIn('click_probability', scores)

    def test_macro_retrieval_keeps_ineligible_clicks_in_denominator(self):
        import pandas as pd
        requests = pd.DataFrame({'behaviour_id': ['1:2', '2:2'],
                                 'article_ids_clicked': [[8, 999], [9]]})
        metrics, rows = self.f.retrieval(requests, np.arange(2), np.array([[8, 7], [9, 4]]))
        self.assertEqual(metrics['known_click_recall@100'], .75)
        self.assertEqual(metrics['requests_evaluated'], 2)
        self.assertEqual(len(rows), 2)


if __name__ == '__main__':
    unittest.main()
