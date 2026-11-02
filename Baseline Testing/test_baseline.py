"""Small correctness checks for ranking ties, history cutoffs, and slate sampling."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import ndcg_score, roc_auc_score

from train_baseline import build_features, evaluate, load_profiles, sample_impressions


class BaselineChecks(unittest.TestCase):
    def test_ties_across_top_k_boundary_receive_expected_credit(self):
        y = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        frame = pd.DataFrame({'article_age_hours': [1] * 7 + [50]})
        actual = evaluate(frame, y, np.array([8]), np.zeros(8))
        self.assertAlmostEqual(actual['ndcg@5'],
                               ndcg_score([y], [np.zeros(8)], k=5))
        self.assertAlmostEqual(actual['older_than_24h_share@5'], 1 / 8)

    def test_metrics_match_reference_with_ties(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0])
        sizes = np.array([4, 3])
        scores = np.array([0.2, 0.8, 0.8, 0.8, 0.1, 0.5, 0.9])
        frame = pd.DataFrame({'article_age_hours': [1, 25, 3, 50, 4, 5, 100]})
        actual = evaluate(frame, y, sizes, scores)
        reference_auc = np.mean([roc_auc_score(y[:4], scores[:4]),
                                 roc_auc_score(y[4:], scores[4:])])
        self.assertAlmostEqual(actual['mean_impression_auc'], reference_auc)
        for k in [5, 10]:
            reference = np.mean([ndcg_score([y[:4]], [scores[:4]], k=k),
                                 ndcg_score([y[4:]], [scores[4:]], k=k)])
            self.assertAlmostEqual(actual[f'ndcg@{k}'], reference)
        random = evaluate(frame, y, sizes, np.zeros(7))
        self.assertAlmostEqual(random['mean_impression_auc'], 0.5)

    def test_history_excludes_equal_and_future_timestamps(self):
        cutoff = pd.Timestamp('2023-05-18 07:00:00')
        table = pa.table({
            'user_id': pa.array([1], type=pa.uint32()),
            'article_id_fixed': [[10, 20, 30]],
            'impression_time_fixed': [[cutoff - pd.Timedelta('1h'), cutoff,
                                       cutoff + pd.Timedelta('1h')]],
        })
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.parquet'
            pq.write_table(table, path)
            profiles, excluded = load_profiles(path, [1], cutoff, {10: 'news'})
        self.assertEqual(excluded, 2)
        self.assertEqual(profiles[1][0], 1)
        self.assertEqual(profiles[1][2].tolist(), [10])

    def test_old_and_missing_articles_are_kept(self):
        b = pd.DataFrame({
            'impression_id': [1], 'user_id': [7],
            'impression_time': [pd.Timestamp('2023-05-18')], 'device_type': [1],
            'article_ids_inview': [[10, 20, 30]], 'article_ids_clicked': [[20]],
        })
        articles = {'category': {10: 'news', 20: 'sport'},
                    'article_type': {10: 'article', 20: 'article'},
                    'published_time': {10: pd.Timestamp('2023-01-01'),
                                       20: pd.Timestamp('2023-05-17')},
                    'premium': {10: False, 20: True}}
        x, y, sizes = build_features(b, articles, {})
        self.assertEqual(x.candidate_article_id.tolist(), [10, 20, 30])
        self.assertEqual(y.tolist(), [0, 1, 0])
        self.assertEqual(sizes.tolist(), [3])
        self.assertGreater(x.article_age_hours.iloc[0], 24)
        self.assertEqual(x.age_missing.iloc[2], 1)

    def test_sampling_is_reproducible_and_preserves_slates(self):
        n = 30
        table = pa.table({
            'impression_id': list(range(n)), 'user_id': [1] * n,
            'impression_time': [pd.Timestamp('2023-05-18')] * n,
            'device_type': [1] * n, 'article_ids_inview': [[10, 20, 30]] * n,
            'article_ids_clicked': [[20]] * n,
        })
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'behaviors.parquet'
            pq.write_table(table, path)
            a, total = sample_impressions(path, 7, 42, batch_size=8)
            b, _ = sample_impressions(path, 7, 42, batch_size=8)
        self.assertEqual(total, n)
        self.assertEqual(len(a), 7)
        self.assertEqual(a.impression_id.tolist(), b.impression_id.tolist())
        self.assertTrue(all(list(ids) == [10, 20, 30] for ids in a.article_ids_inview))


if __name__ == '__main__':
    unittest.main()
