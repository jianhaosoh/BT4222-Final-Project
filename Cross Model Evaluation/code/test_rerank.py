"""Required invariant test for the Stage 2 reranker."""

import unittest

import pandas as pd

from rerank import rerank


class RerankerInvariantTests(unittest.TestCase):
    def test_zero_weights_and_lambda_one_preserve_stage1_top_five(self):
        request_time = pd.Timestamp("2023-05-29T12:00:00Z")
        article_ids = list(range(100, 106))
        articles = pd.DataFrame({
            "article_id": article_ids,
            "published_time": [request_time - pd.Timedelta(hours=i + 1) for i in article_ids],
            "topics": [[f"topic-{i % 3}"] for i in article_ids],
            "category": [i % 2 for i in article_ids],
        }).set_index("article_id", drop=False)
        rankings = pd.DataFrame({
            "behaviour_id": ["7:91"],
            "article_ids": [article_ids],
        })
        requests = pd.DataFrame({
            "behaviour_id": ["7:91"],
            "user_id": [7],
            "impression_id": [91],
            "impression_time": [request_time],
        })
        constants = {"s_same": 0.3, "s_diff": 0.1, "s_all": 0.2}

        result = rerank(
            rankings, requests, articles, constants,
            freshness_weight=0.0,
            continuity_weight=0.0,
            diversity_lambda=1.0,
            output_k=5,
        )

        self.assertEqual(result.loc[0, "article_ids"], article_ids[:5])


if __name__ == "__main__":
    unittest.main()
