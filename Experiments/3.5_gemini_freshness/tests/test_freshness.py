"""Synthetic scoring/eligibility checks, not shared-data performance claims."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import catalogue_pipeline as cp
import freshness_pipeline as fp
fixture_path=Path(__file__).resolve().parents[2]/'3.1_pretrained_similarity_gemini/tests/test_similarity.py'
spec=importlib.util.spec_from_file_location('_similarity_fixture',fixture_path)
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


def settings(root,shared):
    return dict(fixtures.config(root,shared),history_limit=10,history_aggregation='topk_history',
                history_top_k=3,validation_requests=2,freshness_weights=[0.,.01,.03,.1])


class FreshnessChecks(unittest.TestCase):
    def test_metrics_stop_at_100_and_use_correct_ndcg_ideal(self):
        ranking=list(range(1,201))
        self.assertEqual(fp.retrieval_metrics(ranking,{101,200}),(0.,0.))
        recall,ndcg=fp.retrieval_metrics(ranking,set(range(1,151)))
        self.assertAlmostEqual(recall,100/150)
        self.assertAlmostEqual(ndcg,1.)
        self.assertIsNone(fp.retrieval_metrics(ranking,set()))

    def test_soft_age_penalty_uses_request_time_and_keeps_old_articles(self):
        catalogue=pd.DataFrame({'article_id':[1,2,3,4,5],
            'published_time':pd.to_datetime(['2023-01-01','2023-01-03 23:00','2023-01-05',None,'2023-01-03 23:30'],format='mixed')}).set_index('article_id',drop=False)
        vectors={1:np.array([1.,0.],dtype='float32'),2:np.array([.8,.6],dtype='float32')}
        profiles=cp.HistoryProfiles({7:(np.array([[1.,0.]],dtype='float32'),1)},aggregation='topk_history',top_k=3)
        options=fp.variants({'freshness_weights':[0,.1]})
        request=('7:10',7,10,'2023-01-04','[]')
        ranked,eligible=fp.rank_request(request,catalogue,profiles,vectors,options,batch_size=2)
        self.assertEqual(eligible.tolist(),[1,2,5])
        self.assertEqual(ranked['similarity_only']['article_ids'],[1,2,5])
        self.assertEqual(ranked['similarity_lambda_0p1']['article_ids'],[2,1,5])
        self.assertEqual(ranked['freshness_only']['article_ids'],[5,2,1])
        blend=ranked['similarity_lambda_0p1']
        values=dict(zip(blend['article_ids'],blend['scores']))
        self.assertAlmostEqual(values[1],1-.1*np.log1p(72),places=6)
        self.assertAlmostEqual(values[2],.8-.1*np.log1p(1),places=6)
        self.assertIn(1,blend['article_ids'])  # No 24-hour cliff.
        missing=ranked['similarity_lambda_0p1']['article_ids'].index(5)
        self.assertEqual(blend['similarity_scores'][missing],0.)

    def test_ties_at_boundary_are_deterministic_and_missing_profiles_retained(self):
        ids=np.arange(1,251)
        catalogue=pd.DataFrame({'article_id':ids,'published_time':pd.to_datetime(['2023-01-01']*250)}).set_index('article_id',drop=False)
        ranked,_=fp.rank_request(('7:10',7,10,'2023-01-02','[]'),catalogue,{}, {},fp.variants({}),batch_size=17)
        for value in ranked.values():
            self.assertEqual(value['article_ids'],list(range(1,201)))
            self.assertTrue(value['ties_crossing_200'])
            self.assertTrue(value['ties_crossing_100'])
        self.assertTrue(all(score< -1 for score in ranked['freshness_only']['scores']))

    def test_grid_matches_original_baseline_and_reuses_same_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixtures.fixture(root)
            prepared=cp.prepare(settings(root,shared),root/'work')
            grid=fp.evaluate_grid(prepared)
            profiles,_=cp.profiles_for(prepared,'validation')
            try:
                original=cp.evaluate_catalogue(cp.model_spec(prepared['config']),prepared['validation_requests'],
                    prepared['dbs']['validation'],prepared['catalogue'],profiles,prepared['vectors'],2)
            finally:profiles.close()
            baseline=next(row['metrics'] for row in grid['results'] if row['variant']['name']=='similarity_only')
            for key in ['known_click_recall','observed_click_ndcg']:
                self.assertAlmostEqual(baseline[key+'@100'],original[key+'@200'])
            for key in ['known_clicks_excluded_by_eligibility','known_click_eligibility_coverage']:
                self.assertAlmostEqual(baseline[key],original[key])
            self.assertEqual(grid['candidate_pairs_scored_once'],8)
            self.assertEqual(grid['request_ids'],[r[0] for r in prepared['validation_requests']])
            self.assertEqual(fp.evaluate_grid(prepared),grid)
            rows=pd.read_csv(root/'work/validation_per_request.csv')
            self.assertEqual(len(rows),2*5)

    def test_selected_export_scores_replay_persistence_and_no_test_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixtures.fixture(root)
            prepared=cp.prepare(settings(root,shared),root/'work')
            grid=fp.evaluate_grid(prepared);export=fp.export_selected(prepared,grid)
            self.assertEqual(export['checks']['validation']['requests_verified'],2)
            self.assertEqual(export['checks']['validation']['catalogue_pairs_rescored_for_verification'],8)
            self.assertEqual(export['checks']['validation']['full_score_rows_exported'],0)
            expected=grid['selected']['metrics'];actual=export['exported_validation']
            self.assertAlmostEqual(expected['known_click_recall@100'],actual['known_click_recall@100'])
            self.assertAlmostEqual(expected['observed_click_ndcg@100'],actual['observed_click_ndcg@100'])
            record=json.loads((root/'work/run_config.json').read_text())
            self.assertEqual(record['score_column'],'scores');self.assertIsNone(record['test_metrics'])
            self.assertFalse((root/'work/predictions').exists())
            frame=pd.read_csv(root/'work/rankings/validation/submission_top_200.csv')
            for row in frame.itertuples():
                ids=json.loads(row.article_ids);scores=json.loads(row.scores);ages=json.loads(row.article_age_hours)
                cosine=json.loads(row.similarity_scores)
                np.testing.assert_allclose(scores,fp.blend(cosine,ages,grid['selected']['variant']),atol=2e-6)
                self.assertEqual(len(ids),len(scores));self.assertEqual(scores,sorted(scores,reverse=True))
            self.assertEqual(fp.export_selected(prepared,grid),export)
            destination=fp.persist(prepared)
            self.assertTrue((destination/'run_config.json').is_file())
            self.assertFalse((destination/'indexes').exists())

    def test_test_labels_do_not_select_weight_and_invalid_budgets_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixtures.fixture(root);results=[]
            test_file=shared/'feature_engineering/test_features.parquet'
            for label in [0,1]:
                frame=pd.read_parquet(test_file);frame['clicked']=label;frame.to_parquet(test_file,index=False)
                prepared=cp.prepare(settings(root,shared),root/f'work_{label}')
                grid=fp.evaluate_grid(prepared)
                results.append((grid['selected']['variant'],[row['metrics'] for row in grid['results']]))
            self.assertEqual(results[0],results[1])
            small=cp.prepare(dict(settings(root,shared),max_export_pairs=1),root/'small')
            with self.assertRaises(RuntimeError):fp.evaluate_grid(small)
            with self.assertRaises(ValueError):fp.variants({'freshness_weights':[-.1]})
            with self.assertRaises(ValueError):fp.blend([.5],[-1],{'kind':'similarity_plus_age','weight':.1})
            tie=[{'variant':{'name':'similarity_only'},'metrics':{'known_click_recall@100':.2,'observed_click_ndcg@100':.1}},
                 {'variant':{'name':'freshness_only'},'metrics':{'known_click_recall@100':.2,'observed_click_ndcg@100':.1}}]
            self.assertIs(fp.select(tie),tie[0])

if __name__=='__main__':unittest.main()
