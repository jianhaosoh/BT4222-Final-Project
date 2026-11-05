"""Check candidate-to-individual-history aggregation and persisted replay."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import catalogue_pipeline as cp
from test_similarity import fixture,config

class IndividualHistoryChecks(unittest.TestCase):
    def test_exact_scores_missingness_repeats_and_recent_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixture(root)
            vectors,_=cp.load_embeddings([shared/'train_val/article embeddings/text_embeddings.parquet',shared/'test/article embeddings/text_embeddings.parquet'])
            source=shared/'train_val/train/history.parquet'
            ids=np.array([1,2,3,6])
            expectations={'max_history':[1,0,1,1/np.sqrt(2)],
                          'topk_history':[2/3,0,1/3,1/np.sqrt(2)]}
            for method,expected in expectations.items():
                profiles,_=cp.history_profiles(source,{7,8},'2023-05-18',vectors,
                    history_limit=10,aggregation=method,top_k=3)
                actual=cp.semantic_values(ids,7,profiles,vectors)
                np.testing.assert_allclose(actual[:,0],expected,atol=1e-6)
                self.assertEqual(actual[:,1].tolist(),[0,1,0,0])
                np.testing.assert_array_equal(cp.cosine_scores(ids,8,profiles,vectors),np.zeros(4))
                recent,_=cp.history_profiles(source,{7},'2023-05-18',vectors,
                    history_limit=1,aggregation=method,top_k=3)
                np.testing.assert_allclose(cp.cosine_scores(np.array([1,3]),7,recent,vectors),[0,1])
                # SQLite replay retains the history matrix's shape and aggregation.
                disk,_=cp.history_profiles(source,{7},'2023-05-18',vectors,
                    output_db=root/f'{method}.sqlite',history_limit=10,aggregation=method,top_k=3)
                try:np.testing.assert_allclose(cp.cosine_scores(ids,7,disk,vectors),expected,atol=1e-6)
                finally:disk.close()

    def test_opposing_interests_do_not_cancel_individual_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'history.parquet'
            pd.DataFrame({'user_id':[7],'article_id_fixed':[[1,2]],
                          'impression_time_fixed':[['2020-01-01','2020-01-02']]}).to_parquet(source,index=False)
            vectors={1:np.array([1,0],dtype='float32'),2:np.array([-1,0],dtype='float32')}
            mean,_=cp.history_profiles(source,{7},'2021-01-01',vectors,history_limit=10)
            self.assertIsNone(mean[7][0])
            maximum,_=cp.history_profiles(source,{7},'2021-01-01',vectors,history_limit=10,aggregation='max_history')
            np.testing.assert_allclose(cp.cosine_scores(np.array([1,2]),7,maximum,vectors),[1,1])
            top3,_=cp.history_profiles(source,{7},'2021-01-01',vectors,history_limit=10,aggregation='topk_history')
            np.testing.assert_allclose(cp.cosine_scores(np.array([1,2]),7,top3,vectors),[0,0])

    def test_tensor_and_numpy_aggregation_match(self):
        try:import torch
        except ImportError:self.skipTest('PyTorch unavailable')
        vectors={1:np.array([1,0],dtype='float32'),2:np.array([0,1],dtype='float32')}
        table=cp.SimilarityVectors(vectors,device='cpu',min_gpu_batch=1)
        table._torch=torch;table._article_index={1:1,2:2}
        table._matrix=torch.tensor([[0.,0.],[1.,0.],[0.,1.]])
        history=np.array([[1.,0.],[0.,1.],[1.,0.]],dtype='float32')
        for method in ['max_history','topk_history']:
            for k in [1,2,3,9]:
                for matrix in [history,history[:1]]:
                    profiles=cp.HistoryProfiles({7:(matrix,1)},aggregation=method,top_k=k)
                    ids=np.array([2,99,1,1])
                    np.testing.assert_allclose(cp.semantic_values(ids,7,profiles,table),
                                               cp.semantic_values(ids,7,profiles,vectors),atol=2e-6)

    def test_export_scores_reload_and_method_change_guard(self):
        for method in ['max_history','topk_history']:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);shared=fixture(root)
                settings=dict(config(root,shared),history_limit=10,history_aggregation=method,history_top_k=3)
                prepared=cp.prepare(settings,root/'work');result=cp.execute(prepared)
                self.assertTrue(result['persistence']['reload_from_persistent_copy'])
                self.assertEqual(result['metrics']['evaluated_requests'],2)
                lookup,_=cp.profiles_for(prepared,'validation')
                try:
                    frame=pd.read_parquet(root/'work/rankings/validation/submission_top_200_scores.parquet')
                    for row in frame.itertuples():
                        uid=int(row.behaviour_id.split(':')[0]);ids=np.asarray(row.article_ids)
                        np.testing.assert_allclose(row.similarity_scores,
                                                   cp.cosine_scores(ids,uid,lookup,prepared['vectors']),atol=2e-6)
                finally:lookup.close()
                saved=json.loads((root/'work/run_config.json').read_text())
                self.assertEqual(saved['model']['history_aggregation'],method)
                self.assertTrue(cp.execute(prepared)['reused_completed_run'])
                with self.assertRaisesRegex(ValueError,'new run ID'):
                    cp.prepare(dict(settings,history_aggregation='mean_profile'),root/'work')

    def test_individual_history_configuration_is_bounded(self):
        for settings in [dict(history_aggregation='bad',history_limit=10),
                         dict(history_aggregation='max_history',history_limit=None),
                         dict(history_aggregation='topk_history',history_limit=10,history_top_k=0)]:
            with self.assertRaises(ValueError):cp.model_spec(settings)

if __name__=='__main__':unittest.main()
