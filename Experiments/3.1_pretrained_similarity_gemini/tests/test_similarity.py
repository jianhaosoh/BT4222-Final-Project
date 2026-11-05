"""Synthetic checks for similarity-only retrieval; no project performance claims."""
from pathlib import Path
import sys
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
# Permit discovery from the repository root or this experiment folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import catalogue_pipeline as cp


def fixture(root):
    shared=root/'shared'
    for folder in ['feature_engineering','train_val/train','train_val/validation','test',
                   'train_val/article embeddings','test/article embeddings']:
        (shared/folder).mkdir(parents=True,exist_ok=True)
    # No engineered predictor columns or vocabulary are needed for this method.
    pd.DataFrame({'article_id':[1,2,3,4,5],
        'published_time':pd.to_datetime(['2020-01-01']*3+['2030-01-01',None])}).to_parquet(shared/'train_val/articles.parquet',index=False)
    pd.DataFrame({'article_id':[2,6],'published_time':pd.to_datetime(['2020-01-01']*2)}).to_parquet(shared/'test/articles.parquet',index=False)
    pd.DataFrame({'article_id':[1,2,3,4,5],'text_embedding':[[1.,0.],None,[0.,1.],[0.,1.],[-1.,0.]],
        'has_text_embedding':[True,False,True,True,True]}).to_parquet(shared/'train_val/article embeddings/text_embeddings.parquet',index=False)
    pd.DataFrame({'article_id':[2,6],'text_embedding':[[0.,1.],[1.,1.]],'has_text_embedding':[True,True]}).to_parquet(shared/'test/article embeddings/text_embeddings.parquet',index=False)
    for split,date in [('train','2023-05-18'),('validation','2023-05-25'),('test','2023-06-01')]:
        rows=[]
        for user in [7,8]:
            for aid in [1,2,3,4]:
                rows.append({'split':split,'user_id':user,'impression_id':10,'candidate_article_id':aid,
                    'impression_time':pd.Timestamp(date)+pd.Timedelta(hours=user-7),
                    'clicked':None if split=='test' else int(aid==(1 if user==7 else 4))})
        pd.DataFrame(rows).to_parquet(shared/f'feature_engineering/{split}_features.parquet',index=False)
        folder=shared/f'train_val/{split}' if split!='test' else shared/'test'
        pd.DataFrame({'user_id':[7,8],'article_id_fixed':[[1,1,3],[]],
            'impression_time_fixed':[[pd.Timestamp('2022-01-01'),pd.Timestamp('2022-01-02'),pd.Timestamp('2022-01-03')],[]]}).to_parquet(folder/'history.parquet',index=False)
    return shared


def config(root,shared):
    return {'drive_root':str(shared),'experiment_id':'3.1_pretrained_similarity','run_id':'synthetic_similarity',
        'seed':42,'scope':'final','owner':'synthetic fixture','input_version':'fixture-v1',
        'embedding_source':{'model':'synthetic vectors','revision':'v1'},'embedding_dimension':2,
        'similarity_device':'cpu','history_limit':None,'validation_requests':1,
        'export_requests':{'train':None,'validation':None,'test':None},'hash_inputs':True,
        'catalogue_batch_size':2,'max_export_pairs':10000,'max_export_gb':1,'max_index_gb':1,
        'persistent_root':str(root/'persistent')}


class RetrievalChecks(unittest.TestCase):
    def test_direct_cosine_mean_and_recent_history_without_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixture(root)
            vectors,_=cp.load_embeddings([shared/'train_val/article embeddings/text_embeddings.parquet',shared/'test/article embeddings/text_embeddings.parquet'])
            profiles,counts=cp.history_profiles(shared/'train_val/train/history.parquet',{7,8},'2023-05-18',vectors)
            scores=cp.cosine_scores(np.array([1,2,3]),7,profiles,vectors)
            np.testing.assert_allclose(scores,[2/np.sqrt(5),0,1/np.sqrt(5)],atol=1e-6)
            self.assertEqual(counts['selected_history_events'],3)
            recent,_=cp.history_profiles(shared/'train_val/train/history.parquet',{7},'2023-05-18',vectors,history_limit=1)
            np.testing.assert_allclose(cp.cosine_scores(np.array([1,3]),7,recent,vectors),[0,1])
            self.assertFalse(cp.model_spec(config(root,shared))['learned_parameters'])

    def test_eligibility_first_snapshot_missing_vectors_and_composite_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixture(root)
            cat,_=cp.load_catalogue([shared/'train_val/articles.parquet',shared/'test/articles.parquet'])
            vectors,info=cp.load_embeddings([shared/'train_val/article embeddings/text_embeddings.parquet',shared/'test/article embeddings/text_embeddings.parquet'])
            self.assertNotIn(2,vectors)
            self.assertEqual(cp.eligible_ids(cat,'2023-01-01').tolist(),[1,2,3,6])
            self.assertNotEqual(cp.identity(7,10),cp.identity(8,10))
            self.assertEqual(cp.retrieval_metrics([1,2],{1,4})[0],0.5)

    def test_duplicate_keys_rejected_and_test_labels_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixture(root)
            path=shared/'feature_engineering/train_features.parquet';frame=pd.read_parquet(path)
            pd.concat([frame,frame.iloc[:1]],ignore_index=True).to_parquet(path,index=False)
            with self.assertRaisesRegex(ValueError,'Duplicate source'):
                cp.index_shared(path,'train',root/'duplicates.sqlite',42)
            path=shared/'feature_engineering/test_features.parquet';frame=pd.read_parquet(path)
            indexed=[]
            for value in [0,1,None]:
                frame['clicked']=value
                frame.to_parquet(path,index=False)
                db_path=root/f'test_{value}.sqlite'
                counts=cp.index_shared(path,'test',db_path,42)
                self.assertFalse(counts['outcome_column_loaded'])
                self.assertEqual(counts['outcome_policy'],'ignored_for_test_inference')
                with sqlite3.connect(db_path) as db:
                    indexed.append(db.execute('SELECT * FROM candidates ORDER BY behaviour_id,article_id').fetchall())
                    self.assertEqual(db.execute('SELECT count(*) FROM candidates WHERE clicked IS NOT NULL').fetchone()[0],0)
                db.close()
            self.assertEqual(indexed[0],indexed[1])
            self.assertEqual(indexed[1],indexed[2])
            frame.drop(columns='clicked').to_parquet(path,index=False)
            counts=cp.index_shared(path,'test',root/'test_without_labels.sqlite',42)
            self.assertFalse(counts['source_has_clicked_column'])

    def test_future_history_and_invalid_vectors_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'history.parquet'
            pd.DataFrame({'user_id':[7],'article_id_fixed':[[1]],'impression_time_fixed':[[pd.Timestamp('2023-05-18')]]}).to_parquet(path,index=False)
            with self.assertRaisesRegex(ValueError,'History reaches'):
                cp.history_profiles(path,{7},'2023-05-18',{1:np.array([1.,0.],dtype='float32')})
            bad=root/'bad.parquet'
            pd.DataFrame({'article_id':[1],'text_embedding':[[np.nan,1.]],'has_text_embedding':[True]}).to_parquet(bad,index=False)
            with self.assertRaisesRegex(ValueError,'Invalid vector'):cp.load_embeddings([bad])

    def test_ties_and_missing_history_keep_candidates_and_top200(self):
        ids=np.arange(1,251)
        cat=pd.DataFrame({'article_id':ids,'published_time':pd.to_datetime(['2020-01-01']*250)}).set_index('article_id',drop=False)
        request=('7:10',7,10,'2023-01-01','[]');stats={}
        ranking,count=cp.score_request({},request,cat,{}, {},17,stats=stats)
        self.assertEqual(ranking,list(range(1,201)))
        self.assertEqual(count,250);self.assertTrue(stats['ties_crossing_200'])

    def test_empty_pool_exports_its_request_and_no_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixture(root)
            cat,_=cp.load_catalogue([shared/'train_val/articles.parquet',shared/'test/articles.parquet'])
            db=root/'empty.sqlite'
            with sqlite3.connect(db) as connection:
                connection.execute('CREATE TABLE candidates (behaviour_id TEXT,article_id INTEGER,clicked INTEGER)')
                connection.execute('INSERT INTO candidates VALUES (?,?,?)',('7:10',1,1))
            connection.close()
            req=('7:10',7,10,'1990-01-01','[]')
            cp.export_split({},[req],db,'validation',root,cat,{}, {},2)
            frame=pd.read_parquet(root/'rankings/validation/submission_top_200.parquet')
            self.assertEqual(frame.article_ids.iloc[0].tolist(),[])
            scores=pd.read_parquet(root/'rankings/validation/submission_top_200_scores.parquet')
            self.assertEqual(scores.similarity_scores.iloc[0].tolist(),[])
            self.assertFalse((root/'rankings/validation/submission_top_200.csv').exists())
            self.assertFalse((root/'predictions').exists())
            cp.validate_export(root,'validation',lambda:iter([req]),cat,db,0,model={},profiles={},vectors={})

    def test_end_to_end_score_only_bundle_persistent_reload_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixture(root);settings=config(root,shared)
            prepared=cp.prepare(settings,root/'work')
            result=cp.execute(prepared)
            self.assertTrue(result['handoff_complete'])
            self.assertTrue(result['persistence']['embedding_lookup_reopened'])
            self.assertEqual(result['metrics']['known_click_recall@200'],0.5)
            self.assertEqual(result['metrics']['evaluated_requests'],2)
            self.assertEqual(result['validation_sample']['evaluated_requests'],1)
            for split in cp.SPLITS:
                self.assertEqual(result['checks'][split]['catalogue_pairs_rescored_for_verification'],8)
                self.assertEqual(result['checks'][split]['shortlisted_articles_verified'],8)
                self.assertFalse((root/f'work/predictions/{split}').exists())
                self.assertFalse((root/'work/diagnostics').exists())
            manifest=json.loads((root/'work/run_config.json').read_text())
            self.assertEqual(manifest['export_profile'],'top200_only')
            self.assertFalse(manifest['shared_full_score_contract_complete'])
            frame=pd.read_parquet(root/'work/rankings/validation/submission_top_200_scores.parquet')
            for row in frame.itertuples():
                ids=row.article_ids.tolist();scores=row.similarity_scores.tolist()
                self.assertEqual(len(ids),len(scores))
                self.assertTrue(all(-1<=v<=1 for v in scores))
                self.assertEqual(scores,sorted(scores,reverse=True))
            row=frame.iloc[0]
            score_map=dict(zip(row.article_ids.tolist(),row.similarity_scores.tolist()))
            self.assertAlmostEqual(score_map[1],2/np.sqrt(5),places=6)
            self.assertAlmostEqual(score_map[3],1/np.sqrt(5),places=6)
            self.assertFalse(list((root/'work').rglob('*.joblib')))
            self.assertTrue(cp.execute(prepared)['reused_completed_run'])
            source=Path(result['persistence']['persistent_location'])
            second=cp.prepare(dict(settings,run_id='loaded'),root/'second')
            model=cp.load_run(second,source)
            replay=cp.execute(second,model)
            self.assertEqual(replay['metrics'],result['metrics'])
            path=root/'work/rankings/test/submission_top_200.parquet'
            import pyarrow as pa
            import pyarrow.parquet as pq
            native=pq.read_table(path).to_pylist();native[0]['article_ids']=[4]
            pq.write_table(pa.Table.from_pylist(native,schema=cp.shortlist_schemas()[0]),path)
            with self.assertRaisesRegex(ValueError,'Top-200'):
                cp.validate_export(root/'work','test',lambda:cp.request_iter(prepared['dbs']['test']),prepared['catalogue'],prepared['dbs']['test'],8,
                    model=cp.model_spec(settings),profiles={},vectors=prepared['vectors'])

    def test_all_validation_requests_can_be_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixture(root)
            settings=dict(config(root,shared),validation_requests=None)
            prepared=cp.prepare(settings,root/'work')
            self.assertEqual(len(prepared['validation_requests']),2)
            self.assertEqual(prepared['audit']['validation_requests'],
                             [r[0] for r in cp.request_iter(prepared['dbs']['validation'])])
            result=cp.evaluate(prepared)
            self.assertEqual(result['metrics']['evaluated_requests'],2)
            self.assertEqual(result['metrics']['known_click_recall@200'],0.5)
            self.assertEqual(result['eligible_pairs'],8)

    def test_budget_changes_and_old_hybrid_run_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixture(root);settings=config(root,shared)
            prepared=cp.prepare(dict(settings,max_export_pairs=1),root/'work')
            with self.assertRaisesRegex(RuntimeError,'beyond max_export_pairs'):cp.check_cost(prepared)
            with self.assertRaisesRegex(ValueError,'new run ID'):
                cp.prepare(dict(settings,seed=43),root/'work')
            older=root/'older';older.mkdir();(older/'input_audit.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'hybrid/older'):
                cp.prepare(settings,older)


class SimilarityDeviceChecks(unittest.TestCase):
    def test_cpu_features_preserve_missingness_and_duplicates(self):
        from catalogue_pipeline import SimilarityVectors,semantic_values
        vectors={10:np.array([1.,0.],dtype='float32'),20:np.array([0.,1.],dtype='float32')}
        profile=np.array([0.6,0.8],dtype='float32')
        ids=np.array([10,99,20,10])
        wrapped=SimilarityVectors(vectors,device='cpu',min_gpu_batch=1)
        expected=semantic_values(ids,7,{7:(profile,0.5)},vectors)
        actual=semantic_values(ids,7,{7:(profile,0.5)},wrapped)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(wrapped.info['similarity_device'],'cpu')
        self.assertEqual(actual[:,1].tolist(),[0,1,0,0])
        missing=semantic_values(ids,8,{},wrapped)
        self.assertTrue((missing[:,0]==0).all())
        self.assertTrue((missing[:,2]==1).all())

    def test_auto_without_torch_falls_back_and_explicit_cuda_fails(self):
        from catalogue_pipeline import SimilarityVectors
        from unittest.mock import patch
        vectors={10:np.array([1.,0.],dtype='float32')}
        with patch.dict('sys.modules',{'torch':None}):
            lookup=SimilarityVectors(vectors,device='auto')
            self.assertEqual(lookup.info['similarity_device'],'cpu')
            with self.assertRaisesRegex(RuntimeError,'requires PyTorch'):
                SimilarityVectors(vectors,device='cuda')

    def test_tensor_branch_matches_numpy_and_refreshes_single_profile_cache(self):
        try:import torch
        except ImportError:self.skipTest('Optional PyTorch dependency unavailable')
        from catalogue_pipeline import SimilarityVectors,semantic_values
        vectors={10:np.array([1.,0.],dtype='float32'),20:np.array([0.,1.],dtype='float32')}
        lookup=SimilarityVectors(vectors,device='cpu',min_gpu_batch=1)
        # Exercise the exact tensor calculation on CPU without claiming CUDA validation.
        lookup._torch=torch
        lookup._article_index={10:1,20:2}
        lookup._matrix=torch.tensor([[0.,0.],[1.,0.],[0.,1.]])
        ids=np.array([10,99,20,10])
        for profile in [np.array([0.6,0.8],dtype='float32'),np.array([-1.,0.],dtype='float32')]:
            profiles={7:(profile,0.75)}
            np.testing.assert_allclose(semantic_values(ids,7,profiles,lookup),
                semantic_values(ids,7,profiles,vectors),atol=2e-6,rtol=2e-6)
        self.assertEqual(semantic_values([],7,profiles,lookup).shape,(0,4))

    def test_actual_cuda_parity_when_hardware_is_available(self):
        try:import torch
        except ImportError:self.skipTest('Optional PyTorch dependency unavailable')
        if not torch.cuda.is_available():self.skipTest('No CUDA hardware in this environment')
        from catalogue_pipeline import SimilarityVectors,semantic_values
        rng=np.random.default_rng(42)
        matrix=rng.normal(size=(103,128)).astype('float32')
        matrix/=np.linalg.norm(matrix,axis=1,keepdims=True)
        vectors={int(i):row for i,row in enumerate(matrix)}
        profile=matrix[:10].mean(axis=0);profile/=np.linalg.norm(profile)
        ids=np.r_[np.arange(103),999,3]
        lookup=SimilarityVectors(vectors,device='cuda',min_gpu_batch=1)
        np.testing.assert_allclose(semantic_values(ids,7,{7:(profile,1.)},lookup),
            semantic_values(ids,7,{7:(profile,1.)},vectors),atol=2e-6,rtol=2e-6)
        self.assertEqual(lookup.info['similarity_device'],'cuda')

if __name__=='__main__':unittest.main()
