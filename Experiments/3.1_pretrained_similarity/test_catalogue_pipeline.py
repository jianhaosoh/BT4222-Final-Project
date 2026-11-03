"""Synthetic protocol checks. These are not shared-dataset model results."""
import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd
from catalogue_pipeline import (REQUEST,ARTICLE,POLICY,identity,load_catalogue,load_embeddings,
    article_values,index_shared,request_iter,eligible_ids,retrieval_metrics,score_request,
    prepare,execute,validate_export,load_trained_run,predict,ProfileLookup,export_split,history_profiles,check_cost)


def fixture(root):
    shared=root/'shared'
    for directory in ['feature_engineering','train_val/train','train_val/validation','test',
                      'train_val/article embeddings','test/article embeddings']:
        (shared/directory).mkdir(parents=True,exist_ok=True)
    (shared/'feature_engineering/feature_vocabulary.json').write_text(json.dumps({'version':1}))
    catalogue=pd.DataFrame({
        'article_id':[1,2,3,4,5],
        'published_time':pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03','2030-01-01',None]),
        'premium':[False,True,False,False,False],
        'body':['one two','a b c',None,'future','unknown'],
        'topics':[['sport','sport'],['news'],[],['news'],[]],
    })
    catalogue.to_parquet(shared/'train_val/articles.parquet',index=False)
    testcat=catalogue.iloc[[1]].copy()
    extra=pd.DataFrame({'article_id':[6],'published_time':[pd.Timestamp('2020-01-04')],
                        'premium':[False],'body':['test archive'],'topics':[['history']]})
    pd.concat([testcat,extra],ignore_index=True).to_parquet(shared/'test/articles.parquet',index=False)
    pd.DataFrame({'article_id':[1,2,3,4,5],'text_embedding':[[1.,0.,0.],None,[0.,1.,0.],[0.,0.,1.],[1.,1.,0.]],
        'has_text_embedding':[True,False,True,True,True]}).to_parquet(shared/'train_val/article embeddings/text_embeddings.parquet',index=False)
    pd.DataFrame({'article_id':[2,6],'text_embedding':[[0.,0.,1.],[1.,0.,1.]],
        'has_text_embedding':[True,True]}).to_parquet(shared/'test/article embeddings/text_embeddings.parquet',index=False)
    for split,day in [('train','2023-05-18'),('validation','2023-05-25'),('test','2023-06-01')]:
        rows=[]
        # Same impression ID for two users must remain two requests.
        for uid,imp,when in [(7,10,pd.Timestamp(day)),(8,10,pd.Timestamp(day)+pd.Timedelta('1h'))]:
            for aid in [1,2,3,4]:
                article=catalogue[catalogue.article_id.eq(aid)].iloc[0]
                age=(when-article.published_time).total_seconds()/3600
                age=age if age>=0 else np.nan
                rows.append({'split':split,'user_id':uid,'impression_id':imp,
                    'candidate_article_id':aid,'impression_time':when,
                    'clicked':None if split=='test' else int(aid==(1 if uid==7 else 4)),
                    'hour':when.hour,'day_of_week':when.dayofweek,'is_weekend':when.dayofweek>=5,
                    'is_subscriber':False,'is_sso_user':True,'is_front_page':True,
                    'history_size':2 if uid==7 else 0,'premium':article.premium,
                    'body_word_count':len(article.body.split()) if isinstance(article.body,str) else None,
                    'topic_count':len(set(article.topics)),'has_topics':bool(article.topics),
                    'article_age_h':age,'article_age_missing':pd.isna(age)})
        pd.DataFrame(rows).to_parquet(shared/f'feature_engineering/{split}_features.parquet',index=False)
        folder=shared/f'train_val/{split}' if split!='test' else shared/'test'
        pd.DataFrame({'user_id':[7,8],'article_id_fixed':[[1,3],[]],
            'impression_time_fixed':[[pd.Timestamp('2022-01-01'),pd.Timestamp('2022-01-02')],[]]}).to_parquet(folder/'history.parquet',index=False)
    return shared


def configuration(root,shared):
    return {'drive_root':str(shared),'experiment_id':'3.1_pretrained_similarity','run_id':'synthetic',
        'seed':42,'feature_case':'raw','scope':'final','input_version':'synthetic-v1','owner':'synthetic fixture',
        'train_requests':None,'selection_requests':1,'embedding_source':{'model':'synthetic test vectors','revision':'fixture-v1'},
        'export_requests':{'train':None,'validation':None,'test':None},
        'alphas':[0.001,0.01],'epochs':2,'patience':2,'batch_size':3,'catalogue_batch_size':2,
        'embedding_dimension':3,'hash_inputs':True,'max_export_pairs':10000,
        'max_export_gb':1,'max_index_gb':1,'persistent_root':str(root/'persistent')}


class CatalogueChecks(unittest.TestCase):
    def test_composite_ids_and_original_denominator(self):
        self.assertNotEqual(identity(7,10),identity(8,10))
        with self.assertRaises(ValueError):identity(7.0,10)
        recall,ndcg=retrieval_metrics([1,2],{1,999})
        self.assertEqual(recall,0.5)
        self.assertAlmostEqual(ndcg,1/(1+1/np.log2(3)))
        self.assertEqual(retrieval_metrics([],{999}),(0,0))

    def test_missing_vectors_first_snapshot_wins_and_future_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            shared=fixture(Path(directory))
            cat,_=load_catalogue([shared/'train_val/articles.parquet',shared/'test/articles.parquet'])
            vectors,info=load_embeddings([shared/'train_val/article embeddings/text_embeddings.parquet',shared/'test/article embeddings/text_embeddings.parquet'],expected_dimension=3)
            self.assertNotIn(2,vectors)
            self.assertEqual(eligible_ids(cat,'2023-01-01').tolist(),[1,2,3,6])
            self.assertEqual(eligible_ids(cat,'1990-01-01').tolist(),[])
            self.assertEqual(info['sources'][1]['overlap_first_wins'],1)

    def test_duplicate_candidate_keys_and_test_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); shared=fixture(root)
            p=shared/'feature_engineering/train_features.parquet'
            f=pd.read_parquet(p);pd.concat([f,f.iloc[:1]],ignore_index=True).to_parquet(p,index=False)
            with self.assertRaisesRegex(ValueError,'Duplicate source'):
                index_shared(p,'train',root/'duplicates.sqlite',42)
            p=shared/'feature_engineering/test_features.parquet'
            f=pd.read_parquet(p);f['clicked']=0;f.to_parquet(p,index=False)
            with self.assertRaisesRegex(ValueError,'Test targets'):
                index_shared(p,'test',root/'test.sqlite',42)

    def test_malformed_vectors_fail_instead_of_becoming_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'bad.parquet'
            pd.DataFrame({'article_id':[1],'text_embedding':[[np.nan,1.]],'has_text_embedding':[True]}).to_parquet(p,index=False)
            with self.assertRaisesRegex(ValueError,'Invalid vector'):load_embeddings([p])

    def test_shared_reconstruction_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixture(root)
            cat,_=load_catalogue([shared/'train_val/articles.parquet',shared/'test/articles.parquet'])
            p=shared/'feature_engineering/train_features.parquet';f=pd.read_parquet(p)
            f.loc[0,'body_word_count']=10000;f.to_parquet(p,index=False)
            with self.assertRaisesRegex(ValueError,'reconstruction'):
                index_shared(p,'train',root/'audit.sqlite',42,catalogue=cat)

    def test_exact_top100_ties_and_excluded_known_click_denominator(self):
        from unittest.mock import patch
        ids=np.arange(1,151)
        cat=pd.DataFrame({'article_id':ids,'published_time':pd.to_datetime(['2020-01-01']*150),
            'premium':False,'body_word_count':2,'topic_count':1,'has_topics':True}).set_index('article_id',drop=False)
        req=('7:10',7,10,'2023-01-01',json.dumps([0,0,False,False,True,True,0]))
        stats={}
        with patch('catalogue_pipeline.predict',side_effect=lambda bundle,raw:(np.zeros(len(raw)),np.full(len(raw),0.5))):
            ranking,count=score_request({},req,cat,{}, {},17,stats=stats)
        self.assertEqual(ranking,list(range(1,101)))
        self.assertTrue(stats['ties_crossing_100'])
        self.assertEqual(retrieval_metrics(ranking,{1,150})[0],0.5)

    def test_empty_pool_is_exported_and_keeps_its_request(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixture(root)
            cat,_=load_catalogue([shared/'train_val/articles.parquet',shared/'test/articles.parquet'])
            db=root/'empty.sqlite'
            with sqlite3.connect(db) as con:
                con.execute('CREATE TABLE candidates (behaviour_id TEXT,article_id INTEGER,clicked INTEGER)')
                con.execute('INSERT INTO candidates VALUES (?,?,?)',('7:10',1,1))
            con.close()
            req=('7:10',7,10,'1990-01-01',json.dumps([0,0,False,False,True,True,0]))
            with patch('catalogue_pipeline.predict',side_effect=lambda bundle,raw:(np.zeros(len(raw)),np.full(len(raw),0.5))):
                export_split({},[req],db,'validation',root,cat,{}, {},2)
            frame=pd.read_csv(root/'rankings/validation/submission_top_100.csv')
            self.assertEqual(frame.article_ids.tolist(),['[]'])
            checks=validate_export(root,'validation',lambda:iter([req]),cat,db,0)
            self.assertEqual(checks['requests_verified'],1)

    def test_future_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'history.parquet'
            pd.DataFrame({'user_id':[7],'article_id_fixed':[[1]],
                'impression_time_fixed':[[pd.Timestamp('2023-05-18')]]}).to_parquet(path,index=False)
            with self.assertRaisesRegex(ValueError,'History reaches'):
                history_profiles(path,{7},'2023-05-18',{1:np.array([1.,0.],dtype='float32')})

    def test_budget_refusal_and_frozen_sampling_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixture(root);config=configuration(root,shared)
            config['max_export_pairs']=1
            prepared=prepare(config,root/'work')
            self.assertEqual(prepared['budgets']['validation']['eligible_pairs'],8)
            with self.assertRaisesRegex(RuntimeError,'beyond max_export_pairs'):
                check_cost(prepared)
            self.assertFalse((root/'work/predictions').exists())
            changed=dict(config,seed=43)
            with self.assertRaisesRegex(ValueError,'sampling seed changed'):
                prepare(changed,root/'work')

    def test_end_to_end_exports_persistence_reload_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=fixture(root);config=configuration(root,shared)
            prepared=prepare(config,root/'work')
            self.assertEqual(prepared['audit']['request_counts']['test']['requests'],2)
            result=execute(prepared)
            self.assertTrue(result['handoff_complete'])
            self.assertTrue(result['persistence']['reload_from_persistent_copy'])
            self.assertEqual(result['metrics']['known_click_recall@100'],0.5)
            self.assertEqual(result['metrics']['evaluated_requests'],2)
            self.assertEqual(result['selection_validation']['evaluated_requests'],1)
            self.assertEqual(result['metrics']['known_click_exclusion_reasons']['future_publication'],1)
            for split in ['train','validation','test']:
                self.assertEqual(result['checks'][split]['catalogue_pairs_verified'],8)
                self.assertEqual(result['checks'][split]['recorded_pairs_verified'],8)
            source=Path(result['persistence']['persistent_location'])
            # A fresh load-only run restores the selected persistent model without fitting.
            second=dict(config,run_id='load_only')
            prepared2=prepare(second,root/'loaded_work')
            load_trained_run(prepared2,source)
            from unittest.mock import patch
            with patch('catalogue_pipeline.train',side_effect=AssertionError('No fitting in load-only mode')):
                again=execute(prepared2,train_model=False)
            self.assertEqual(again['metrics'],result['metrics'])
            # Deliberately corrupt one shortlist; exported full scores must catch it.
            p=root/'work/rankings/test/submission_top_100.csv'
            frame=pd.read_csv(p);frame.loc[0,'article_ids']='[4]';frame.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Top-100'):
                validate_export(root/'work','test',lambda:request_iter(prepared['dbs']['test']),prepared['catalogue'],prepared['dbs']['test'],8)


if __name__=='__main__':unittest.main()
