"""Synthetic dataset checks; no Drive data or model performance claims."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import catalogue_pipeline as cp
import training_examples as te


def make_shared(root):
    shared=root/'shared'
    for folder in ['feature_engineering','train_val/train','train_val/article embeddings','test/article embeddings']:
        (shared/folder).mkdir(parents=True,exist_ok=True)
    ids=list(range(1,9))
    pd.DataFrame({'article_id':ids,'published_time':pd.to_datetime(['2020-01-01']*7+['2030-01-01'])}).to_parquet(shared/'train_val/articles.parquet',index=False)
    pd.DataFrame({'article_id':[1],'published_time':pd.to_datetime(['2020-01-01'])}).to_parquet(shared/'test/articles.parquet',index=False)
    values=[[1.,0.],[0.,1.],[1.,1.],[-1.,0.],[0.,-1.],None,[-1.,1.],[1.,0.]]
    pd.DataFrame({'article_id':ids,'text_embedding':values,'has_text_embedding':[v is not None for v in values]}).to_parquet(shared/'train_val/article embeddings/text_embeddings.parquet',index=False)
    # First-snapshot precedence must also hold in the training-vector bundle.
    pd.DataFrame({'article_id':[1],'text_embedding':[[0.,1.]],'has_text_embedding':[True]}).to_parquet(shared/'test/article embeddings/text_embeddings.parquet',index=False)
    rows=[]
    for user,imp,slate,positives in [(7,10,ids,{1,2}),(7,11,[3,4],{3}),(8,10,[1,3],{1}),
                                    (7,12,[3,4],set()),(7,13,[8,4],{8})]:
        for aid in slate:
            rows.append({'user_id':user,'impression_id':imp,'candidate_article_id':aid,
                         'impression_time':pd.Timestamp('2023-01-01')+pd.Timedelta(hours=imp-10),
                         'split':'train','clicked':int(aid in positives)})
    pd.DataFrame(rows).to_parquet(shared/'feature_engineering/train_features.parquet',index=False)
    pd.DataFrame({'user_id':[7,8],'article_id_fixed':[[1,2],[]],
                  'impression_time_fixed':[[pd.Timestamp('2022-01-01'),pd.Timestamp('2022-01-02')],[]]}).to_parquet(shared/'train_val/train/history.parquet',index=False)
    return shared


def settings(shared):
    return {'drive_root':str(shared),'seed':42,'train_requests':50_000,
            'history_limit':10,'negatives_per_positive':4,'hash_inputs':True,
            'embedding_dimension':2,'max_index_gb':1,'paths':{}}


class TrainingExamplesChecks(unittest.TestCase):
    def test_examples_membership_profiles_vectors_and_missingness(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=make_shared(root);work=root/'work'
            summary=te.build(settings(shared),work,cp)
            examples=pd.read_parquet(work/'training_examples.parquet')
            self.assertEqual(len(examples),3)
            self.assertEqual(summary['counts']['selected_requests'],5)
            self.assertEqual(summary['counts']['requests_without_usable_profile'],1)
            self.assertEqual(summary['counts']['requests_without_positive_clicks'],1)
            self.assertEqual(summary['counts']['positive_clicks_excluded_by_eligibility'],1)
            self.assertEqual(summary['counts']['negative_candidates_excluded_missing_vector'],1)
            first=examples[examples.behaviour_id=='7:10']
            self.assertEqual(set(first.positive_article_id),{1,2})
            for row in first.itertuples():
                self.assertEqual(set(row.negative_article_ids),{3,4,5,7})
                self.assertTrue({1,2}.isdisjoint(row.negative_article_ids))
            second=examples[examples.behaviour_id=='7:11'].iloc[0]
            self.assertEqual(second.negative_article_ids.tolist(),[4])
            self.assertEqual(second.negative_count,1)
            profile=pd.read_parquet(work/'user_profiles.parquet').iloc[0]
            self.assertEqual(profile.user_id,7)
            np.testing.assert_allclose(profile.profile_embedding,[1/np.sqrt(2)]*2,atol=1e-6)
            articles=pd.read_parquet(work/'article_vectors.parquet').set_index('article_id')
            np.testing.assert_array_equal(articles.loc[1,'text_embedding'],[1,0])
            self.assertNotIn(6,articles.index);self.assertNotIn(8,articles.index)
            recipe=json.loads((work/'training_config.json').read_text())
            self.assertFalse(recipe['model_training_performed'])
            self.assertFalse(any('features_test' in key or 'features_validation' in key for key in recipe['signature']['sources']))
            before=(work/'training_examples.parquet').read_bytes()
            self.assertEqual(te.build(settings(shared),work,cp),summary)
            self.assertEqual(before,(work/'training_examples.parquet').read_bytes())

    def test_sampling_reproducibility_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=make_shared(root);config=dict(settings(shared),negatives_per_positive=2)
            for folder in ['a','b']:te.build(config,root/folder,cp)
            a=pd.read_parquet(root/'a/training_examples.parquet');b=pd.read_parquet(root/'b/training_examples.parquet')
            self.assertEqual(a.to_json(),b.to_json())
            self.assertTrue(all(len(v)<=2 for v in a.negative_article_ids))
            with self.assertRaisesRegex(ValueError,'new run ID'):
                te.build(dict(config,seed=43),root/'a',cp)
            with self.assertRaisesRegex(RuntimeError,'budget'):
                te.build(dict(config,max_index_gb=0),root/'small',cp)
            with self.assertRaises(ValueError):te.build(dict(config,train_requests=0),root/'invalid',cp)

    def test_sample_is_complete_requests_and_source_changes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=make_shared(root);config=dict(settings(shared),train_requests=4)
            summary=te.build(config,root/'work',cp)
            self.assertEqual(summary['counts']['selected_requests'],4)
            selected=set(summary['selected_request_ids'])
            examples=pd.read_parquet(root/'work/training_examples.parquet')
            self.assertTrue(set(examples.behaviour_id)<=selected)
            if '7:10' in selected:
                self.assertEqual(set(examples[examples.behaviour_id=='7:10'].positive_article_id),{1,2})
            source=shared/'feature_engineering/train_features.parquet';frame=pd.read_parquet(source)
            frame.loc[0,'clicked']=0;frame.to_parquet(source,index=False)
            with self.assertRaisesRegex(ValueError,'changed'):
                te.build(config,root/'work',cp)

    def test_temporal_history_and_split_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=make_shared(root);config=settings(shared)
            source=shared/'train_val/train/history.parquet';frame=pd.read_parquet(source)
            frame['impression_time_fixed']=frame['impression_time_fixed'].apply(lambda values:[pd.Timestamp(v) for v in values])
            frame.at[0,'impression_time_fixed']=[pd.Timestamp('2024-01-01'),pd.Timestamp('2024-01-02')]
            frame.to_parquet(source,index=False)
            with self.assertRaisesRegex(ValueError,'History reaches'):
                te.build(config,root/'future',cp)
            source=shared/'feature_engineering/train_features.parquet';frame=pd.read_parquet(source)
            frame['split']='validation';frame.to_parquet(source,index=False)
            with self.assertRaisesRegex(ValueError,'Split column'):
                te.build(config,root/'wrong_split',cp)

    def test_persistent_copy_and_no_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=make_shared(root);te.build(settings(shared),root/'work',cp)
            destination=te.persist(root/'work',root/'saved')
            for name in te.ARTIFACTS:self.assertEqual((root/'work'/name).read_bytes(),(destination/name).read_bytes())
            self.assertFalse((destination/'_indexes').exists())
            with self.assertRaises(FileExistsError):te.persist(root/'work',destination)

if __name__=='__main__':unittest.main()
