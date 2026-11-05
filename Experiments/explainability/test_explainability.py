"""Tiny synthetic checks; no project data or fitted-model evaluation."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
EXPERIMENTS = Path(__file__).resolve().parents[1]
for folder in ['2.1_lstm','2.2_huim','2.2_huim/tests']:
    sys.path.insert(0,str(EXPERIMENTS/folder))


class CommonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('explain_common'))
        import explain_common
        self.c = explain_common

    def test_sample_is_order_independent_and_preserves_composite_ids(self):
        r = pd.DataFrame({'behaviour_id':['1:4','2:4','3:5'], 'user_id':[1,2,3],
                          'impression_id':[4,4,5], 'impression_time':pd.to_datetime(['2023-01-01']*3)})
        a = self.c.sample_requests(r,2)
        b = self.c.sample_requests(r.iloc[::-1],2)
        self.assertEqual(a.behaviour_id.tolist(),b.behaviour_id.tolist())

    def test_unknown_exposure_is_not_a_negative_label(self):
        self.assertEqual(self.c.exposure_status(10,[10,20],[10]),'clicked')
        self.assertEqual(self.c.exposure_status(20,[10,20],[10]),'shown_not_clicked')
        self.assertEqual(self.c.exposure_status(30,[10,20],[10]),'unexposed_unknown')

    def test_score_parts_reconstruct_float32_recipe(self):
        raw=np.array([2.,-1.],np.float32)
        age=np.array([0.,24.],np.float64)
        p=self.c.score_parts(raw,age,2.)
        np.testing.assert_allclose(p['final_score'],raw-2*np.log1p(age).astype(np.float32),atol=0,rtol=0)
        self.assertEqual(float(p['freshness_penalty'][0]),0.)

    def test_sensitivity_summary_balances_requests(self):
        rows=pd.DataFrame({'behaviour_id':['a','a','b'],'group':['text']*3,
                           'neural_delta':[2.,2.,0.]})
        out=self.c.sensitivity_summary(rows)
        self.assertEqual(out.iloc[0]['mean_absolute_change'],1.)


def lstm_fixture():
    import lstm_data as D, lstm_model as M
    from lstm_explain import LSTMExplainer
    n,L,t = 7,4,1_700_000_000
    rng=np.random.default_rng(2)
    names=['log_body_words_z','body_words_missing','sentiment_z','sentiment_missing',
           'premium','premium_missing','in_catalogue','has_text_vector','has_image_vector']
    numeric=np.zeros((n,9),np.float32);numeric[1:,6:]=1
    art=D.ArticleTable(article_ids=np.array([-1,11,22,33,44,55,66]),category=np.array([0]+[3]*6),
        article_type=np.array([0]+[3]*6),topics=np.array([[0,0]]+[[2,0]]*6),numeric=numeric,numeric_names=names,
        published_s=np.array([0]+[t-86400]*6),published_missing=np.array([True]+[False]*6),
        published_us=np.array([0]+[(t-86400)*1_000_000]*6),text_row=np.arange(n),
        text=rng.normal(size=(n,4)).astype(np.float32),has_text=np.array([False]+[True]*6),
        image_row=np.arange(n),image=rng.normal(size=(n,3)).astype(np.float32),
        has_image=np.array([False]+[True]*6),id_bucket=np.zeros(n,np.int64))
    pre={'numeric_stats':{'log_candidate_age_h':{'mean':0.,'std':1.},'log_gap_prev_h':{'mean':0.,'std':1.}}}
    times=np.array([[t-10800,t-7200,t-3600,0],[t-14400,t-10800,t-7200,0],[0]*4])
    features=np.zeros((3,L,8),np.float32);features[:2,0,5]=1;features[:2,1:3,4]=np.log(2.)
    arrays=D.SplitArrays(split='validation',user_id=np.array([1,2]),impression_id=np.array([7,7]),
        imp_time_s=np.array([t,t]),imp_user_row=np.array([0,1]),
        imp_context=np.array([[1.,0.,1.],[0.,0.,1.]],np.float32),offsets=np.array([0,2,4]),
        cand_article_id=np.array([11,22,11,22]),cand_article_row=np.array([1,2,1,2]),
        cand_features=np.tile(np.array([np.log(25.),0.],np.float32),(4,1)),labels=np.array([1.,0.,1.,0.]),
        hist_article_row=np.array([[1,2,3,0],[2,3,4,0],[0]*4]),hist_times_s=times,hist_features=features,
        hist_lengths=np.array([3,3,0]),hist_full_lengths=np.array([3,3,0]),
        context_names=['is_subscriber','is_subscriber_missing','log_history_length_z'],
        candidate_names=['log_age_h_z','age_missing'],event_names=[str(i) for i in range(8)],
        imp_time_us=np.array([t,t])*1_000_000)
    cfg=M.ModelConfig(n_categories=5,n_types=5,n_topics=5,n_article_numeric=9,text_dim_in=4,image_dim_in=3,
        article_dim=8,lstm_hidden=8,attention_dim=4,interaction_dim=4,n_context_features=3,head_hidden=(8,))
    torch.manual_seed(2)
    model=M.build_model(cfg,M.article_table_tensors(art,torch.device('cpu')),torch.device('cpu')).eval()
    req=pd.DataFrame({'behaviour_id':['1:7','2:7'],'user_id':[1,2],'impression_id':[7,7]})
    return LSTMExplainer(model,art,arrays,M.SplitTensors(arrays,torch.device('cpu')),pre,req,D,M)


class AdapterTests(unittest.TestCase):
    def test_lstm_original_forward_and_table_restore(self):
        e=lstm_fixture();original=e.model.articles._table
        with torch.inference_mode():expected=e.model(e.tensors.batch(torch.tensor([0]))).numpy()
        scores,weights=e.score('1:7',[11,22])
        np.testing.assert_allclose(scores,expected,atol=1e-6)
        self.assertTrue(np.allclose(weights.sum(1),1))
        for group in ['text','image','metadata','history','context']:
            modified,w=e.score('1:7',[11,22],group=group)
            self.assertTrue(np.isfinite(modified).all())
            self.assertIs(e.model.articles._table,original)
        again,_=e.score('1:7',[11,22]);np.testing.assert_array_equal(again,scores)

    def test_lstm_removal_repacks_and_recomputes_gaps(self):
        e=lstm_fixture();batch=e.tensors.batch(torch.tensor([0]))
        e._remove_event(batch,1)
        self.assertEqual(batch['hist_rows'][0].tolist(),[1,3,0])
        self.assertEqual(int(batch['hist_len'][0]),2)
        self.assertAlmostEqual(float(batch['hist_feats'][0,1,4]),np.log(3.),places=6)
        self.assertEqual(float(batch['hist_feats'][0,0,5]),1.)

    def test_lstm_request_alignment_uses_composite_ids(self):
        import lstm_explain
        req=pd.DataFrame({'behaviour_id':['2:7','1:7'],'value':[20,10]})
        aligned=lstm_explain.ordered_requests(req,np.array([1,2]),np.array([7,7]))
        self.assertEqual(aligned.value.tolist(),[10,20])

    def test_lstm_missing_publication_matches_saved_explicit_penalty(self):
        e=lstm_fixture();e.articles.published_missing[1]=True;e.articles.published_s[1]=0
        self.assertTrue(np.isnan(e._age('1:7',[11])[0]))
        self.assertEqual(e.age_hours('1:7',[11])[0],e.arrays.imp_time_s[0]/3600.)

    def test_huim_forward_and_nonmutating_interventions(self):
        self.assertIsNotNone(importlib.util.find_spec('huim_explain'))
        from huim_explain import HUIMExplainer
        from test_static import synthetic_world,make_model
        from huim import features as F
        _,_,_,_,pre,art,ctx=synthetic_world(n_articles=20,n_requests=4)
        model=make_model(pre,art)
        e=HUIMExplainer(model,art,ctx,pre,F)
        bid=ctx.requests.behaviour_id.iloc[0];ids=ctx.recorded['article_id'][:2]
        row=art.lookup(ids)
        with torch.inference_mode():expected=model(ctx.tensors,art.tensors,len(art.ids),torch.tensor([0,0]),torch.as_tensor(row)).numpy()
        raw,w=e.score(bid,ids);np.testing.assert_allclose(raw,expected,atol=1e-6)
        original={k:v.clone() for k,v in ctx.tensors.items()}
        for group in ['text','image','metadata','history','context']:
            s,_=e.score(bid,ids,group=group);self.assertTrue(np.isfinite(s).all())
        for k,v in original.items():torch.testing.assert_close(ctx.tensors[k],v,rtol=0,atol=0)
        np.testing.assert_array_equal(e.score(bid,ids)[0],raw)

    def test_tiny_explanation_and_report_end_to_end(self):
        import tempfile
        import explain_common as C
        import explain_report as R
        e=lstm_fixture();e.weight=2.
        req=e.requests.copy();req['impression_time']=pd.to_datetime(e.arrays.imp_time_s,unit='s')
        req['article_ids_inview']=[[11,66],[22,33]];req['article_ids_clicked']=[[66],[22]]
        e.requests=req
        ids=e.articles.article_ids[1:]
        meta=pd.DataFrame({'article_id':ids,'title':[f'Article {i}' for i in ids],
                           'published_time':pd.to_datetime(e.articles.published_s[1:],unit='s')}).set_index('article_id')
        sources={}
        for name in ['lstm','huim']:
            ranks=[];scores=[]
            for r in req.itertuples():
                raw,_=e.score(r.behaviour_id,ids)
                final=C.score_parts(raw,e.age_hours(r.behaviour_id,ids),2.)['final_score']
                if name=='huim':final=-final
                inside=ids[ids!=66];order=np.argsort(-final[ids!=66],kind='stable')
                ranks.append({'behaviour_id':r.behaviour_id,'article_ids':inside[order]})
                scores.extend({'behaviour_id':r.behaviour_id,'article_id':int(a),'score':float(s)} for a,s in zip(ids,final))
            sources[name]={'rankings':pd.DataFrame(ranks).set_index('behaviour_id'),
                'scores':pd.DataFrame(scores).set_index(['behaviour_id','article_id']).score,
                'recipe':{'freshness_weight':2.}}
        pairs=C.build_pairs(req,sources,meta)
        e.set_reference(req.behaviour_id.iloc[0])
        with tempfile.TemporaryDirectory() as tmp:
            base,pert,summary,attn,weights=C.explain_pairs(e,pairs,sources,'lstm',tmp)
            cases=C.choose_cases(base,req,sources,'lstm')
            removal=C.event_removal(e,cases,weights,meta)
            coverage=R.coverage_table(e,meta,req,sources['lstm']['rankings'])
            R.make_report(e,base,pert,summary,attn,weights,cases,removal,coverage,meta,'lstm',tmp)
            self.assertTrue((Path(tmp)/'findings.md').is_file())
            self.assertTrue((Path(tmp)/'figures'/'group_sensitivity.png').is_file())
            self.assertGreaterEqual(len(cases),2)
            self.assertEqual(len(summary),5)


if __name__=='__main__':
    unittest.main()
