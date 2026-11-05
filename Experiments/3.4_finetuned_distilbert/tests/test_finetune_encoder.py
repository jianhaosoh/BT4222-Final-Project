"""Synthetic encoder-gradient checks; no pretrained downloads or quality claims."""
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
import finetune_encoder as ft

# Reuse the small article/request fixture, not any real project dataset.
fixture_path=Path(__file__).resolve().parents[2]/'3.2_pretrained_similarity_bert/tests/test_similarity.py'
spec=importlib.util.spec_from_file_location('_frozen_fixture',fixture_path)
fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)


def settings(shared):
    return {'drive_root':str(shared),'seed':42,'train_requests':2,'validation_requests':2,
            'history_limit':10,'negatives_per_positive':4,'hash_inputs':True,'max_index_gb':1,
            'batch_size':1,'gradient_accumulation':2,'temperature':.1,'amp':False,
            'chunk_batch_size':3,'encoding_article_batch':3,'catalogue_batch_size':2,
            'similarity_device':'cpu','paths':{},'additional_text_catalogues':[]}


def make_shared(root):
    shared=fixture_module.fixture(root)
    for part in ['train_val','test']:
        path=shared/part/'articles.parquet';frame=pd.read_parquet(path)
        frame['title']=[f'Article {aid} title' for aid in frame.article_id]
        frame['subtitle']=[f'Different topic {aid}' for aid in frame.article_id]
        frame.to_parquet(path,index=False)
    # Test behaviour is unnecessary and must not be accessed by this experiment.
    (shared/'feature_engineering/test_features.parquet').unlink()
    return shared


class PreparationChecks(unittest.TestCase):
    def test_train_only_examples_and_future_history_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=make_shared(root);data=ft.prepare(settings(shared),root/'work',cp)
            self.assertEqual(len(data['examples']),1)
            example=data['examples'][0]
            self.assertEqual(example['positive_article_id'],1)
            self.assertTrue(set(example['negative_article_ids'])<={2,3})
            self.assertNotIn(1,example['negative_article_ids'])
            self.assertFalse(data['audit']['test_behaviour_labels_read'])
            self.assertEqual(data['audit']['validation_requests_selected'],2)
            self.assertEqual(ft.text_for_article({'title':' A ','subtitle':' B '}),'title: A\nsubtitle: B')
            source=shared/'train_val/train/history.parquet';frame=pd.read_parquet(source)
            frame['impression_time_fixed']=frame.impression_time_fixed.apply(lambda values:[pd.Timestamp(v) for v in values])
            frame.at[0,'impression_time_fixed']=[pd.Timestamp('2024-01-01')]*3
            frame.to_parquet(source,index=False)
            with self.assertRaisesRegex(ValueError,'History reaches'):
                ft.prepare(settings(shared),root/'future',cp)

    def test_extra_history_text_does_not_expand_candidate_catalogue(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=make_shared(root)
            extra=root/'history_articles.parquet'
            pd.DataFrame({'article_id':[999],'title':['Extra'],'subtitle':['Historical text']}).to_parquet(extra,index=False)
            config=dict(settings(shared),additional_text_catalogues=[str(extra)])
            data=ft.prepare(config,root/'work',cp)
            self.assertIn(999,data['texts']);self.assertNotIn(999,data['catalogue'].index)

    def test_selection_uses_metrics_and_preserves_frozen_ties(self):
        frozen={'name':'frozen','metrics':{'known_click_recall@200':.03,'observed_click_ndcg@200':.01}}
        tied={'name':'epoch_1','metrics':dict(frozen['metrics'])}
        self.assertIs(ft.select_checkpoint([frozen,tied]),frozen)
        improved={'name':'epoch_2','metrics':{'known_click_recall@200':.04,'observed_click_ndcg@200':.001}}
        self.assertIs(ft.select_checkpoint([frozen,improved]),improved)


HAS_TRANSFORMERS=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None

@unittest.skipUnless(HAS_TRANSFORMERS,'PyTorch/Transformers unavailable')
class EncoderGradientChecks(unittest.TestCase):
    def setUp(self):
        import torch
        self.previous_threads=torch.get_num_threads();torch.set_num_threads(1);torch.manual_seed(42)
    def tearDown(self):
        import torch
        torch.set_num_threads(self.previous_threads)
    def tiny_encoder(self):
        import torch
        from transformers import DistilBertConfig,DistilBertModel
        class Tokenizer:
            model_input_names=['input_ids','attention_mask'];pad_token_id=0
            def __call__(self,text,**kwargs):
                ids=[ord(c)%47+1 for c in text]
                parts=[ids[i:i+4] for i in range(0,len(ids),4)]
                masks=[[1]*len(p)+[0]*(4-len(p)) for p in parts]
                values=[p+[0]*(4-len(p)) for p in parts]
                return {'input_ids':torch.tensor(values),'attention_mask':torch.tensor(masks),
                        'special_tokens_mask':torch.zeros((len(parts),4),dtype=torch.long)}
        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                config=DistilBertConfig(vocab_size=48,max_position_embeddings=16,n_layers=3,n_heads=2,dim=8,
                                       hidden_dim=16,dropout=0.,attention_dropout=0.,pad_token_id=0)
                self.auto_model=DistilBertModel(config)
            def forward(self,features):
                output=self.auto_model(input_ids=features['input_ids'],attention_mask=features['attention_mask'])
                return {'tokens':output.last_hidden_state,'mask':features['attention_mask']}
        class PoolDense(torch.nn.Module):
            def __init__(self):super().__init__();self.dense=torch.nn.Linear(8,4)
            def forward(self,features):
                mask=features['mask'][:,:,None]
                pooled=(features['tokens']*mask).sum(1)/mask.sum(1).clamp_min(1)
                return {'sentence_embedding':torch.tanh(self.dense(pooled))}
        class TinyEncoder(torch.nn.Sequential):
            tokenizer=Tokenizer();max_seq_length=4
            @property
            def device(self):return next(self.parameters()).device
            def get_sentence_embedding_dimension(self):return 4
            def save(self,path):
                path=Path(path);path.mkdir(parents=True)
                torch.save(self.state_dict(),path/'tiny_test_weights.pt')
        return TinyEncoder(Transformer(),PoolDense())

    def test_real_transformer_weights_change_and_frozen_weights_do_not(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=make_shared(root);data=ft.prepare(settings(shared),root/'work',cp)
            model=self.tiny_encoder();counts=ft.configure_trainable(model,last_layers=1)
            self.assertGreater(counts['trainable_parameters'],0)
            self.assertLess(counts['trainable_parameters'],counts['total_parameters'])
            before={name:p.detach().clone() for name,p in model.named_parameters()}
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
            log=ft.train_epoch(model,optimizer,data,1)
            self.assertTrue(np.isfinite(log['training_loss']))
            changes=[name for name,p in model.named_parameters() if not torch.equal(before[name],p.detach())]
            self.assertTrue(any('transformer.layer.2.' in name for name in changes),changes)
            for name,p in model.named_parameters():
                if not p.requires_grad:self.assertTrue(torch.equal(before[name],p.detach()),name)
            self.assertTrue(any('1.dense.' in name for name in changes))

    def test_chunk_pooling_gradients_and_full_catalogue_checkpoint_evaluation(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=make_shared(root);data=ft.prepare(settings(shared),root/'work',cp)
            model=self.tiny_encoder();ft.configure_trainable(model,last_layers=1)
            values=ft.encode_articles(model,['some long title and subtitle',None,'short'],chunk_batch=2)
            self.assertIsNone(values[1]);self.assertTrue(values[0].requires_grad)
            np.testing.assert_allclose([float(v.detach().norm()) for v in values if v is not None],[1,1],atol=1e-6)
            values[0][0].backward()
            self.assertTrue(any(p.grad is not None and bool((p.grad!=0).any()) for p in model.parameters() if p.requires_grad))
            baseline=ft.evaluate_checkpoint(model,data,'frozen',cp)
            self.assertEqual(baseline['metrics']['evaluated_requests'],2)
            self.assertEqual(baseline['metrics']['known_click_recall@200'],.5)
            vectors=pd.read_parquet(baseline['embeddings_path'])
            self.assertEqual(len(vectors),len(data['texts']))
            self.assertTrue(vectors.has_text_embedding.all())
            report=ft.save_selected(data,[baseline],[],cp)
            self.assertTrue(report['selected_is_frozen_baseline'])
            self.assertFalse(report['test_labels_used'])

if __name__=='__main__':unittest.main()
