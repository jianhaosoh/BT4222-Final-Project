"""Test notebook embedding/cache plumbing without downloading a pretrained model."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import catalogue_pipeline as cp

HAS_TORCH = importlib.util.find_spec('torch') is not None

@unittest.skipUnless(HAS_TORCH, 'torch unavailable')
class EmbeddingNotebookChecks(unittest.TestCase):
    def test_generation_reuse_resume_and_source_guard(self):
        import torch
        nb = json.loads((Path(__file__).resolve().parents[1]/'experiment.ipynb').read_text())
        code = ''.join(next(c for c in nb['cells'] if c.get('id')=='generate-embeddings')['source'])
        ast.parse(code)
        counters = {'loads':0, 'chunks':0}
        batch_sizes=[]
        class Tokenizer:
            model_input_names = ['input_ids','attention_mask']
            def __call__(self,text,**kwargs):
                assert kwargs['return_overflowing_tokens']
                ids = [ord(c)%9+1 for c in text]
                parts=[ids[i:i+4] for i in range(0,len(ids),4)]
                masks=[[1]*len(p)+[0]*(4-len(p)) for p in parts]
                data=[p+[0]*(4-len(p)) for p in parts]
                counters['chunks']+=len(parts)
                return {'input_ids':torch.tensor(data),'attention_mask':torch.tensor(masks),
                        'special_tokens_mask':torch.zeros((len(parts),4),dtype=torch.long)}
        class Encoder:
            max_seq_length=4
            device='cpu'
            tokenizer=Tokenizer()
            def __init__(self,*args,**kwargs):counters['loads']+=1
            def eval(self):return self
            def get_sentence_embedding_dimension(self):return 2
            def __call__(self,features):
                batch_sizes.append(len(features['input_ids']))
                return {'sentence_embedding':torch.stack([features['input_ids'].sum(1),features['attention_mask'].sum(1)],dim=1).float()}
        st=types.ModuleType('sentence_transformers');st.SentenceTransformer=Encoder
        hub=types.ModuleType('huggingface_hub')
        hub.HfApi=lambda:types.SimpleNamespace(model_info=lambda name:types.SimpleNamespace(sha='fixed-test-revision'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for part in ['train_val','test']:(root/part).mkdir()
            pd.DataFrame({'article_id':[1,2],'title':['Long title',None],
                          'subtitle':['Subtitle',None],'body':['body '*30,None]}).to_parquet(root/'train_val/articles.parquet',index=False)
            pd.DataFrame({'article_id':[1,3],'title':['Different later snapshot','Third'],
                          'subtitle':['Other','Sub'],'body':['Other body','Third body']}).to_parquet(root/'test/articles.parquet',index=False)
            env={'np':np,'pd':pd,'Path':Path,'json':json,'hashlib':hashlib,'cp':cp,
                 'DRIVE_ROOT':root,'EXPERIMENT_ID':'bert_test','MODEL_NAME':'org/model',
                 'TEXT_VARIANT':'title_subtitle_body','INTRO_CHARACTERS':1000,
                 'ENCODING_BATCH_SIZE':3,'ARTICLE_BATCH_SIZE':128,'CONFIG':{'paths':{},'embedding_source':{}}}
            with patch.dict(sys.modules,{'sentence_transformers':st,'huggingface_hub':hub}),patch.object(torch.cuda,'is_available',return_value=False):
                exec(compile(code,'embedding_cell','exec'),env)
                frames={p:pd.read_parquet(path).set_index('article_id') for p,path in env['EMBEDDING_PATHS'].items()}
                np.testing.assert_allclose(frames['train_val'].loc[1,'text_embedding'],frames['test'].loc[1,'text_embedding'])
                self.assertFalse(frames['train_val'].loc[2,'has_text_embedding'])
                self.assertAlmostEqual(np.linalg.norm(frames['test'].loc[3,'text_embedding']),1,places=6)
                self.assertGreater(counters['chunks'],10)
                text=env['text_for_article']({'title':'A','subtitle':'B','body':'C'})
                self.assertEqual(text,'title: A\nsubtitle: B\nbody: C')
                # Four short articles share a three-chunk GPU batch, then one remainder.
                batch_sizes.clear()
                encoder=Encoder()
                texts=['a','b','c','d',None,'']
                result=env['encode_articles'](texts,encoder)
                self.assertEqual(batch_sizes,[3,1])
                for text,vector in zip(texts,result):
                    if not text:
                        self.assertIsNone(vector)
                    else:
                        expected=np.asarray([ord(text)%9+1,1],dtype='float64')
                        expected/=np.linalg.norm(expected)
                        np.testing.assert_allclose(vector,expected,atol=1e-6)
                # Pooling remains equal to independent article processing for mixed lengths.
                texts=['one','much longer text '*13,None,'two words']
                mixed=env['encode_articles'](texts,encoder)
                single=[env['encode_articles']([text],encoder)[0] for text in texts]
                for actual,expected in zip(mixed,single):
                    if expected is None:self.assertIsNone(actual)
                    else:np.testing.assert_allclose(actual,expected,atol=1e-6)
                with patch.dict(env,{'ENCODING_BATCH_SIZE':0}):
                    with self.assertRaisesRegex(ValueError,'positive'):
                        env['encode_articles'](['a'],encoder)
                before=dict(counters)
                exec(compile(code,'embedding_cell','exec'),env)
                self.assertEqual(counters,before)
                # Simulate interruption after vector caching but before final export completion.
                meta=json.loads(env['metadata_path'].read_text());meta['complete']=False
                cp.save_json(env['metadata_path'],meta)
                env['EMBEDDING_PATHS']['test'].unlink()
                exec(compile(code,'embedding_cell','exec'),env)
                self.assertEqual(counters['chunks'],before['chunks'])
                self.assertTrue(env['EMBEDDING_PATHS']['test'].exists())
                # A changed source cannot silently reuse old article vectors.
                changed=pd.read_parquet(root/'test/articles.parquet');changed.loc[1,'body']='Changed text'
                changed.to_parquet(root/'test/articles.parquet',index=False)
                with self.assertRaisesRegex(ValueError,'changed'):
                    exec(compile(code,'embedding_cell','exec'),env)

if __name__=='__main__':unittest.main()
