"""Fitted LSTM input adapter; interventions never change weights or source files."""
from dataclasses import fields
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch


def ordered_requests(requests,user_id,impression_id):
    ids = np.asarray(user_id).astype(str)+':'+np.asarray(impression_id).astype(str)
    return requests.set_index('behaviour_id').loc[ids].reset_index()


class LSTMExplainer:
    def __init__(self, model, articles, arrays, tensors, preprocessing, requests, D, M):
        self.model, self.articles, self.arrays, self.tensors = model.eval(), articles, arrays, tensors
        self.pre, self.requests, self.D, self.M = preprocessing, requests, D, M
        self.device = tensors.device
        self.positions = {b:i for i,b in enumerate(requests.behaviour_id)}
        self.reference = 0
        self.weight = 0.

    @classmethod
    def load(cls, data_root, run_dir, device):
        import lstm_data as D
        import lstm_model as M
        run_dir = Path(run_dir); bundle = run_dir/'model'
        pre = json.loads((bundle/'preprocessing.json').read_text())
        with np.load(bundle/'article_inputs.npz',allow_pickle=False) as saved:
            values = {f.name:saved[f.name] for f in fields(D.ArticleTable)}
        values['numeric_names'] = values['numeric_names'].tolist()
        articles = D.ArticleTable(**values)
        cfg = M.ModelConfig.from_json(json.loads((bundle/'model_config.json').read_text()))
        model = M.build_model(cfg,M.article_table_tensors(articles,device),device)
        model.load_state_dict(torch.load(bundle/'model_state.pt',map_location=device,weights_only=True))
        model.requires_grad_(False)
        paths,_ = D.resolve_paths(Path(data_root),required=['features_validation','history_validation','behaviors_validation'])
        frame,info = D.load_candidate_table(paths['features_validation'],'validation')
        offsets = info['offsets']
        history = D.build_history(paths['history_validation'],pd.Timestamp(frame.impression_time.min()),
                                  pre['history_max_len'],users_needed=frame.user_id.unique())
        arrays = D.build_split_arrays('validation',frame,offsets,history,articles,pre['numeric_stats'],pre['device_columns'])
        requests = pd.read_parquet(paths['behaviors_validation'],columns=[
            'user_id','impression_id','impression_time','article_ids_inview','article_ids_clicked'])
        requests['behaviour_id'] = requests.user_id.astype(str)+':'+requests.impression_id.astype(str)
        requests = ordered_requests(requests,arrays.user_id,arrays.impression_id)
        result = cls(model,articles,arrays,M.SplitTensors(arrays,device),pre,requests,D,M)
        result.weight = float(json.loads((bundle/'score_recipe.json').read_text())['freshness_weight'])
        return result

    def set_reference(self, bid):
        self.reference = self.positions[bid]

    def flags(self, ids):
        rows = self.articles.rows_for(np.asarray(ids,np.int64))
        return self.articles.has_text[rows],self.articles.has_image[rows]

    def _age(self,bid,ids):
        rows = self.articles.rows_for(np.asarray(ids,np.int64))
        pos = self.positions[bid]
        return self.D.candidate_age_hours(self.articles.published_s[rows],self.articles.published_missing[rows],
                                          np.full(len(rows),self.arrays.imp_time_s[pos],np.int64))

    def age_hours(self,bid,ids):
        # Match the frozen inference helper's explicit penalty, including its
        # published_s=0 convention for recorded candidates with no publication.
        # Neural candidate features still use _age() and its missingness flag.
        rows=self.articles.rows_for(np.asarray(ids,np.int64))
        t=self.arrays.imp_time_s[self.positions[bid]]
        return np.maximum((t-self.articles.published_s[rows]).astype(np.float64)/3600.,0.)

    def history(self,bid):
        u = int(self.arrays.imp_user_row[self.positions[bid]])
        n = int(self.arrays.hist_lengths[u])
        return pd.DataFrame({'article_id':self.articles.article_ids[self.arrays.hist_article_row[u,:n]],
                             'time_s':self.arrays.hist_times_s[u,:n]})

    def _article_reference(self,group):
        original = self.model.articles._table
        t = dict(original)
        if group in ['text','image']:
            t['has_'+group] = torch.zeros_like(t['has_'+group])
            t['numeric'] = t['numeric'].clone()
            name = 'has_text_vector' if group=='text' else 'has_image_vector'
            t['numeric'][:,self.articles.numeric_names.index(name)] = 0
        elif group=='metadata':
            for key in ['category','article_type']:
                t[key] = torch.ones_like(t[key]);t[key][0]=0
            t['topics'] = torch.zeros_like(t['topics'])
            t['numeric'] = t['numeric'].clone()
            reference = {'log_body_words_z':0.,'body_words_missing':1.,'sentiment_z':0.,
                         'sentiment_missing':1.,'premium':0.,'premium_missing':1.}
            for key,value in reference.items():
                t['numeric'][1:,self.articles.numeric_names.index(key)] = value
        return t

    def _remove_event(self,batch,index):
        n = int(batch['hist_len'][0]); keep = [i for i in range(n) if i!=index]
        if index<0 or index>=n:raise IndexError('History-event index outside the valid sequence')
        for key in ['hist_rows','hist_feats','hist_time']:
            old = batch[key];new = torch.zeros_like(old)
            if keep:new[0,:len(keep)] = old[0,keep]
            batch[key] = new
        batch['hist_len'] = torch.as_tensor([len(keep)],device=self.device)
        if keep:
            stats = self.pre['numeric_stats']['log_gap_prev_h']
            times = batch['hist_time'][0,:len(keep)]
            gaps = (times[1:]-times[:-1]).double()/3600.
            batch['hist_feats'][0,0,self.M.GAP_PREV_FEATURE] = 0.
            batch['hist_feats'][0,0,self.M.GAP_PREV_MISSING] = 1.
            if len(keep)>1:
                batch['hist_feats'][0,1:len(keep),self.M.GAP_PREV_FEATURE] = (
                    (torch.log1p(gaps.clamp_min(0))-stats['mean'])/stats['std']).float()
                batch['hist_feats'][0,1:len(keep),self.M.GAP_PREV_MISSING] = 0.

    @torch.inference_mode()
    def score(self,bid,ids,group=None,remove_event=None):
        ids = np.asarray(ids,np.int64);pos = self.positions[bid]
        batch = self.tensors.batch(torch.as_tensor([pos],device=self.device))
        batch['cand_rows'] = torch.as_tensor(self.articles.rows_for(ids),device=self.device)
        batch['cand_imp'] = torch.zeros(len(ids),dtype=torch.long,device=self.device)
        batch['cand_feats'] = torch.as_tensor(self.D.candidate_features(self._age(bid,ids),self.pre['numeric_stats']),device=self.device)
        if group=='history':
            batch['hist_len'] = torch.zeros_like(batch['hist_len'])
        if group=='context':
            batch['imp_ctx'] = batch['imp_ctx'].clone()
            cols = [i for i,n in enumerate(self.arrays.context_names)
                    if n.startswith('ohe_device_type_') or n.startswith('is_sso_user') or n.startswith('is_subscriber')]
            batch['imp_ctx'][:,cols] = self.tensors.imp_ctx[self.reference,cols]
        if remove_event is not None:self._remove_event(batch,int(remove_event))
        original = self.model.articles._table
        try:
            self.model.articles.set_table(self._article_reference(group))
            raw,weights = self.model(batch,return_attention=True)
        finally:
            self.model.articles.set_table(original)
        if weights is None:weights = torch.zeros((len(ids),0),device=self.device)
        return raw.float().cpu().numpy(),weights.float().cpu().numpy()
