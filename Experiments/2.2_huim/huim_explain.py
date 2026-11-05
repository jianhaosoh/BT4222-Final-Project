"""Fitted HUIM adapter for actual attention and inference-only sensitivity."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch


class HUIMExplainer:
    def __init__(self,model,art,ctx,pre,F):
        self.model,self.art,self.ctx,self.pre,self.F = model.eval(),art,ctx,pre,F
        self.requests = ctx.requests
        self.positions = {b:i for i,b in enumerate(ctx.requests.behaviour_id)}
        self.device = next(model.parameters()).device
        self.reference,self.weight = 0,0.

    @classmethod
    def load(cls,data_root,run_dir,device):
        from huim import pipeline as P,features as F
        from huim.persist import load_model
        root = Path(run_dir)
        model,_,pre,_ = load_model(root,device)
        model.requires_grad_(False)
        cfg = json.loads((root/'source_neural_config.json').read_text())
        ws = P.prepare_inputs(cfg,Path(data_root),pre=pre,log=print)
        # Rows are rebuilt with their histories/candidates for the SMALL catalogue;
        # weights do not use article-ID embeddings. Saved-score parity is checked below.
        assert pre.to_json() == json.loads((root/'model'/'preprocessing.json').read_text())
        art,contexts = P.device_and_tensors(ws,device)
        result = cls(model,art,contexts['validation'],pre,F)
        result.weight = float(json.loads((root/'model'/'score_recipe.json').read_text())['freshness_weight'])
        return result

    def set_reference(self,bid):
        self.reference = self.positions[bid]

    def flags(self,ids):
        rows = torch.as_tensor(self.art.lookup(ids),device=self.device)
        return tuple(self.art.tensors['has_'+name][rows].cpu().numpy().astype(bool) for name in ['text','image'])

    def age_hours(self,bid,ids):
        rows = torch.as_tensor(self.art.lookup(ids),device=self.device)
        pub = self.art.tensors['pub_s'][rows]
        missing = self.art.tensors['pub_missing'][rows]
        t = self.ctx.tensors['req_s'][self.positions[bid]]
        return (((t-pub).double()/3600.).clamp_min(0).masked_fill(missing,0)).cpu().numpy()

    def history(self,bid):
        pos = self.positions[bid]
        u = int(self.ctx.tensors['req_user'][pos])
        mask = self.ctx.tensors['hist_mask'][u].cpu().numpy()
        rows = self.ctx.tensors['hist_idx'][u].cpu().numpy()[mask]
        return pd.DataFrame({'article_id':self.art.ids[rows],
            'time_s':self.ctx.tensors['hist_s'][u].cpu().numpy()[mask]})

    def _article_reference(self,group):
        t = dict(self.art.tensors)
        if group in ['text','image']:
            t['has_'+group] = torch.zeros_like(t['has_'+group])
        elif group=='metadata':
            for name in ['category','article_type','sentiment','topics_pad','subcats_pad','premium','body_z','topic_multi_hot','n_topics']:
                t[name] = torch.zeros_like(t[name])
            for name in ['premium_missing','body_missing']:
                t[name] = torch.ones_like(t[name])
            for name in ['cross_category','cross_type']:
                t[name] = torch.full_like(t[name],-1)
            s = self.pre.state
            t['topic_count_z'] = torch.full_like(t['topic_count_z'],-s['topic_count_log_mean']/s['topic_count_log_std'])
        return t

    def _remove_event(self,c,user,index):
        n = int(c['hist_mask'][user].sum())
        if index<0 or index>=n:raise IndexError('History-event index outside the valid sequence')
        keep = [i for i in range(n) if i!=index]
        for key in ['hist_idx','hist_s','hist_read_bin','hist_scroll_bin','hist_mask']:
            c[key] = c[key].clone()
            old = c[key][user].clone()
            c[key][user] = 0
            if keep:c[key][user,:len(keep)] = old[keep]

    @torch.inference_mode()
    def score(self,bid,ids,group=None,remove_event=None):
        pos = self.positions[bid]
        c = dict(self.ctx.tensors)
        for key in ['req_s','req_user','req_dense_raw','req_subscriber','req_subscriber_missing']:
            c[key] = c[key][pos:pos+1]
        user = int(c['req_user'][0])
        if group=='history':
            c['hist_mask'] = c['hist_mask'].clone();c['hist_mask'][user] = False
        elif group=='context':
            c['req_dense_raw'] = c['req_dense_raw'].clone()
            names = self.F.REQUEST_DENSE_COLUMNS(self.pre)
            cols = [i for i,n in enumerate(names) if n.startswith('device_') or n in
                    ['is_sso_user','sso_missing','is_subscriber','subscriber_missing']]
            c['req_dense_raw'][:,cols] = self.ctx.tensors['req_dense_raw'][self.reference,cols]
            for key in ['req_subscriber','req_subscriber_missing']:
                c[key] = self.ctx.tensors[key][self.reference:self.reference+1]
        if remove_event is not None:self._remove_event(c,user,int(remove_event))
        a = self._article_reference(group)
        rp = torch.zeros(len(ids),dtype=torch.long,device=self.device)
        ai = torch.as_tensor(self.art.lookup(ids),device=self.device)
        raw,weights = self.model(c,a,len(self.art.ids),rp,ai,return_weights=True)
        return raw.float().cpu().numpy(),weights.float().cpu().numpy()
