"""Shared validation-only explanation protocol. No training or model selection."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

MODEL_RUNS = {
    'lstm': 'Experiments/2.1_lstm/2.1.3_improved_run/20261104_175920_improved_seed42',
    'huim': 'Experiments/2.2_huim/2.2.2_freshness_run/20261104_234712_freshness_seed42',
}
GROUPS = {
    'text': 'Text representation + availability',
    'image': 'Image representation + availability',
    'metadata': 'Non-temporal article metadata',
    'history': 'Event-history pathway (profiles fixed)',
    'context': 'Device/account observed reference',
}
COURSE_REFERENCES = [
    'Week 8: Topic 7 RNN and Transformer Part B, pp24–28; '
    'BT4222_Sequential_Recommendation_LSTM_with_Attention.ipynb, source cell26.',
    'Week 11: Topic 9 Advanced Applications of Deep Learning Part B, '
    'pp16–20 and pp33–35: attention interpretation and confounding audit.',
    'Week 1–2: Topic 1 Introduction, pp29–32: interpretation, causality and '
    'attention/LIME/SHAP. SHAP/LIME are named rather than demonstrated.',
    'Week 12: Topic 11 Ensemble Method, pp15–16: RF importance and correlation '
    'caveats; its impurity importance is not a neural importance estimator.',
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')


def sample_requests(requests, n=200, seed=42):
    r = requests.copy()
    r['_priority'] = [hashlib.blake2b(f'{seed}:validation_explainability:{b}'.encode(),
                                     digest_size=8).hexdigest() for b in r.behaviour_id]
    return r.sort_values(['_priority','behaviour_id']).head(n).drop(columns='_priority').reset_index(drop=True)


def exposure_status(article_id, inview, clicked):
    if int(article_id) in set(map(int, clicked)):
        return 'clicked'
    if int(article_id) in set(map(int, inview)):
        return 'shown_not_clicked'
    return 'unexposed_unknown'


def score_parts(raw, age_hours, weight):
    raw = np.asarray(raw, np.float32)
    penalty = -(float(weight)*torch.log1p(torch.as_tensor(age_hours, dtype=torch.float64).clamp_min(0)).float()).numpy()
    return dict(neural_score=raw, freshness_penalty=penalty,
                final_score=raw+penalty, age_hours=np.asarray(age_hours, float))


def sensitivity_summary(rows):
    r = rows.assign(absolute_change=rows.neural_delta.abs())
    # Equal request weight, even when the explanation pair cohort sizes differ.
    per_request = r.groupby(['group','behaviour_id'],as_index=False)[['neural_delta','absolute_change']].mean()
    return per_request.groupby('group',as_index=False).agg(
        mean_signed_change=('neural_delta','mean'),
        mean_absolute_change=('absolute_change','mean'), requests=('behaviour_id','nunique'))


def load_metadata(data_root):
    frames = []
    for split in ['train_val','test']:
        path = Path(data_root)/split/'articles.parquet'
        available = pq.ParquetFile(path).schema_arrow.names
        cols = [c for c in ['article_id','title','published_time','category','category_str','article_type','premium']
                if c in available]
        frames.append(pd.read_parquet(path, columns=cols))
    meta = pd.concat(frames,ignore_index=True).drop_duplicates('article_id',keep='first').set_index('article_id')
    return meta


def title(meta, article_id, limit=90):
    if int(article_id) in meta.index and 'title' in meta:
        value = meta.at[int(article_id),'title']
        if isinstance(value,str) and value.strip():
            return value if limit is None else value[:limit]
    return f'Article {int(article_id)}'


def read_sources(data_root, requests):
    bids = requests.behaviour_id.astype(str).tolist()
    sources = {}
    for name, relative in MODEL_RUNS.items():
        root = Path(data_root)/relative
        ranking = pd.read_parquet(root/'rankings'/'validation'/'submission_top_200.parquet').set_index('behaviour_id')
        ranking = ranking.loc[bids]
        score_parts_ = []
        for path in [root/'predictions'/'validation'/'submission_article_scores.parquet',
                     root/'diagnostics'/'recorded_slates'/'validation_scores.parquet']:
            score_parts_.append(pd.read_parquet(path, columns=['behaviour_id','article_id','score'],
                                                filters=[('behaviour_id','in',bids)]))
        scores = pd.concat(score_parts_,ignore_index=True).drop_duplicates(['behaviour_id','article_id'])
        sources[name] = {'root':root, 'rankings':ranking,
            'scores':scores.set_index(['behaviour_id','article_id']).score,
            'recipe':json.loads((root/'model'/'score_recipe.json').read_text())}
    return sources


def build_pairs(requests, sources, meta):
    """Shared cohort: both models' top3, rank100, within-buffer neural favourite,
    plus original known clicks. This is explanation-only; no shortlist is edited.
    """
    rows = []
    for r in requests.itertuples():
        candidates = set(map(int,r.article_ids_clicked))
        for source in sources.values():
            ids = list(map(int,source['rankings'].at[r.behaviour_id,'article_ids']))
            candidates.update(ids[:3])
            if len(ids)>=100:
                candidates.add(ids[99])
            age = np.maximum((pd.Timestamp(r.impression_time)-pd.to_datetime(meta.reindex(ids).published_time))
                             .dt.total_seconds().to_numpy()/3600.,0)
            final = np.array([source['scores'].loc[(r.behaviour_id,a)] for a in ids])
            raw = final+float(source['recipe']['freshness_weight'])*np.log1p(age).astype(np.float32)
            if len(ids):
                candidates.add(ids[int(np.argmax(raw))])
        for aid in sorted(candidates):
            row = dict(behaviour_id=r.behaviour_id,user_id=int(r.user_id),impression_id=int(r.impression_id),
                article_id=aid, outcome=exposure_status(aid,r.article_ids_inview,r.article_ids_clicked),
                title=title(meta,aid,None))
            for name, source in sources.items():
                ids = list(map(int,source['rankings'].at[r.behaviour_id,'article_ids']))
                row[f'{name}_rank_in_buffer'] = ids.index(aid)+1 if aid in ids else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def choose_cases(baseline, requests, sources, own):
    """First qualifying request in the predeclared hash order, not best-looking."""
    selected = {}
    for r in requests.itertuples():
        g = baseline[baseline.behaviour_id==r.behaviour_id]
        ids = list(map(int,sources[own]['rankings'].at[r.behaviour_id,'article_ids']))
        clicked = g[g.outcome=='clicked']
        success = clicked[clicked[f'{own}_rank_in_buffer'].le(100)]
        misses = clicked[clicked[f'{own}_rank_in_buffer'].gt(100) | clicked[f'{own}_rank_in_buffer'].isna()]
        candidates = []
        if len(success): candidates.append(('recovered_known_click',[int(success.iloc[0].article_id),ids[0]]))
        if len(misses): candidates.append(('missed_known_click',[int(misses.iloc[0].article_id),ids[0]]))
        other = 'huim' if own=='lstm' else 'lstm'
        other_top = int(sources[other]['rankings'].at[r.behaviour_id,'article_ids'][0])
        if ids[0]!=other_top: candidates.append(('model_disagreement',[ids[0],other_top]))
        inside = g[g[f'{own}_rank_in_buffer'].notna()]
        favourite = inside.sort_values(['neural_score','article_id'],ascending=[False,True]).iloc[0]
        if int(favourite.article_id)!=ids[0]:
            candidates.append(('freshness_ordering_in_buffer',[ids[0],int(favourite.article_id)]))
        for label, articles in candidates:
            if label not in selected:
                selected[label] = {'case':label,'behaviour_id':r.behaviour_id,
                                   'article_ids':list(dict.fromkeys(articles))}
    return list(selected.values())


def explain_pairs(adapter, pairs, sources, own, output):
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    baseline, perturbations, attention_summary, attentions = [], [], [], {}
    for bid, g in pairs.groupby('behaviour_id',sort=False):
        ids = g.article_id.to_numpy(np.int64)
        raw, weights = adapter.score(bid,ids)
        frame = g.reset_index(drop=True).copy()
        for key,value in score_parts(raw,adapter.age_hours(bid,ids),adapter.weight).items():frame[key]=value
        expected = np.array([sources[own]['scores'].get((bid,int(a)),np.nan) for a in ids],float)
        frame['saved_score'] = expected
        frame['saved_score_difference'] = frame.final_score-expected
        baseline.append(frame)
        attentions[bid] = {'article_ids':ids,'weights':weights}
        h = adapter.history(bid)
        for j,aid in enumerate(ids):
            w = weights[j,:len(h)]
            entropy = float(-(w*np.log(np.maximum(w,1e-12))).sum())
            attention_summary.append(dict(behaviour_id=bid,article_id=int(aid),history_length=len(h),
                max_weight=float(w.max(initial=0)), normalised_entropy=entropy/np.log(len(h)) if len(h)>1 else 0.,
                recent5_mass=float(w[-5:].sum()) if len(w) else 0.))
        for group in GROUPS:
            modified,_ = adapter.score(bid,ids,group=group)
            for aid,delta in zip(ids,modified-raw):
                perturbations.append(dict(behaviour_id=bid,article_id=int(aid),group=group,neural_delta=float(delta)))
    baseline = pd.concat(baseline,ignore_index=True)
    err = baseline.saved_score_difference.dropna().abs()
    assert len(err) and float(err.max())<=1e-4, f'Baseline does not match saved scores: {err.max()}'
    sensitivity = pd.DataFrame(perturbations)
    summary = sensitivity_summary(sensitivity)
    for name,frame in [('explained_pairs',baseline),('group_score_changes',sensitivity),
                       ('group_sensitivity_summary',summary),('attention_summary',pd.DataFrame(attention_summary))]:
        frame.to_csv(output/f'{name}.csv',index=False)
    width=max((v['weights'].shape[1] for v in attentions.values()),default=0)
    saved_weights=np.zeros((len(baseline),width),np.float32)
    start=0
    for value in attentions.values():
        w=value['weights'];saved_weights[start:start+len(w),:w.shape[1]]=w;start+=len(w)
    np.savez_compressed(output/'attention_weights.npz',
        behaviour_id=baseline.behaviour_id.to_numpy().astype(str),
        article_id=baseline.article_id.to_numpy(np.int64),weights=saved_weights)
    write_json(output/'baseline_parity.json',{'pairs_compared':len(err),'max_abs_difference':float(err.max()),'tolerance':1e-4,'passed':True})
    return baseline,sensitivity,summary,pd.DataFrame(attention_summary),attentions


def event_removal(adapter, cases, attentions, meta):
    rows = []
    for case in cases:
        bid = case['behaviour_id'];data = attentions[bid];history=adapter.history(bid)
        for aid in case['article_ids']:
            pos = list(data['article_ids']).index(aid)
            weights = data['weights'][pos,:len(history)]
            if not len(weights):continue
            base,_ = adapter.score(bid,[aid])
            for label,index in [('high_attention',int(np.argmax(weights))),('low_attention',int(np.argmin(weights)))]:
                modified,_ = adapter.score(bid,[aid],remove_event=index)
                event=history.iloc[index]
                rows.append(dict(case=case['case'],behaviour_id=bid,article_id=aid,selection=label,
                    event_index=index+1,event_article_id=int(event.article_id),
                    event_title=title(meta,event.article_id,None),attention_weight=float(weights[index]),
                    neural_score_change=float(modified[0]-base[0])))
    return pd.DataFrame(rows)
