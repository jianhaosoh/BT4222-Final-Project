"""Readable course-aligned attention reports; no model fitting or reranking."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from explain_common import GROUPS,COURSE_REFERENCES,title,write_json


def coverage_table(adapter,meta,requests,own_rankings):
    ids=meta.index.to_numpy(np.int64)
    pub=pd.to_datetime(meta.published_time).astype('datetime64[us]').astype('int64').to_numpy()
    valid=pd.notna(meta.published_time).to_numpy()
    order=np.argsort(pub[valid],kind='stable')
    ids,pub=ids[valid][order],pub[valid][order]
    text,image=adapter.flags(ids)
    request_time=pd.to_datetime(requests.impression_time).astype('datetime64[us]').astype('int64').to_numpy()
    bounds=np.searchsorted(pub,request_time,side='right')
    def fractions(text,image):
        return dict(no_text=float((~text).mean()),no_image=float((~image).mean()),
                    neither_vector=float((~text & ~image).mean()))
    row={'population':'Eligible article/request pairs','count':int(bounds.sum())}
    for name,mask in [('no_text',~text),('no_image',~image),('neither_vector',~text & ~image)]:
        cumulative=np.r_[0,np.cumsum(mask)]
        row[name]=float(cumulative[bounds].sum()/bounds.sum())
    rows=[row]
    top_ids=np.concatenate([np.asarray(own_rankings.at[b,'article_ids'],np.int64)[:100] for b in requests.behaviour_id])
    click_ids=np.concatenate([np.asarray(c,np.int64) for c in requests.article_ids_clicked])
    for label,values in [('Saved top 100',top_ids),('Observed known clicks',click_ids)]:
        t,i=adapter.flags(values)
        rows.append(dict(population=label,count=len(values),**fractions(t,i)))
    return pd.DataFrame(rows)


def attention_variation(attentions):
    rows=[]
    for bid,value in attentions.items():
        w=value['weights']
        if len(w)>1 and w.shape[1] and w[0].sum()>0:
            rows.append(dict(behaviour_id=bid,total_variation=float(np.abs(w[0]-w[1]).sum()/2),
                             candidate_a=int(value['article_ids'][0]),candidate_b=int(value['article_ids'][1])))
    return pd.DataFrame(rows,columns=['behaviour_id','total_variation','candidate_a','candidate_b'])


def make_report(adapter,baseline,sensitivity,summary,attention_summary,attentions,
                cases,removal,coverage,meta,own,output):
    out=Path(output);figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout();fig.savefig(figures/f'{name}.png',dpi=160,bbox_inches='tight');plt.close(fig)
    data=summary.sort_values('mean_absolute_change')
    fig,ax=plt.subplots(figsize=(9,4.2))
    ax.barh([GROUPS[g] for g in data.group],data.mean_absolute_change,color='#315b7d')
    ax.set_xlabel('Mean absolute neural-score change (equal request weight)')
    ax.set_title('Sensitivity to declared reference replacements — this fitted model')
    save(fig,'group_sensitivity')

    coverage.to_csv(out/'vector_coverage.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4.2));x=np.arange(len(coverage));width=.24
    for j,(key,label,color) in enumerate([('no_text','No text vector','#315b7d'),
            ('no_image','No image vector','#a5b9c7'),('neither_vector','Neither vector','#c77b4c')]):
        ax.bar(x+(j-1)*width,coverage[key]*100,width,label=label,color=color)
    ax.set_xticks(x,coverage.population);ax.set_ylabel('Share (%)');ax.set_ylim(0,100)
    ax.set_title('Availability audit: catalogue, recommendations and observed clicks')
    ax.legend();save(fig,'vector_coverage')

    variation=attention_variation(attentions);variation.to_csv(out/'candidate_attention_variation.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    req=attention_summary.groupby('behaviour_id').max_weight.mean()
    axes[0].hist(req,bins=15,color='#315b7d');axes[0].set_xlabel('Mean maximum attention weight per request')
    axes[0].set_ylabel('Requests');axes[0].set_title('Concentration of actual attention')
    axes[1].hist(variation.total_variation,bins=15,color='#c77b4c')
    axes[1].set_xlabel('Total variation between two candidate attention vectors')
    axes[1].set_title('Candidate dependence (fixed article-ID pair rule)')
    save(fig,'attention_diagnostics')

    top_events=[];case_candidates=[]
    for case in cases:
        bid=case['behaviour_id'];a=attentions[bid];history=adapter.history(bid)
        fig,axes=plt.subplots(1,3,figsize=(16,4.5))
        for j,aid in enumerate(case['article_ids']):
            row=baseline[(baseline.behaviour_id==bid)&(baseline.article_id==aid)].iloc[0]
            case_candidates.append({'case':case['case'],**row.to_dict()})
            x=np.arange(3)+(j-.5)*.32
            axes[0].bar(x,[row.neural_score,row.freshness_penalty,row.final_score],width=.32,
                        label=f'{aid}: {title(meta,aid,35)}')
            pos=list(a['article_ids']).index(aid);w=a['weights'][pos,:len(history)]
            axes[1].plot(np.arange(1,len(w)+1),w,marker='.',label=str(aid))
            for event_index in np.argsort(-w,kind='stable')[:5]:
                event=history.iloc[int(event_index)]
                top_events.append(dict(case=case['case'],behaviour_id=bid,candidate_article_id=aid,
                    history_position=int(event_index)+1,history_article_id=int(event.article_id),
                    history_time=str(pd.to_datetime(event.time_s,unit='s')),
                    history_title=title(meta,event.article_id,None),attention_weight=float(w[event_index])))
        axes[0].set_xticks(np.arange(3),['Neural','Age penalty','Final'])
        axes[0].axhline(0,color='gray',linewidth=.6);axes[0].set_title('Exact final-score components')
        axes[0].set_ylabel('Ranking-score units');axes[0].legend(fontsize=7,loc='best')
        axes[1].set_xlabel('Chronological history position (oldest → newest)')
        axes[1].set_ylabel('Actual attention weight');axes[1].set_title('Candidate-specific attention')
        axes[1].legend(title='Candidate ID',fontsize=8)
        changes=removal[removal['case']==case['case']]
        if len(changes):
            axes[2].barh(np.arange(len(changes)),changes.neural_score_change,color='#315b7d')
            axes[2].set_yticks(np.arange(len(changes)),[
                f'{r.article_id}: {r.selection.replace("_attention","")} event {r.event_index}'
                for r in changes.itertuples()])
        axes[2].axvline(0,color='gray',linewidth=.6)
        axes[2].set_xlabel('Neural-score change after deleting the event')
        axes[2].set_title('Recomputed event pathway; profile features fixed')
        fig.suptitle(f'{case["case"].replace("_"," ")} | request {bid}',y=1.04)
        save(fig,'case_'+case['case'])
    top_events=pd.DataFrame(top_events);top_events.to_csv(out/'case_top_history_events.csv',index=False)
    case_candidates=pd.DataFrame(case_candidates);case_candidates.to_csv(out/'case_candidates.csv',index=False)
    removal.to_csv(out/'event_removal_checks.csv',index=False)
    write_json(out/'cases.json',cases)

    first=summary.sort_values('mean_absolute_change',ascending=False).iloc[0]
    empty=int(attention_summary.groupby('behaviour_id').history_length.first().eq(0).sum())
    text=f'''# {own.upper()} explanation findings from this executed run

The report explains {baseline.behaviour_id.nunique()} fixed validation requests and
{len(baseline)} shared explanation pairs. Candidate selection includes both final
models' top3, original rank100, within-buffer neural favourite and known clicked
articles. This is a conditional explanation sample, not a new ranking evaluation.

The largest request-balanced mean absolute score change followed replacement of
**{GROUPS[first['group']]}** ({first['mean_absolute_change']:.4f} neural-score units).
This measures sensitivity to the declared reference, not intrinsic importance or
a causal effect. One-at-a-time effects need not sum to the final score. Absolute
score magnitudes from independently fitted models are not directly comparable.

Actual attention and high/low-weight event-removal results are saved for
{len(cases)} cases selected by fixed rules. The event-removal score changes affect
the event-history pathway while profile aggregates remain fixed. Empty-history
requests in this cohort: {empty}. LSTM attention weights belong to contextual
prefix states; HUIM attention weights belong to separately encoded events.
Neither alone establishes a user's psychological motive.

The availability table compares the eligible pool, saved top100 and observed
clicks. Imbalances can confound interpretation of modality sensitivity; they do
not by themselves prove leakage or measure relevance of unexposed articles.

Freshness is an exact fitted age penalty. Input perturbations hold this explicit
term fixed, while neural weights and model selection remain unchanged. No model
was retrained, no holdout outcomes were evaluated, and no evaluator exports were
modified. Use case_candidates.csv, the top-history table and removal checks to
write the specific successful/missed recommendation interpretations.
'''
    (out/'findings.md').write_text(text)
    return top_events,case_candidates,variation
