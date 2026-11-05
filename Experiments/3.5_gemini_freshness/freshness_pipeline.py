"""Gemini history similarity plus soft publication-age penalties.

Candidate eligibility is unchanged. Validation selects the formula/weight; test
labels never participate in selection or automatic test export metrics.
"""
from contextlib import closing
import csv
import json
import math
from pathlib import Path
import shutil
import sqlite3
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import catalogue_pipeline as cp

VERSION='gemini_soft_freshness_at100_v2'
EVALUATION_K=100


def retrieval_metrics(ranking,known):
    """Evaluate the first 100 while retaining all original known clicks."""
    if not known:return None
    retained=ranking[:EVALUATION_K]
    hits=[int(a in known) for a in retained]
    ideal=sum(1/math.log2(i+2) for i in range(min(EVALUATION_K,len(known))))
    dcg=sum(hit/math.log2(i+2) for i,hit in enumerate(hits))
    return sum(hits)/len(known),dcg/ideal


def variants(config):
    weights=config.get('freshness_weights',[0.,.01,.03,.1])
    if not weights or any(not np.isfinite(w) or w<0 for w in weights):
        raise ValueError('freshness_weights must contain finite nonnegative weights')
    weights=sorted(set([0.]+[float(w) for w in weights]))
    result=[{'name':'similarity_only' if w==0 else f'similarity_lambda_{w:g}'.replace('.','p'),
             'kind':'similarity_plus_age','weight':w} for w in weights]
    result.append({'name':'freshness_only','kind':'freshness_only','weight':None})
    return result


def blend(similarities,ages,variant):
    similarities=np.asarray(similarities,dtype='float64');ages=np.asarray(ages,dtype='float64')
    if not np.isfinite(ages).all() or (ages<0).any():raise ValueError('Article ages must be finite and nonnegative')
    if variant['kind']=='freshness_only':return -np.log1p(ages)
    return similarities-float(variant['weight'])*np.log1p(ages)


def rank_request(request,catalogue,profiles,vectors,options,batch_size=4096):
    """Compute history cosines once per batch and reuse them across all formulas."""
    ids=cp.eligible_ids(catalogue,request[3]);best={v['name']:[] for v in options}
    for start in range(0,len(ids),batch_size):
        part=ids[start:start+batch_size]
        similarities=cp.cosine_scores(part,int(request[1]),profiles,vectors)
        ages=((pd.Timestamp(request[3])-catalogue.loc[part,'published_time']).dt.total_seconds()/3600).to_numpy(dtype='float64')
        for option in options:
            scores=blend(similarities,ages,option)
            if not np.isfinite(scores).all():raise ValueError('Nonfinite ranking scores')
            positions=np.lexsort((part,-scores))[:201]
            entries=[(float(scores[i]),int(part[i]),float(similarities[i]),float(ages[i])) for i in positions]
            best[option['name']]=sorted(best[option['name']]+entries,key=lambda row:(-row[0],row[1]))[:201]
    output={}
    for option in options:
        top=best[option['name']];retained=top[:200]
        output[option['name']]={
            'article_ids':[row[1] for row in retained], 'scores':[row[0] for row in retained],
            'similarity_scores':[row[2] for row in retained], 'article_age_hours':[row[3] for row in retained],
            'ties_crossing_100':len(top)>100 and top[99][0]==top[100][0],
            'ties_crossing_200':len(top)>200 and top[199][0]==top[200][0],
        }
    return output,ids


def _signature(prepared):
    return json.loads(json.dumps({'version':VERSION,'base_version':cp.PIPELINE_VERSION,
        'config':prepared['config'],'inputs':prepared['audit']['inputs']},default=str))


def _coverage(known,eligible,catalogue,when):
    reasons={'missing_article':0,'missing_publication':0,'future_publication':0}
    for aid in known-eligible:
        if aid not in catalogue.index:reasons['missing_article']+=1
        elif pd.isna(catalogue.loc[aid,'published_time']):reasons['missing_publication']+=1
        else:reasons['future_publication']+=1
    return reasons


def evaluate(prepared,requests,split,options,profiles,per_request_path=None):
    """Macro observed-click recovery, retaining original positive denominators."""
    if split=='test':raise ValueError('Test labels are not used for freshness selection')
    values={o['name']:[] for o in options};eligible_values={o['name']:[] for o in options}
    tie_counts={o['name']:0 for o in options}
    counts={'requests_scored':0,'requests_without_known_clicks':0,'empty_eligible_pools':0,
        'known_clicks':0,'known_clicks_excluded_by_eligibility':0,'requests_without_usable_user_profile':0}
    reasons={'missing_article':0,'missing_publication':0,'future_publication':0}
    pairs=0;started=time.perf_counter()
    stream=Path(per_request_path).open('w',newline='') if per_request_path else None
    writer=csv.writer(stream) if stream else None
    if writer:writer.writerow(['behaviour_id','variant','known_clicks','recall@100','ndcg@100'])
    try:
        with closing(sqlite3.connect(prepared['dbs'][split])) as db:
            for number,request in enumerate(requests,1):
                ranked,ids=rank_request(request,prepared['catalogue'],profiles,prepared['vectors'],options,
                    prepared['config'].get('catalogue_batch_size',4096))
                eligible=set(ids);pairs+=len(ids);counts['requests_scored']+=1
                counts['empty_eligible_pools']+=not len(ids)
                counts['requests_without_usable_user_profile']+=profiles.get(int(request[1]),(None,0))[0] is None
                known={int(row[0]) for row in db.execute('SELECT article_id FROM candidates WHERE behaviour_id=? AND clicked=1',(request[0],))}
                counts['known_clicks']+=len(known)
                counts['known_clicks_excluded_by_eligibility']+=len(known-eligible)
                for key,value in _coverage(known,eligible,prepared['catalogue'],request[3]).items():reasons[key]+=value
                if not known:counts['requests_without_known_clicks']+=1
                for option in options:
                    name=option['name'];prediction=ranked[name]
                    tie_counts[name]+=prediction['ties_crossing_100']
                    result=retrieval_metrics(prediction['article_ids'],known)
                    if result is not None:
                        values[name].append(result)
                        if known&eligible:eligible_values[name].append(len(set(prediction['article_ids'][:EVALUATION_K])&known)/len(known&eligible))
                    if writer:writer.writerow([request[0],name,len(known),*(result if result else [None,None])])
                if number%100==0:print(f'Freshness validation: {number:,} requests; {time.perf_counter()-started:.1f}s',flush=True)
    finally:
        if stream:stream.close()
    if not any(values.values()):raise ValueError('No requests with known clicks for validation')
    results=[]
    for option in options:
        name=option['name'];rows=values[name]
        metrics=dict(counts,
            known_click_recall_at_100=float(np.mean([r[0] for r in rows])),
            observed_click_ndcg_at_100=float(np.mean([r[1] for r in rows])),
            evaluated_requests=len(rows),known_click_exclusion_reasons=reasons,
            known_click_eligibility_coverage=(1-counts['known_clicks_excluded_by_eligibility']/counts['known_clicks']) if counts['known_clicks'] else None,
            eligible_only_recall_at_100=float(np.mean(eligible_values[name])) if eligible_values[name] else None,
            requests_with_ties_crossing_100=tie_counts[name])
        for old,new in [('known_click_recall_at_100','known_click_recall@100'),('observed_click_ndcg_at_100','observed_click_ndcg@100'),('eligible_only_recall_at_100','eligible_only_recall@100')]:
            metrics[new]=metrics.pop(old)
        metrics['outcomes']='Unshown outcomes unknown; original known-click denominators retained.'
        results.append({'variant':option,'metrics':metrics})
    return {'results':results,'candidate_pairs_scored_once':pairs,'seconds':time.perf_counter()-started,
            'request_ids':[r[0] for r in requests]}


def select(results):
    # Options are baseline, increasing penalties, then freshness-only: earlier wins exact ties.
    return max(results,key=lambda row:(row['metrics']['known_click_recall@100'],row['metrics']['observed_click_ndcg@100']))


def evaluate_grid(prepared):
    work=prepared['work'];signature=_signature(prepared);path=work/'freshness_grid.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['signature']!=signature:raise ValueError('Freshness run changed; use a new run ID')
        print('Reusing completed freshness comparison.',flush=True);return saved
    options=variants(prepared['config']);request_list=prepared['validation_requests']
    pairs=cp.budget(request_list,prepared['catalogue'])['eligible_pairs']
    if pairs>prepared['config'].get('max_export_pairs',150_000_000):raise RuntimeError('Validation scoring exceeds declared pair budget')
    profiles,history=cp.profiles_for(prepared,'validation')
    try:report=evaluate(prepared,request_list,'validation',options,profiles,work/'validation_per_request.csv')
    finally:profiles.close()
    report.update(signature=signature,selected=select(report['results']),history_coverage=history,
        selection='Validation Recall@100, then nDCG@100; earlier option wins exact ties.',
        formula='similarity - lambda * log1p(age_hours); freshness-only = -log1p(age_hours)',
        age_reference='request timestamp minus publication timestamp, never current clock time',
        test_labels_used=False,learned_parameters=False,evaluation_cutoff=EVALUATION_K,eligibility_policy=cp.POLICY)
    cp.save_json(path,report)
    comparison=[dict(variant=r['variant']['name'],weight=r['variant']['weight'],
        recall_at_100=r['metrics']['known_click_recall@100'],ndcg_at_100=r['metrics']['observed_click_ndcg@100'],
        evaluated_requests=r['metrics']['evaluated_requests']) for r in report['results']]
    pd.DataFrame(comparison).to_csv(work/'freshness_comparison.csv',index=False)
    return report


def evaluate_saved_validation(prepared,option,profiles):
    """Evaluate saved lists without an additional catalogue-scoring pass."""
    values=[];eligible_values=[];without=empty=ties=known_count=excluded=missing_profiles=request_count=0
    reasons={'missing_article':0,'missing_publication':0,'future_publication':0}
    path=prepared['work']/'rankings/validation/submission_top_200.csv'
    catalogue=prepared['catalogue']
    with path.open(newline='') as stream,closing(sqlite3.connect(prepared['dbs']['validation'])) as db:
        reader=csv.DictReader(stream)
        for request in cp.request_iter(prepared['dbs']['validation'],prepared['config']['export_requests']['validation']):
            row=next(reader,None)
            if row is None or row['behaviour_id']!=request[0]:raise ValueError('Saved validation request mismatch')
            ranking=json.loads(row['article_ids']);request_count+=1
            empty+=int(row['eligible_article_count'])==0;ties+=row['ties_crossing_100']=='True'
            missing_profiles+=profiles.get(int(request[1]),(None,0))[0] is None
            known={int(r[0]) for r in db.execute('SELECT article_id FROM candidates WHERE behaviour_id=? AND clicked=1',(request[0],))}
            if not known:without+=1;continue
            eligible_known={a for a in known if a in catalogue.index and pd.notna(catalogue.loc[a,'published_time']) and catalogue.loc[a,'published_time']<=pd.Timestamp(request[3])}
            values.append(retrieval_metrics(ranking,known));known_count+=len(known);excluded+=len(known-eligible_known)
            if eligible_known:eligible_values.append(len(set(ranking[:EVALUATION_K])&eligible_known)/len(eligible_known))
            for key,value in _coverage(known,eligible_known,catalogue,request[3]).items():reasons[key]+=value
        if next(reader,None) is not None:raise ValueError('Unexpected extra saved validation request')
    return {'known_click_recall@100':float(np.mean([v[0] for v in values])) if values else None,
        'observed_click_ndcg@100':float(np.mean([v[1] for v in values])) if values else None,
        'evaluated_requests':len(values),'requests_scored':request_count,'requests_without_known_clicks':without,
        'known_clicks':known_count,'known_clicks_excluded_by_eligibility':excluded,
        'known_click_eligibility_coverage':1-excluded/known_count if known_count else None,
        'eligible_only_recall@100':float(np.mean(eligible_values)) if eligible_values else None,
        'known_click_exclusion_reasons':reasons,'empty_eligible_pools':empty,
        'requests_with_ties_crossing_100':ties,'requests_without_usable_user_profile':missing_profiles,
        'outcomes':'Unshown outcomes unknown; original known-click denominators retained.',
        'source':'saved and independently verified validation shortlists; no rescoring for metrics'}


def export_selected(prepared,grid):
    """Export only the validation-selected rule, with independent shortlist replay."""
    work=prepared['work'];config=prepared['config'];signature=_signature(prepared)
    if grid['signature']!=signature:raise ValueError('Grid/config changed; use matching prepared inputs')
    cp.check_cost(prepared)
    result_path=work/'freshness_export.json'
    if result_path.exists():
        saved=json.loads(result_path.read_text())
        if saved['signature']!=signature:raise ValueError('Export settings changed; use a new run ID')
        print('Reusing completed selected export.',flush=True);return saved
    option=grid['selected']['variant'];checks={};history={};request_counts={}
    # Conservative estimate includes IDs, final scores, similarities and ages.
    estimate=sum(256*b['requests']+160*b['shortlisted_articles'] for b in prepared['budgets'].values())
    if estimate>config.get('max_export_gb',10)*1e9 or estimate*2>shutil.disk_usage(work).free:
        raise RuntimeError('Freshness export exceeds disk/copy budget')
    for split in cp.SPLITS:
        profiles,history[split]=cp.profiles_for(prepared,split)
        limit=config['export_requests'][split]
        directory=work/f'rankings/{split}';directory.mkdir(parents=True,exist_ok=True)
        output=directory/'submission_top_200.csv'
        pairs=request_count=shortlisted=0
        try:
            with output.open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['behaviour_id','article_ids','scores','similarity_scores','article_age_hours','eligible_article_count','ties_crossing_100','ties_crossing_200'])
                for request in cp.request_iter(prepared['dbs'][split],limit):
                    ranked,ids=rank_request(request,prepared['catalogue'],profiles,prepared['vectors'],[option],config.get('catalogue_batch_size',4096))
                    row=ranked[option['name']]
                    writer.writerow([request[0],*(json.dumps(row[key]) for key in ['article_ids','scores','similarity_scores','article_age_hours']),len(ids),row['ties_crossing_100'],row['ties_crossing_200']])
            # Reopen and recompute the selected full-catalogue shortlist, not the grid.
            with output.open(newline='') as stream:
                reader=csv.DictReader(stream)
                for request in cp.request_iter(prepared['dbs'][split],limit):
                    saved=next(reader,None)
                    if saved is None or saved['behaviour_id']!=request[0]:raise ValueError('Export request coverage mismatch')
                    ranked,ids=rank_request(request,prepared['catalogue'],profiles,prepared['vectors'],[option],config.get('catalogue_batch_size',4096))
                    row=ranked[option['name']]
                    if (int(saved['eligible_article_count'])!=len(ids) or
                        (saved['ties_crossing_100']=='True')!=row['ties_crossing_100'] or
                        (saved['ties_crossing_200']=='True')!=row['ties_crossing_200']):
                        raise ValueError('Export pool/tie metadata mismatch')
                    if json.loads(saved['article_ids'])!=row['article_ids']:raise ValueError('Top-200 order/membership mismatch')
                    for key in ['scores','similarity_scores','article_age_hours']:
                        actual=np.asarray(json.loads(saved[key]),dtype='float64')
                        if actual.shape!=(len(row['article_ids']),) or not np.isfinite(actual).all() or not np.allclose(actual,row[key],rtol=0,atol=2e-6):
                            raise ValueError(f'Export {key} mismatch')
                    pairs+=len(ids);request_count+=1;shortlisted+=len(row['article_ids'])
                if next(reader,None) is not None:raise ValueError('Unexpected extra export request')
            if pairs!=prepared['budgets'][split]['eligible_pairs']:raise ValueError('Export pair count differs from audit')
            checks[split]={'requests_verified':request_count,'shortlisted_articles_verified':shortlisted,
                'catalogue_pairs_rescored_for_verification':pairs,'full_score_rows_exported':0}
            request_counts[split]=request_count
        finally:profiles.close()
    profiles,_=cp.profiles_for(prepared,'validation')
    try:
        exported=evaluate_saved_validation(prepared,option,profiles)
    finally:profiles.close()
    record={'pipeline_version':VERSION,'config':config,'selected_variant':option,'grid_path':'freshness_grid.json',
        'inputs':prepared['audit']['inputs'],'eligibility_policy':cp.POLICY,'prediction_pool':'time_eligible_catalogue',
        'shortlist_size':200,'evaluation_cutoff':EVALUATION_K,'final_display_size':5,'score_column':'scores',
        'score_semantics':grid['formula'],'similarity_column':'similarity_scores','probability_column':None,
        'age_reference':grid['age_reference'],'history_model':cp.model_spec(config),'history_coverage':history,
        'learned_parameters':False,'embedding_fine_tuned':False,'test_metrics':None,
        'validation':grid['selected']['metrics'],'exported_validation':exported,'export_checks':checks,
        'all_validation_requests_exported':config['export_requests']['validation'] is None,
        'exports':{s:f'rankings/{s}/submission_top_200.csv' for s in cp.SPLITS},
        'limitations':['Validation repeatedly used for development; no untouched holdout claim.',
            'Soft freshness changes retrieval scores, not catalogue eligibility.',
            'Age-weight selection is not encoder improvement or evidence of retention gains.',
            'Raw cosine zero fallback receives the same age penalty as other candidates.',
            'Stage 2 business reranking is separate; no AUC or test outcome metric computed here.']}
    cp.save_json(work/'run_config.json',record)
    cp.save_json(work/'metrics.json',{'validation':record['validation'],'exported_validation':exported,'test':None})
    shutil.copy2(cp.__file__,work/'catalogue_pipeline.py');shutil.copy2(__file__,work/'freshness_pipeline.py')
    (work/'requirements.txt').write_text(''.join(f'{name}=={version(name)}\n' for name in ['numpy','pandas','pyarrow','scikit-learn']))
    result={'signature':signature,'selected_variant':option,'checks':checks,'exported_validation':exported,
        'estimated_output_bytes':estimate,'requests_exported':request_counts}
    cp.save_json(result_path,result)
    return result


def persist(prepared):
    root=prepared['config'].get('persistent_root')
    if not root:raise ValueError('Set persistent_root before copying')
    work=prepared['work'];target=Path(root)/prepared['config']['experiment_id']/prepared['config']['run_id']
    if target.exists():raise FileExistsError('Persistent destination exists; choose a fresh copy suffix')
    if target.resolve()==work.resolve() or work.resolve() in target.resolve().parents:raise ValueError('Copy destination must be outside local work')
    if not (work/'freshness_export.json').is_file():raise ValueError('Finish the selected export first')
    shutil.copytree(work,target,ignore=shutil.ignore_patterns('indexes','*.tmp'))
    copied=0
    for path in work.rglob('*'):
        if path.is_file() and 'indexes' not in path.relative_to(work).parts:
            other=target/path.relative_to(work)
            if not other.is_file() or other.stat().st_size!=path.stat().st_size:raise RuntimeError('Incomplete persistent copy')
            copied+=1
    cp.save_json(work/'persistence.json',{'location':str(target),'copied_files_verified_by_size':copied})
    shutil.copy2(work/'persistence.json',target/'persistence.json')
    return target
