"""Stage 1: frozen shared text embeddings -> user cosine similarity -> top 200.

No classifier, learned scorer, fitted preprocessing, probability, or business
reranking. Shared recorded rows supply requests and observed-click diagnostics.
"""
from __future__ import annotations
import csv
from contextlib import closing, ExitStack
from functools import lru_cache
import hashlib
import heapq
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

SPLITS=('train','validation','test')
KEYS=['user_id','impression_id','candidate_article_id','impression_time']
PIPELINE_VERSION='similarity_individual_history_v6'
POLICY={'name':'published_at_or_before_request_v1',
        'publication_field':'published_time','missing_publication':'exclude',
        'age_limit_hours':None,'access_filter':None,
        'snapshot':'ordered union of declared catalogues; first snapshot wins'}

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+'\n')
    tmp.replace(path)

def plain(value):
    if value is None or pd.isna(value): return None
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)): return bool(value)
    return value

def identity(user, impression):
    # Reject float IDs rather than losing precision through int(float).
    if not isinstance(user, (int, np.integer)) or not isinstance(impression, (int, np.integer)):
        raise ValueError('Original user/impression identifiers must be integers')
    return f'{int(user)}:{int(impression)}'

def stamp(value):
    value = pd.Timestamp(value)
    if pd.isna(value): raise ValueError('Missing request timestamp')
    if value.tzinfo is not None: raise ValueError('This implementation expects a single declared timezone-naive timestamp convention')
    return value.isoformat()

def fingerprint(path, content_hash=False):
    path=Path(path); stat=path.stat()
    result={'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if path.suffix == '.parquet':
        f=pq.ParquetFile(path)
        result.update(rows=f.metadata.num_rows, schema=str(f.schema_arrow))
    if content_hash:
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2), b''): digest.update(chunk)
        result['sha256']=digest.hexdigest()
    return result

def load_embeddings(paths, column='text_embedding', expected_dimension=None):
    vectors={}; counts=[]; dim=expected_dimension; seen_ids=set()
    for path in paths:
        schema=pq.ParquetFile(path).schema_arrow.names
        required=['article_id',column]
        if not set(required)<=set(schema): raise ValueError(f'{path} lacks {required}')
        flag='has_text_embedding' if 'has_text_embedding' in schema else None
        table=pd.read_parquet(path,columns=required+([flag] if flag else []))
        if table.article_id.isna().any() or table.article_id.duplicated().any():
            raise ValueError('Embedding IDs must be non-null and unique within each source')
        missing=0; overlaps=0
        for record in table.to_dict('records'):
            aid=int(record['article_id']); values=record[column]
            overlapping=aid in seen_ids;seen_ids.add(aid)
            absent=values is None or (isinstance(values,float) and np.isnan(values))
            present=not absent
            if flag and (record[flag] is None or bool(record[flag]) != present):
                raise ValueError(f'Embedding flag/vector disagreement for {aid}')
            if absent:
                missing+=1;overlaps+=int(overlapping);continue
            if not isinstance(values,(list,tuple,np.ndarray)):
                raise ValueError(f'Unreadable vector for {aid}; do not treat a file decoding failure as missingness')
            vector=np.asarray(values,dtype='float32')
            if vector.ndim!=1 or not len(vector) or not np.isfinite(vector).all():
                raise ValueError(f'Invalid vector for {aid}')
            if dim is None: dim=len(vector)
            norm=np.linalg.norm(vector)
            if len(vector)!=dim or not np.isfinite(norm) or norm==0: raise ValueError(f'Embedding dimension/norm mismatch for {aid}')
            if overlapping: overlaps+=1; continue
            vectors[aid]=vector/norm
        counts.append({'path':str(path),'rows':len(table),'missing_vectors':missing,'overlap_first_wins':overlaps})
    if not vectors: raise ValueError('No decoded usable vectors')
    return vectors, {'dimension':dim,'sources':counts,'usable_union_vectors':len(vectors)}

def request_iter(db_path, limit=None):
    with closing(sqlite3.connect(db_path)) as db:
        sql='SELECT behaviour_id,user_id,impression_id,time,context FROM requests'
        if limit is not None:
            if int(limit)<1: raise ValueError('Request limits must be positive or None for all')
            sql=f'SELECT * FROM ({sql} ORDER BY priority,behaviour_id LIMIT {int(limit)})'
        sql+=' ORDER BY time,user_id,impression_id'
        yield from db.execute(sql)

def requests(db_path, limit):
    if limit is None: raise ValueError('Use request_iter for full splits')
    return list(request_iter(db_path,limit))

class HistoryProfiles(dict):
    """History vectors and their aggregation rule, shared by CPU and GPU inference."""
    def __init__(self,*args,aggregation='mean_profile',top_k=3,**kwargs):
        super().__init__(*args,**kwargs)
        self.aggregation=aggregation;self.top_k=top_k

class ProfileLookup:
    """Bounded memory lookup for persistent, fixed history profiles."""
    def __init__(self,path,dimension,aggregation='mean_profile',top_k=3):
        self.path=str(path);self.dimension=dimension
        self.aggregation=aggregation;self.top_k=top_k
        self.connection=sqlite3.connect(self.path)
        self.get=lru_cache(maxsize=1024)(self._get)
    def _get(self,user,default=(None,0.0)):
        row=self.connection.execute('SELECT vector,coverage FROM profiles WHERE user_id=?',(int(user),)).fetchone()
        if row is None:return default
        profile=np.frombuffer(row[0],dtype='float32') if row[0] is not None else None
        if profile is not None and self.aggregation!='mean_profile':profile=profile.reshape(-1,self.dimension)
        return profile,float(row[1])
    def close(self):self.get.cache_clear();self.connection.close()

class SimilarityVectors(dict):
    """CPU-compatible lookups with one optional shared CUDA article table.

    No CUDA tensors enter the saved inference specification. CPU replay remains valid.
    Only one last-used user profile/history matrix is cached on the GPU.
    """
    def __init__(self, vectors, device='auto', min_gpu_batch=256):
        super().__init__(vectors)
        if device not in ('auto','cpu','cuda'):
            raise ValueError('similarity_device must be auto, cpu, or cuda')
        if min_gpu_batch<1:
            raise ValueError('similarity_min_gpu_batch must be positive')
        self.min_gpu_batch=int(min_gpu_batch)
        self._matrix=None;self._torch=None;self._article_index={}
        self._last_profile_source=None;self._last_profile_tensor=None
        self.info={'requested_device':device,'similarity_device':'cpu',
                   'classifier_device':'not_applicable','loading_and_exports_device':'cpu',
                   'minimum_gpu_batch':self.min_gpu_batch,'reason':'CPU explicitly selected'}
        if device=='cpu':return
        try:
            import torch
        except ImportError:
            if device=='cuda':
                raise RuntimeError('CUDA similarity requires PyTorch. Install torch and enable a GPU runtime.') from None
            self.info['reason']='PyTorch unavailable; using CPU'
            return
        if not torch.cuda.is_available():
            if device=='cuda':
                raise RuntimeError('CUDA requested but unavailable. Enable a Colab GPU and reconnect the runtime.')
            self.info['reason']='CUDA unavailable; using CPU'
            return
        self._torch=torch
        ids=list(self)
        dimension=len(next(iter(self.values()))) if self else 0
        self.info['article_table_bytes']=(len(ids)+1)*dimension*4
        try:
            # Row zero is the documented missing-vector placeholder, with flags.
            self._matrix=torch.empty((len(ids)+1,dimension),dtype=torch.float32,device='cuda')
            self._matrix[0].zero_()
            self._article_index={int(a):i+1 for i,a in enumerate(ids)}
            for start in range(0,len(ids),4096):
                block=np.stack([self[a] for a in ids[start:start+4096]])
                self._matrix[start+1:start+1+len(block)].copy_(torch.from_numpy(block))
        except torch.cuda.OutOfMemoryError:
            self._matrix=None;self._article_index={}
            if device=='cuda':
                raise RuntimeError('GPU memory is insufficient for the shared article table; choose similarity_device="cpu".') from None
            self.info['reason']='Insufficient CUDA memory; using CPU without excluding articles'
            return
        self.info.update(similarity_device='cuda',reason='Shared CUDA article table ready',
                         torch_version=torch.__version__,cuda_version=torch.version.cuda,
                         gpu_name=torch.cuda.get_device_name(0))

    def similarities(self, ids, profile, aggregation='mean_profile', top_k=3):
        if self._matrix is None or len(ids)<self.min_gpu_batch:
            return None
        torch=self._torch
        indices=np.fromiter((self._article_index.get(int(a),0) for a in ids),
                            dtype=np.int64,count=len(ids))
        with torch.no_grad():
            if self._last_profile_source is not profile:
                self._last_profile_tensor=torch.as_tensor(np.array(profile,copy=True),
                    dtype=torch.float32,device=self._matrix.device)
                self._last_profile_source=profile
            positions=torch.as_tensor(indices,dtype=torch.long,device=self._matrix.device)
            candidates=self._matrix.index_select(0,positions)
            if aggregation=='mean_profile':
                result=candidates @ self._last_profile_tensor
            else:
                pairwise=candidates @ self._last_profile_tensor.T
                result=(pairwise.max(dim=1).values if aggregation=='max_history' else
                        pairwise.topk(min(top_k,profile.shape[0]),dim=1).values.mean(dim=1))
            return result.clamp(-1,1).cpu().numpy()

def semantic_values(ids, user, profiles, vectors):
    profile,coverage=profiles.get(user,(None,0.0))
    result=np.zeros((len(ids),4),dtype='float32')
    missing=np.fromiter((int(a) not in vectors for a in ids),dtype=bool,count=len(ids))
    result[:,1]=missing;result[:,2]=profile is None;result[:,3]=coverage
    if profile is None or not len(ids):return result
    aggregation=getattr(profiles,'aggregation','mean_profile');top_k=getattr(profiles,'top_k',3)
    accelerated=vectors.similarities(ids,profile,aggregation,top_k) if isinstance(vectors,SimilarityVectors) else None
    if accelerated is not None:
        result[:,0]=accelerated
    else:
        positions=[i for i,a in enumerate(ids) if int(a) in vectors]
        if positions:
            candidates=np.stack([vectors[int(ids[i])] for i in positions])
            if aggregation=='mean_profile':scores=candidates @ profile
            else:
                pairwise=candidates @ profile.T
                k=min(top_k,profile.shape[0])
                scores=(pairwise.max(axis=1) if aggregation=='max_history' else
                        np.partition(pairwise,pairwise.shape[1]-k,axis=1)[:,-k:].mean(axis=1))
            result[positions,0]=scores
        result[:,0]=np.clip(result[:,0],-1,1)
    return result

def eligible_ids(catalogue, when):
    mask=catalogue.published_time.notna() & catalogue.published_time.le(pd.Timestamp(when))
    return catalogue.index[mask].to_numpy(dtype='int64')

def retrieval_metrics(ranking, known):
    if not known: return None
    hits=[int(a in known) for a in ranking]
    recall=sum(hits)/len(known)
    ideal=sum(1/math.log2(i+2) for i in range(min(200,len(known))))
    dcg=sum(v/math.log2(i+2) for i,v in enumerate(hits))
    return recall,dcg/ideal

def evaluate_catalogue(model, reqs, db_path, catalogue, profiles, vectors, batch_size):
    values=[]; eligible_recalls=[]; without=empty=excluded=total_known=ties=0
    exclusion_reasons={'missing_article':0,'missing_publication':0,'future_publication':0}
    with closing(sqlite3.connect(db_path)) as db:
        for req in reqs:
            known={r[0] for r in db.execute('SELECT article_id FROM candidates WHERE behaviour_id=? AND clicked=1',(req[0],))}
            stats={}
            ranking,count=score_request(model,req,catalogue,profiles,vectors,batch_size,stats=stats)
            ties+=stats['ties_crossing_200']
            empty+=count==0
            if not known: without+=1; continue
            metric=retrieval_metrics(ranking,known);values.append(metric)
            elig=set(eligible_ids(catalogue,req[3]));excluded+=len(known-elig);total_known+=len(known)
            if known&elig:eligible_recalls.append(len(set(ranking)&known)/len(known&elig))
            for aid in known-elig:
                reason='missing_article' if aid not in catalogue.index else ('missing_publication' if pd.isna(catalogue.loc[aid,'published_time']) else 'future_publication')
                exclusion_reasons[reason]+=1
    if not values: raise ValueError('Selection sample has no requests with known clicks')
    return {'known_click_recall@200':float(np.mean([v[0] for v in values])),
            'observed_click_ndcg@200':float(np.mean([v[1] for v in values])),
            'evaluated_requests':len(values),'requests_without_known_clicks':without,
            'empty_eligible_pools':empty,'known_clicks_excluded_by_eligibility':excluded,
            'known_clicks':total_known,'eligible_only_recall@200':float(np.mean(eligible_recalls)) if eligible_recalls else None,
            'known_click_exclusion_reasons':exclusion_reasons,'requests_with_ties_crossing_200':ties,'known_click_eligibility_coverage':1-excluded/max(total_known,1),
            'ties':'descending score, ascending original article ID; exact top-200 membership',
            'outcomes':'unshown outcomes unknown; zero observed gain is not a negative label'}

def budget(reqs, catalogue, bytes_per_pair=85):
    published=np.sort(catalogue.published_time.dropna().to_numpy(dtype='datetime64[ns]'))
    counts=[int(np.searchsorted(published,np.datetime64(pd.Timestamp(r[3]),'ns'),side='right')) for r in reqs]
    pairs=sum(counts)
    return {'requests':len(counts),'eligible_pairs':pairs,'empty_pools':sum(c==0 for c in counts),
            'shortlisted_articles':sum(min(200,c) for c in counts),
            'estimated_csv_bytes':sum(100+56*min(200,c) for c in counts),
            'csv_estimate_note':'Top-200 IDs and aligned similarity scores plus per-request eligibility audit; no full-score CSV.',
            'catalogue_articles':len(catalogue),'missing_publication_excluded':int(catalogue.published_time.isna().sum()),
            'runtime_seconds':None,'runtime_note':'Measure a bounded sample; no reliable timing estimate before scoring.'}

def evaluate_exported_validation(out, reqs, db_path, catalogue):
    """Evaluate already exported shortlists, without rescoring the catalogue."""
    values=[];eligible_values=[];without=empty=excluded=total_known=ties=0
    reasons={'missing_article':0,'missing_publication':0,'future_publication':0}
    with closing(sqlite3.connect(db_path)) as db, \
         closing(shortlist_parquet_rows(out/'rankings/validation/submission_top_200.parquet')) as rankings, \
         (out/'eligibility/validation_requests.csv').open(newline='') as ef:
        eligibility=iter(csv.DictReader(ef))
        for req in reqs:
            row=next(rankings);detail=next(eligibility)
            if row['behaviour_id']!=req[0] or detail['behaviour_id']!=req[0]:raise ValueError('Export metric request alignment failure')
            ranking=row['article_ids'];empty+=int(detail['eligible_article_count'])==0
            ties+=detail['ties_crossing_200']=='True'
            known={r[0] for r in db.execute('SELECT article_id FROM candidates WHERE behaviour_id=? AND clicked=1',(req[0],))}
            if not known:without+=1;continue
            values.append(retrieval_metrics(ranking,known));eligible_known=set()
            for aid in known:
                if aid not in catalogue.index:reasons['missing_article']+=1
                elif pd.isna(catalogue.loc[aid,'published_time']):reasons['missing_publication']+=1
                elif catalogue.loc[aid,'published_time']>pd.Timestamp(req[3]):reasons['future_publication']+=1
                else:eligible_known.add(aid)
            excluded+=len(known-eligible_known);total_known+=len(known)
            if eligible_known:eligible_values.append(len(set(ranking)&known)/len(eligible_known))
    return {'known_click_recall@200':float(np.mean([v[0] for v in values])) if values else None,
            'observed_click_ndcg@200':float(np.mean([v[1] for v in values])) if values else None,
            'evaluated_requests':len(values),'requests_without_known_clicks':without,
            'empty_eligible_pools':empty,'known_clicks_excluded_by_eligibility':excluded,
            'known_clicks':total_known,'known_click_eligibility_coverage':1-excluded/max(total_known,1),
            'eligible_only_recall@200':float(np.mean(eligible_values)) if eligible_values else None,
            'known_click_exclusion_reasons':reasons,'requests_with_ties_crossing_200':ties,
            'source':'validated exported catalogue shortlists; original known-click denominators'}

def check_cost(prepared):
    config=prepared['config'];planned=prepared['budgets']
    pairs=sum(v['eligible_pairs'] for v in planned.values())
    size=sum(v['estimated_csv_bytes'] for v in planned.values())
    free=shutil.disk_usage(prepared['work']).free
    if pairs>config.get('max_export_pairs',10_000_000):
        raise RuntimeError(f'Planned export is {pairs:,} pairs, beyond max_export_pairs. Inspect input_audit.json; use an explicitly sampled development scope or raise the declared budget after checking feasibility.')
    if size>config.get('max_export_gb',10)*1e9 or 2*size>free:
        raise RuntimeError('Estimated export exceeds declared disk budget or lacks space for export/persistence copies')
    return {'pairs':pairs,'estimated_csv_bytes':size,'local_free_bytes':free}


def resolve_inputs(config):
    root=Path(config['drive_root'])
    def resolve(name,choices):
        override=config.get('paths',{}).get(name)
        if override:
            p=Path(override);p=p if p.is_absolute() else root/p
            if not p.is_file():raise FileNotFoundError(p)
            return p
        found=list(dict.fromkeys((root/p).resolve() for p in choices if (root/p).is_file()))
        if len(found)!=1:
            raise FileNotFoundError(f'{name}: expected exactly one input, found {found}. Set CONFIG["paths"]["{name}"]. Tried: {choices}')
        return found[0]
    inputs={}
    for part in ['train_val','test']:
        inputs[f'catalogue_{part}']=resolve(f'catalogue_{part}',[f'{part}/articles.parquet'])
        inputs[f'embeddings_{part}']=resolve(f'embeddings_{part}',[
            f'{part}/article embeddings/text_embeddings.parquet',f'embeddings/{part}/text_embeddings.parquet'])
    for split in SPLITS:
        inputs[f'features_{split}']=resolve(f'features_{split}',[f'feature_engineering/{split}_features.parquet'])
        prefix=f'train_val/{split}' if split!='test' else 'test'
        inputs[f'history_{split}']=resolve(f'history_{split}',[f'{prefix}/history.parquet'])
    return inputs


def load_catalogue(paths):
    parts=[]
    for path in paths:
        frame=pd.read_parquet(path,columns=['article_id','published_time'])
        if frame.article_id.isna().any() or frame.article_id.duplicated().any():
            raise ValueError(f'Null/duplicate article IDs in {path}')
        if not pd.api.types.is_integer_dtype(frame.article_id.dtype):raise ValueError('Article IDs must be original integers')
        frame['published_time']=pd.to_datetime(frame.published_time)
        parts.append(frame)
    combined=pd.concat(parts,ignore_index=True)
    overlaps=int(combined.article_id.duplicated().sum())
    catalogue=combined.drop_duplicates('article_id',keep='first').sort_values('article_id').set_index('article_id',drop=False)
    return catalogue,overlaps


def index_shared(path,split,db_path,seed):
    """Index existing pairs without reading engineered predictors or re-exploding."""
    with closing(sqlite3.connect(db_path)) as db:
        db.execute('PRAGMA cache_size=-32768')
        db.executescript('''
          CREATE TABLE IF NOT EXISTS requests (
            behaviour_id TEXT PRIMARY KEY,user_id INTEGER,impression_id INTEGER,
            time TEXT NOT NULL,context TEXT NOT NULL,priority TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS candidates (
            behaviour_id TEXT,article_id INTEGER,clicked INTEGER,
            PRIMARY KEY(behaviour_id,article_id));
        ''')
        if db.execute('SELECT count(*) FROM candidates').fetchone()[0]:raise FileExistsError('Index already populated')
        available=set(pq.ParquetFile(path).schema_arrow.names)
        required=KEYS+['split']
        # Test outcomes never enter the inference index, even if supplied in a new release.
        if split!='test':required+=['clicked']
        if not set(required)<=available:raise ValueError(f'{path} lacks {sorted(set(required)-available)}')
        rows=0
        for batch in pq.ParquetFile(path).iter_batches(columns=required,batch_size=8192):
            unique_requests={};candidates=[]
            for row in batch.to_pylist():
                if row['split']!=split:raise ValueError('Split column disagrees with source file')
                key=identity(row['user_id'],row['impression_id'])
                if not isinstance(row['candidate_article_id'],int):raise ValueError('Article IDs must be integers')
                label=None if split=='test' else row['clicked']
                if split!='test' and label not in (0,1):raise ValueError('Genuine binary click labels required')
                when=stamp(row['impression_time'])
                item=(key,int(row['user_id']),int(row['impression_id']),when,'[]',
                      hashlib.blake2b(f'{seed}:{split}:{key}'.encode(),digest_size=8).hexdigest())
                if key in unique_requests and item!=unique_requests[key]:raise ValueError('Request timestamp varies within slate')
                unique_requests[key]=item
                candidates.append((key,int(row['candidate_article_id']),label))
            for item in unique_requests.values():
                old=db.execute('SELECT time FROM requests WHERE behaviour_id=?',(item[0],)).fetchone()
                if old and old[0]!=item[3]:raise ValueError('Request timestamp changes across batches')
                db.execute('INSERT OR IGNORE INTO requests VALUES (?,?,?,?,?,?)',item)
            try:db.executemany('INSERT INTO candidates VALUES (?,?,?)',candidates)
            except sqlite3.IntegrityError as error:
                raise ValueError('Duplicate source candidate keys; investigate rather than deduplicating') from error
            db.commit();rows+=len(candidates)
        if not rows:raise ValueError('No recorded candidate rows')
        count,start=db.execute('SELECT count(*),min(time) FROM requests').fetchone()
        db.execute('CREATE INDEX request_sampling ON requests(priority,behaviour_id)')
        db.execute('CREATE INDEX request_time ON requests(time,user_id,impression_id)')
        db.commit()
    return {'candidate_rows':rows,'requests':count,'split_start':start,
            'source_has_clicked_column':'clicked' in available,
            'outcome_policy':'ignored_for_test_inference' if split=='test' else 'observed_labels_for_evaluation',
            'outcome_column_loaded':split!='test',
            'sampling_rule':'Smallest BLAKE2b-64(seed:split:behaviour_id); complete recorded slates'}


def history_profiles(path,needed_users,cutoff,vectors,output_db=None,history_limit=None,aggregation='mean_profile',top_k=3):
    validate_aggregation(aggregation,top_k,history_limit)
    if history_limit is not None and (not isinstance(history_limit,int) or history_limit<1):
        raise ValueError('history_limit must be a positive integer or None')
    with ExitStack() as stack:
        store=stack.enter_context(closing(sqlite3.connect(output_db))) if output_db is not None else None
        if store is not None:
            store.execute('CREATE TABLE IF NOT EXISTS profiles (user_id INTEGER PRIMARY KEY,vector BLOB,coverage REAL)')
            if store.execute('SELECT count(*) FROM profiles').fetchone()[0]:raise FileExistsError('History lookup already exists')
        profiles=HistoryProfiles(aggregation=aggregation,top_k=top_k);seen=set();missing_time=selected_events=available_events=0
        dim=len(next(iter(vectors.values())))
        for batch in pq.ParquetFile(path).iter_batches(columns=['user_id','article_id_fixed','impression_time_fixed'],batch_size=64):
            for row in batch.to_pylist():
                uid=int(row['user_id'])
                if uid not in needed_users:continue
                if uid in seen:raise ValueError('Duplicate history user')
                seen.add(uid);ids,times=row['article_id_fixed'],row['impression_time_fixed']
                if ids is None or times is None or len(ids)!=len(times):raise ValueError('Misaligned history arrays')
                valid=[]
                for position,(aid,when) in enumerate(zip(ids,times)):
                    if when is None:missing_time+=1;continue
                    if pd.Timestamp(when)>=pd.Timestamp(cutoff):raise ValueError('History reaches the behavior split')
                    if aid is None:raise ValueError('Missing historical article ID')
                    valid.append((pd.Timestamp(when),position,int(aid)))
                if history_limit is not None:valid=heapq.nlargest(history_limit,valid)
                history=[vectors[aid] for _,_,aid in valid if aid in vectors]
                available=len(history)
                if aggregation=='mean_profile':
                    summed=np.asarray(history,dtype='float64').sum(axis=0) if history else np.zeros(dim)
                    norm=np.linalg.norm(summed)
                    profile=(summed/norm).astype('float32') if norm>1e-8 else None
                else:
                    profile=np.stack(history).astype('float32') if history else None
                coverage=available/max(len(valid),1)
                selected_events+=len(valid);available_events+=available
                if store is None:profiles[uid]=(profile,coverage)
                else:store.execute('INSERT INTO profiles VALUES (?,?,?)',(uid,profile.tobytes() if profile is not None else None,coverage))
            if store is not None:store.commit()
    if output_db is not None:profiles=ProfileLookup(output_db,dim,aggregation,top_k)
    return profiles,{'users_requested':len(needed_users),'users_found':len(seen),
        'missing_history_users':len(needed_users-seen),'events_missing_time_excluded':missing_time,
        'selected_history_events':selected_events,'selected_events_with_vectors':available_events,
        'history_limit':history_limit,'history_aggregation':aggregation,'history_top_k':top_k if aggregation=='topk_history' else None}


def cosine_scores(ids,user,profiles,vectors):
    """Direct cosine relevance. Flags/coverage are diagnostics, not predictors."""
    return semantic_values(ids,user,profiles,vectors)[:,0]


def score_request(model,request,catalogue,profiles,vectors,batch_size,write=None,stats=None):
    ids=eligible_ids(catalogue,request[3]);top=[]
    for start in range(0,len(ids),batch_size):
        part=ids[start:start+batch_size]
        scores=cosine_scores(part,int(request[1]),profiles,vectors)
        if not np.isfinite(scores).all():raise ValueError('Nonfinite cosine output')
        if write:write.writerows((request[0],int(a),float(s)) for a,s in zip(part,scores))
        order=np.lexsort((part,-scores))[:201]
        local=[(float(scores[i]),int(part[i])) for i in order]
        top=sorted(top+local,key=lambda z:(-z[0],z[1]))[:201]
    if stats is not None:
        stats['ties_crossing_200']=len(top)>200 and top[199][0]==top[200][0]
        stats['similarity_scores']=[score for score,_ in top[:200]]
    return [a for _,a in top[:200]],len(ids)


def validate_aggregation(aggregation,top_k,history_limit):
    if aggregation not in ('mean_profile','max_history','topk_history'):
        raise ValueError('history_aggregation must be mean_profile, max_history, or topk_history')
    if not isinstance(top_k,int) or isinstance(top_k,bool) or top_k<1:
        raise ValueError('history_top_k must be a positive integer')
    if aggregation!='mean_profile' and (not isinstance(history_limit,int) or isinstance(history_limit,bool) or history_limit<1):
        raise ValueError('Individual-history methods require a positive bounded history_limit, e.g. 10')


def model_spec(config):
    aggregation=config.get('history_aggregation','mean_profile');top_k=config.get('history_top_k',3)
    validate_aggregation(aggregation,top_k,config.get('history_limit'))
    descriptions={'mean_profile':'L2-normalized mean of selected past-click article vectors',
                  'max_history':'maximum cosine over available selected past-click article vectors',
                  'topk_history':'mean of largest min(history_top_k, available history events) candidate-to-history cosines'}
    return {'pipeline_version':PIPELINE_VERSION,'method':aggregation+'_cosine',
            'history_aggregation':aggregation,'history_top_k':top_k if aggregation=='topk_history' else None,
            'history_limit':config.get('history_limit'),
            'article_normalization':'L2 per nonmissing article vector',
            'user_profile':descriptions[aggregation],
            'history_snapshot':'strictly before first request of each split; no behavior-log updates',
            'repeated_clicks':'retain repeated weight/events, including in top-k aggregation',
            'missing_vectors_or_history':'score 0; retain eligible articles; report flags and ties',
            'score_semantics':'aggregated cosine similarity in [-1,1], not a probability',
            'learned_parameters':False,'fine_tuned_encoder':False,
            'shortlist_size':200,'tie_policy':'descending cosine, ascending original article ID'}


def prepare(config,work):
    if config.get('scope','development') not in ('development','final'):raise ValueError('Scope must be development or final')
    if config.get('history_limit') is not None and (not isinstance(config['history_limit'],int) or config['history_limit']<1):
        raise ValueError('history_limit must be a positive integer or None')
    model_spec(config)  # Validate aggregation before creating caches.
    work=Path(work);(work/'indexes').mkdir(parents=True,exist_ok=True)
    prior_path=work/'input_audit.json'
    if prior_path.exists() and json.loads(prior_path.read_text()).get('pipeline_version')!=PIPELINE_VERSION:
        raise ValueError('This folder contains hybrid/older artifacts. Use a fresh similarity-only run ID.')
    inputs=resolve_inputs(config)
    catalogue,overlap=load_catalogue([inputs['catalogue_train_val'],inputs['catalogue_test']])
    vectors,info=load_embeddings([inputs['embeddings_train_val'],inputs['embeddings_test']],
        config.get('embedding_column','text_embedding'),config.get('embedding_dimension'))
    vectors=SimilarityVectors(vectors,config.get('similarity_device','auto'),config.get('similarity_min_gpu_batch',256))
    print('Similarity-only compute:',vectors.info,flush=True)
    fingerprints={name:fingerprint(path,config.get('hash_inputs',False)) for name,path in inputs.items()}
    estimate=sum(fingerprints[f'features_{s}']['rows'] for s in SPLITS)*320
    if estimate>config.get('max_index_gb',20)*1e9 or estimate>shutil.disk_usage(work).free*0.7:
        save_json(work/'index_budget.json',{'estimated_index_bytes':estimate})
        raise RuntimeError('Generated disk-index estimate exceeds budget; see index_budget.json')
    if prior_path.exists():
        prior=json.loads(prior_path.read_text())
        if prior['inputs']!=fingerprints or prior['seed']!=config['seed'] or prior['model']!=model_spec(config):
            raise ValueError('Sources, seed or similarity configuration changed; use a new run ID')
    dbs={};counts={}
    for split in SPLITS:
        dbs[split]=work/f'indexes/{split}.sqlite';state=work/f'indexes/{split}.json'
        if state.exists():
            old=json.loads(state.read_text())
            if old['input']!=fingerprints[f'features_{split}']:raise ValueError('Indexed source changed')
            counts[split]=old['counts']
        else:
            if dbs[split].exists():dbs[split].unlink()
            counts[split]=index_shared(inputs[f'features_{split}'],split,dbs[split],config['seed'])
            save_json(state,{'input':fingerprints[f'features_{split}'],'counts':counts[split]})
    with closing(sqlite3.connect(dbs['train'])) as db:train_max=db.execute('SELECT max(time) FROM requests').fetchone()[0]
    if train_max>=counts['validation']['split_start']:raise ValueError('Training/validation periods overlap')
    limits=config.get('export_requests',{'train':25,'validation':25,'test':25})
    if set(limits)!=set(SPLITS):raise ValueError('Declare export request limits for all three splits')
    if config.get('scope')=='final':
        if any(limits[s] is not None for s in SPLITS):raise ValueError('Final scope requires all requests in each export split')
        if not config.get('owner') or not config.get('input_version'):raise ValueError('Record owner and shared input release for final handoff')
        if not isinstance(config.get('embedding_source'),dict) or not config['embedding_source'].get('model'):
            raise ValueError('Confirm exact shared embedding encoder identity for final handoff')
    budgets={s:budget(request_iter(dbs[s],limits[s]),catalogue) for s in SPLITS}
    full={s:budget(request_iter(dbs[s]),catalogue) for s in SPLITS}
    # Retain request metadata only; candidate rows remain streamed from SQLite.
    validation=list(request_iter(dbs['validation'],config.get('validation_requests',100)))
    audit={'pipeline_version':PIPELINE_VERSION,'model':model_spec(config),'seed':config['seed'],
        'inputs':fingerprints,'input_version':config.get('input_version'),
        'compute':vectors.info,'request_counts':counts,'catalogue_articles':len(catalogue),
        'catalogue_overlap_first_wins':overlap,'embedding_info':info,
        'catalogue_articles_with_vectors':len(set(catalogue.index)&set(vectors)),
        'catalogue_articles_without_vectors':len(set(catalogue.index)-set(vectors)),
        'planned_exports':budgets,'full_export_costs':full,'estimated_index_bytes':estimate,
        'eligibility_policy':POLICY,'validation_requests':[r[0] for r in validation],
        'feature_usage':'Only original IDs, timestamps and observed labels read from shared feature tables; no engineered predictors.',
        'hash_note':'SHA256 recorded' if config.get('hash_inputs') else 'Size/mtime/schema fingerprints, not full content hashes.'}
    save_json(prior_path,audit)
    return {'config':config,'work':work,'inputs':inputs,'catalogue':catalogue,'vectors':vectors,
        'dbs':dbs,'index_info':counts,'budgets':budgets,'selection_requests':validation,
        'validation_requests':validation,'audit':audit}


def profiles_for(prepared,split):
    config=prepared['config'];users=set()
    limits=[config['export_requests'][split]]
    if split=='validation':limits.append(config.get('validation_requests',100))
    for limit in limits:users.update(int(r[1]) for r in request_iter(prepared['dbs'][split],limit))
    path=prepared['work']/f'indexes/{split}_profiles.sqlite';state=path.with_suffix('.json')
    key={'users_sha256':hashlib.sha256(','.join(map(str,sorted(users))).encode()).hexdigest(),
         'history':prepared['audit']['inputs'][f'history_{split}'],
         'cutoff':prepared['index_info'][split]['split_start'],'model':model_spec(config)}
    if state.exists():
        old=json.loads(state.read_text())
        if old['key']!=key:raise ValueError('History lookup configuration changed; use a new run')
        return ProfileLookup(path,prepared['audit']['embedding_info']['dimension'],
            config.get('history_aggregation','mean_profile'),config.get('history_top_k',3)),old['counts']
    if path.exists():path.unlink()
    profiles,counts=history_profiles(prepared['inputs'][f'history_{split}'],users,key['cutoff'],prepared['vectors'],path,config.get('history_limit'),
        config.get('history_aggregation','mean_profile'),config.get('history_top_k',3))
    save_json(state,{'key':key,'counts':counts})
    return profiles,counts


def define_model(prepared):
    """Save a fixed inference specification. This does not train anything."""
    model=model_spec(prepared['config'])
    path=prepared['work']/'model/model_config.json'
    if path.exists() and json.loads(path.read_text())!=model:
        raise ValueError('Inference configuration already differs; use a new run ID')
    save_json(path,model)
    return model


def evaluate(prepared,model=None):
    model=define_model(prepared) if model is None else model
    if model!=model_spec(prepared['config']):raise ValueError('Model specification does not match prepared configuration')
    profiles,coverage=profiles_for(prepared,'validation')
    try:
        started=time.perf_counter()
        metrics=evaluate_catalogue(model,prepared['validation_requests'],prepared['dbs']['validation'],
            prepared['catalogue'],profiles,prepared['vectors'],prepared['config'].get('catalogue_batch_size',4096))
        elapsed=time.perf_counter()-started
        metrics['requests_without_usable_user_profile']=sum(profiles.get(int(r[1]),(None,0))[0] is None for r in prepared['validation_requests'])
        result={'metrics':metrics,'seconds':elapsed,
                'eligible_pairs':budget(prepared['validation_requests'],prepared['catalogue'])['eligible_pairs'],
                'history_coverage':coverage,'model':model,'fitted_parameters':False,
                'request_ids':[r[0] for r in prepared['validation_requests']],
                'compute':prepared['vectors'].info,
                'input_signature':hashlib.sha256(json.dumps(prepared['audit']['inputs'],sort_keys=True).encode()).hexdigest()}
        save_json(prepared['work']/'validation_metrics.json',result)
        return result
    finally:profiles.close()


def export_split(model,reqs,db_path,split,out,catalogue,profiles,vectors,batch_size):
    ranking_dir=out/f'rankings/{split}';eligibility=out/'eligibility'
    for path in [ranking_dir,eligibility]:path.mkdir(parents=True,exist_ok=True)
    request_count=recorded_count=outside=empty=missing_users=missing_vectors=0
    auc_sum=auc_count=0;pending=[]
    native_schema,score_schema=shortlist_schemas()
    with closing(sqlite3.connect(db_path)) as db, \
         pq.ParquetWriter(ranking_dir/'submission_top_200.parquet',native_schema,compression='zstd') as writer, \
         pq.ParquetWriter(ranking_dir/'submission_top_200_scores.parquet',score_schema,compression='zstd') as score_writer, \
         (eligibility/f'{split}_requests.csv').open('w',newline='') as ef:
        ew=csv.writer(ef)
        ew.writerow(['behaviour_id','impression_time','eligible_article_count',
            'recorded_candidates_outside_pool','ties_crossing_200','user_profile_missing','eligible_articles_missing_vector'])
        for req in reqs:
            request_count+=1;stats={}
            ranking,count=score_request(model,req,catalogue,profiles,vectors,batch_size,stats=stats)
            pending.append({'behaviour_id':req[0],'article_ids':ranking,'similarity_scores':stats['similarity_scores']})
            if len(pending)>=512:
                write_shortlist_batch(pending,writer,score_writer);pending.clear()
            rows=db.execute('SELECT article_id,clicked FROM candidates WHERE behaviour_id=? ORDER BY article_id',(req[0],)).fetchall()
            ids=np.asarray([r[0] for r in rows],dtype='int64')
            scores=cosine_scores(ids,int(req[1]),profiles,vectors)
            known_eligible=eligible_ids(catalogue,req[3]);pool=set(known_eligible)
            excluded=sum(a not in pool for a in ids);missing=sum(int(a) not in vectors for a in known_eligible)
            no_profile=profiles.get(int(req[1]),(None,0.0))[0] is None
            outside+=excluded;empty+=count==0;missing_users+=no_profile;missing_vectors+=missing;recorded_count+=len(rows)
            ew.writerow([req[0],req[3],count,excluded,stats['ties_crossing_200'],no_profile,missing])
            if split!='test':
                y=np.asarray([r[1] for r in rows],dtype='int8')
                if len(np.unique(y))==2:auc_sum+=float(roc_auc_score(y,scores));auc_count+=1
        if pending:write_shortlist_batch(pending,writer,score_writer)
    return {'parquet_rows_exported':request_count,
            'recorded_slate_mean_impression_auc':auc_sum/auc_count if auc_count else None,
            'auc_evaluable_requests':auc_count,'requests_exported':request_count,'recorded_rows':recorded_count,
            'recorded_candidates_outside_pool':outside,'empty_pools':empty,
            'requests_without_usable_user_profile':missing_users,
            'eligible_pairs_missing_article_vector':missing_vectors,
            'test_metrics_available':False if split=='test' else None}


def validate_export(out,split,reqs,catalogue,db_path,expected_pairs,*,model,profiles,vectors,batch_size=4096):
    """Check native Parquet IDs and scores against independently recomputed rankings."""
    ranking_path=out/f'rankings/{split}/submission_top_200.parquet'
    scores_path=out/f'rankings/{split}/submission_top_200_scores.parquet'
    pairs=request_count=shortlisted=0
    with closing(shortlist_parquet_rows(ranking_path)) as reader, closing(shortlist_score_rows(scores_path)) as score_reader:
        for req in reqs():
            row=next(reader,None);score_row=next(score_reader,None)
            if row is None or row['behaviour_id']!=req[0]:raise ValueError('Ranking request coverage mismatch')
            ranking=row['article_ids'];stats={}
            expected,count=score_request(model,req,catalogue,profiles,vectors,batch_size,stats=stats)
            if ranking!=expected:raise ValueError('Top-200 ordering or membership mismatch')
            if score_row is None or score_row['behaviour_id']!=req[0] or score_row['article_ids']!=ranking:
                raise ValueError('Parquet score request/order mismatch')
            scores=np.asarray(score_row['similarity_scores'],dtype='float64')
            if scores.ndim!=1 or len(scores)!=len(ranking) or not np.isfinite(scores).all():
                raise ValueError('Invalid shortlist similarity scores')
            if not np.allclose(scores,stats['similarity_scores'],rtol=0,atol=2e-6):raise ValueError('Shortlist similarity scores mismatch')
            request_count+=1;pairs+=count;shortlisted+=len(ranking)
        if next(reader,None) is not None or next(score_reader,None) is not None:raise ValueError('Unexpected extra export requests')
    if pairs!=expected_pairs:raise ValueError('Full catalogue pair count differs from audit')
    return {'catalogue_pairs_rescored_for_verification':pairs,'requests_verified':request_count,
            'shortlisted_articles_verified':shortlisted,'full_score_rows_exported':0}


def load_run(prepared,source_run):
    """Load a frozen similarity specification, never an old classifier checkpoint."""
    source=Path(source_run)
    record=json.loads((source/'run_config.json').read_text())
    if record.get('pipeline_version')!=PIPELINE_VERSION:raise ValueError('Source is not a similarity-only run; hybrid checkpoints are incompatible')
    if record['inputs']!=prepared['audit']['inputs'] or record['model']!=model_spec(prepared['config']):
        raise ValueError('Saved sources or inference configuration changed')
    model=json.loads((source/'model/model_config.json').read_text())
    if model!=model_spec(prepared['config']):raise ValueError('Saved model specification mismatch')
    define_model(prepared)
    save_json(prepared['work']/'source_run.json',{'source_run':str(source.resolve()),'learned_parameters':False})
    return model


def persist_and_reload(prepared,probe):
    config=prepared['config'];work=prepared['work'];root=config.get('persistent_root')
    if not root:return {'persistent_copy_complete':False,'reload_from_persistent_copy':False,
                        'reason':'Set persistent_root; copy the complete run preserving its structure.'}
    target=Path(root)/config['experiment_id']/config['run_id']
    if target.resolve()==work.resolve() or work.resolve() in target.resolve().parents:
        raise ValueError('Persistent copy must be outside the local run')
    if target.exists():raise FileExistsError(f'Persistent destination already exists: {target}. Use a new run ID.')
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(work,target,ignore=shutil.ignore_patterns('indexes','*.tmp'))
    copied=0
    for path in work.rglob('*'):
        if path.is_file() and 'indexes' not in path.relative_to(work).parts:
            other=target/path.relative_to(work)
            if not other.is_file() or other.stat().st_size!=path.stat().st_size:raise RuntimeError('Incomplete persistent copy')
            copied+=1
    saved=json.loads((target/'model/model_config.json').read_text())
    if saved!=model_spec(config):raise ValueError('Persistent model configuration mismatch')
    module_name='_saved_similarity_pipeline'
    spec=importlib.util.spec_from_file_location(module_name,target/'model/catalogue_pipeline.py')
    loaded=importlib.util.module_from_spec(spec);sys.modules[module_name]=loaded;spec.loader.exec_module(loaded)
    with np.load(target/'model/reload_probe.npz',allow_pickle=False) as data:
        profile=None if data['profile_missing'].item() else data['profile']
        profiles=loaded.HistoryProfiles({int(data['user_id']):(profile,float(data['coverage']))},
            aggregation=config.get('history_aggregation','mean_profile'),top_k=config.get('history_top_k',3))
        # Reopen versioned input files to verify preprocessing and lookup reconstruction,
        # rather than only reusing an in-memory vector dictionary.
        vectors,info=loaded.load_embeddings([prepared['inputs']['embeddings_train_val'],prepared['inputs']['embeddings_test']],
            config.get('embedding_column','text_embedding'),config.get('embedding_dimension'))
        actual=loaded.cosine_scores(data['article_ids'],int(data['user_id']),profiles,vectors)
        np.testing.assert_allclose(actual,data['scores'],rtol=2e-6,atol=2e-6)
    return {'persistent_copy_complete':True,'persistent_location':str(target),
            'copied_files_verified_by_size':copied,'reload_from_persistent_copy':True,
            'embedding_lookup_reopened':True,'learned_parameters':False,'rtol':2e-6,'atol':2e-6,
            'probe_rows':len(probe['article_ids'])}


def execute(prepared,model=None):
    """Validate and export one fixed similarity configuration. No fitting occurs."""
    config=prepared['config'];work=prepared['work']
    result_path=work/'result.json'
    if result_path.exists():
        old=json.loads((work/'run_config.json').read_text())
        canonical=json.loads(json.dumps(config,default=str))
        if old['config']!=canonical:raise ValueError('Completed run configuration differs; use a new run ID')
        result=json.loads(result_path.read_text());result['reused_completed_run']=True
        print('Reusing completed similarity-only run:',work,flush=True)
        return result
    cost=check_cost(prepared)
    model=define_model(prepared) if model is None else model
    if model!=model_spec(config):raise ValueError('Model configuration mismatch')
    benchmark_path=work/'validation_metrics.json'
    benchmark=json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
    signature=hashlib.sha256(json.dumps(prepared['audit']['inputs'],sort_keys=True).encode()).hexdigest()
    if (benchmark is None or benchmark.get('model')!=model or
        benchmark.get('request_ids')!=[r[0] for r in prepared['validation_requests']] or
        benchmark.get('compute')!=prepared['vectors'].info or benchmark.get('input_signature')!=signature):
        benchmark=evaluate(prepared,model)
    seconds=benchmark['seconds'];pairs=benchmark['eligible_pairs']
    cost.update(validation_benchmark_seconds=seconds,validation_benchmark_pairs=pairs,
        estimated_scoring_seconds_excluding_io=2*seconds*cost['pairs']/pairs if pairs else None,
        runtime_note='Includes export scoring and independent shortlist verification; excludes CSV I/O, indexing and persistence.')
    save_json(work/'export_budget.json',cost)
    if cost['estimated_scoring_seconds_excluding_io'] is not None and cost['estimated_scoring_seconds_excluding_io']>config.get('max_scoring_seconds',3600):
        raise RuntimeError('Measured catalogue scoring estimate exceeds runtime budget; inspect export_budget.json')
    checks={};diagnostics={};history_counts={}
    with ExitStack() as resources:
        profiles={}
        for split in SPLITS:
            profiles[split],history_counts[split]=profiles_for(prepared,split)
            resources.callback(profiles[split].close)
            limit=config['export_requests'][split]
            diagnostics[split]=export_split(model,request_iter(prepared['dbs'][split],limit),prepared['dbs'][split],split,
                work,prepared['catalogue'],profiles[split],prepared['vectors'],config.get('catalogue_batch_size',4096))
            checks[split]=validate_export(work,split,lambda s=split,n=limit:request_iter(prepared['dbs'][s],n),
                prepared['catalogue'],prepared['dbs'][split],prepared['budgets'][split]['eligible_pairs'],
                model=model,profiles=profiles[split],vectors=prepared['vectors'],
                batch_size=config.get('catalogue_batch_size',4096))
        exported_validation=evaluate_exported_validation(work,request_iter(prepared['dbs']['validation'],config['export_requests']['validation']),
            prepared['dbs']['validation'],prepared['catalogue'])
        req=prepared['validation_requests'][0]
        ids=eligible_ids(prepared['catalogue'],req[3])[:10]
        if not len(ids):
            with closing(sqlite3.connect(prepared['dbs']['validation'])) as db:
                ids=np.asarray([r[0] for r in db.execute('SELECT article_id FROM candidates WHERE behaviour_id=? LIMIT 10',(req[0],))],dtype='int64')
        profile,coverage=profiles['validation'].get(int(req[1]),(None,0.0))
        probe={'article_ids':ids,'user_id':np.asarray(int(req[1]),dtype='int64'),
               'profile_missing':np.asarray(profile is None),
               'profile':profile if profile is not None else np.zeros(prepared['audit']['embedding_info']['dimension'],dtype='float32'),
               'coverage':np.asarray(coverage),
               'scores':cosine_scores(ids,int(req[1]),profiles['validation'],prepared['vectors'])}
        np.savez_compressed(work/'model/reload_probe.npz',**probe)
        shutil.copy2(__file__,work/'model/catalogue_pipeline.py')
        versions={name:version(name) for name in ['numpy','pandas','pyarrow','scikit-learn']}
        if prepared['vectors'].info['similarity_device']=='cuda':versions['torch']=version('torch')
        (work/'requirements.txt').write_text(''.join(f'{name}=={v}\n' for name,v in versions.items()))
        ranking,_=score_request(model,req,prepared['catalogue'],profiles['validation'],prepared['vectors'],config.get('catalogue_batch_size',4096))
        sample_ids=np.asarray(ranking[:5],dtype='int64')
        details=semantic_values(sample_ids,int(req[1]),profiles['validation'],prepared['vectors'])
        save_json(work/'validation_example.json',{'behaviour_id':req[0],
            'articles':[{'article_id':int(a),'cosine_similarity':float(row[0]),'candidate_vector_missing':bool(row[1]),
                         'user_profile_missing':bool(row[2]),'history_vector_coverage':float(row[3])} for a,row in zip(sample_ids,details)],
            'note':'Scores are direct similarity, with no learned weighting or click probability.'})
        selected_metrics=exported_validation if config.get('scope')=='final' else benchmark['metrics']
        save_json(work/'metrics.json',{'validation':selected_metrics,'validation_sample':benchmark['metrics'],
            'exported_validation':exported_validation,'recorded_slate_diagnostics':diagnostics,'test_metrics':None,
            'scope':config.get('scope','development'),'model_selection':'No search performed; one predeclared fixed similarity configuration.'})
        record={'pipeline_version':PIPELINE_VERSION,'config':config,'model':model,'export_schema_version':7,
            'export_profile':'top200_only','shortlist_score_column':None,'score_companion_column':'similarity_scores','shared_full_score_contract_complete':False,
            'prediction_pool':'time_eligible_catalogue','score_semantics':'cosine similarity, not click probability',
            'probability_column':None,'shortlist_size':200,'final_display_size':5,
            'inputs':prepared['audit']['inputs'],'embedding_source':config.get('embedding_source'),
            'embedding_dimension':prepared['audit']['embedding_info']['dimension'],'compute':prepared['vectors'].info,
            'eligibility_policy':POLICY,'history_coverage':history_counts,'export_checks':checks,'export_cost':cost,
            'learned_parameters':False,'trained_components':[],'frozen_components':['shared text embeddings'],
            'feature_usage':'Only the declared candidate-to-history cosine aggregation ranks articles. Metadata, flags and labels are not score predictors.',
            'python':sys.version,'packages':versions,'code_provenance':config.get('code_provenance',{}),
            'scope':config.get('scope','development'),
            'all_agreed_requests_exported':all(config['export_requests'][s] is None for s in SPLITS),
            'exports':{s:{'shortlist':f'rankings/{s}/submission_top_200.parquet',
                'shortlist_scores':f'rankings/{s}/submission_top_200_scores.parquet',
                'eligibility':f'eligibility/{s}_requests.csv'} for s in SPLITS},
            'handoff_complete':False,'persistence':{'persistent_copy_complete':False,'reload_from_persistent_copy':False},
            'limitations':['Top-200-only handoff requested; exhaustive score files from the original shared contract are omitted.',
                'Observed-click recovery is not fully observed catalogue relevance or retention.',
                'Validation has already informed development; it is not an untouched final test.',
                'Zero fallback scores can outrank negative cosine; missingness and ties are reported.',
                'Catalogue policy/input versions must match other models for fair comparison.',
                'Provided article snapshots/encoder pretraining may postdate historical requests.']}
        save_json(work/'run_config.json',record)
        try:persistence=persist_and_reload(prepared,probe)
        except BaseException as error:
            record['persistence'].update(status='incomplete',error_type=type(error).__name__,message=str(error))
            save_json(work/'run_config.json',record);save_json(work/'reload_check.json',record['persistence'])
            raise
        record['persistence']=persistence
        record['handoff_complete']=persistence['persistent_copy_complete'] and record['all_agreed_requests_exported'] and config.get('scope')=='final'
        result={'metrics':selected_metrics,'validation_sample':benchmark['metrics'],
            'exported_validation':exported_validation,'diagnostics':diagnostics,'checks':checks,
            'persistence':persistence,'handoff_complete':record['handoff_complete']}
        save_json(work/'run_config.json',record);save_json(work/'reload_check.json',persistence);save_json(result_path,result)
        if persistence['persistent_copy_complete']:
            target=Path(persistence['persistent_location'])
            for name in ['run_config.json','reload_check.json','result.json']:shutil.copy2(work/name,target/name)
        return result


def shortlist_schemas():
    native=pa.schema([pa.field('behaviour_id',pa.string(),nullable=False),
                      pa.field('article_ids',pa.list_(pa.int64()),nullable=False)])
    scores=pa.schema(list(native)+[pa.field('similarity_scores',pa.list_(pa.float32()),nullable=False)])
    return native,scores


def write_shortlist_batch(rows,writer,score_writer=None):
    native,score_schema=shortlist_schemas()
    writer.write_table(pa.Table.from_pylist([
        {'behaviour_id':row['behaviour_id'],'article_ids':row['article_ids']} for row in rows],schema=native))
    if score_writer is not None:score_writer.write_table(pa.Table.from_pylist(rows,schema=score_schema))


def write_shortlist_parquet(csv_path, output_path=None):
    """Convert a legacy CSV without rescoring; new model exports do not create CSVs."""
    csv_path=Path(csv_path)
    target=Path(output_path) if output_path is not None else csv_path.with_suffix('.parquet')
    if target.resolve()==csv_path.resolve():raise ValueError('Parquet output cannot overwrite the source CSV')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.tmp.parquet')
    score_target=target.with_name(target.stem+'_scores.parquet')
    score_temporary=score_target.with_suffix('.tmp.parquet')
    native_schema,score_schema=shortlist_schemas()
    seen=set();pending=[];rows=articles=0;has_scores=False
    try:
        with csv_path.open(newline='') as source,ExitStack() as resources:
            reader=csv.DictReader(source)
            if not {'behaviour_id','article_ids'}<=set(reader.fieldnames or []):raise ValueError('CSV must contain behaviour_id and article_ids')
            has_scores='similarity_scores' in reader.fieldnames
            writer=resources.enter_context(pq.ParquetWriter(temporary,native_schema,compression='zstd'))
            score_writer=resources.enter_context(pq.ParquetWriter(score_temporary,score_schema,compression='zstd')) if has_scores else None
            for row in reader:
                key=row['behaviour_id'];ids=json.loads(row['article_ids'])
                if not key or key in seen:raise ValueError('Missing or duplicate behaviour_id')
                if not isinstance(ids,list) or len(ids)>200 or any(type(a) is not int for a in ids):
                    raise ValueError('article_ids must be an ordered list of up to 200 original integer IDs')
                if len(ids)!=len(set(ids)):raise ValueError('Duplicate article IDs in shortlist')
                item={'behaviour_id':key,'article_ids':ids}
                if has_scores:
                    scores=json.loads(row['similarity_scores'])
                    array=np.asarray(scores,dtype='float64')
                    if array.ndim!=1 or len(array)!=len(ids) or not np.isfinite(array).all():raise ValueError('Invalid similarity scores')
                    item['similarity_scores']=scores
                seen.add(key);rows+=1;articles+=len(ids);pending.append(item)
                if len(pending)>=512:
                    write_shortlist_batch(pending,writer,score_writer);pending.clear()
            if pending:write_shortlist_batch(pending,writer,score_writer)
        temporary.replace(target)
        if has_scores:score_temporary.replace(score_target)
    except BaseException:
        temporary.unlink(missing_ok=True);score_temporary.unlink(missing_ok=True);raise
    return {'path':str(target),'scores_path':str(score_target) if has_scores else None,
            'rows':rows,'article_ids':articles,'schema':'behaviour_id:string,article_ids:list<int64>',
            'order':'same recommendation order as the source; no sorting/grouping applied'}


def shortlist_parquet_rows(path):
    with Path(path).open('rb') as source:
        parquet=pq.ParquetFile(source);native,_=shortlist_schemas()
        if not parquet.schema_arrow.equals(native,check_metadata=False):raise ValueError('Incorrect native Parquet shortlist schema')
        for batch in parquet.iter_batches(batch_size=512):yield from batch.to_pylist()


def shortlist_score_rows(path):
    with Path(path).open('rb') as source:
        parquet=pq.ParquetFile(source);_,expected=shortlist_schemas()
        if not parquet.schema_arrow.equals(expected,check_metadata=False):raise ValueError('Incorrect Parquet score schema')
        for batch in parquet.iter_batches(batch_size=512):yield from batch.to_pylist()


def export_parquet_submissions(work,persistent_location=None):
    """Convert legacy runs or report direct Parquet exports; never recompute scores."""
    work=Path(work);converted={}
    for split in SPLITS:
        directory=work/f'rankings/{split}';csv_path=directory/'submission_top_200.csv'
        target=directory/'submission_top_200.parquet';score_path=directory/'submission_top_200_scores.parquet'
        if csv_path.is_file():converted[split]=write_shortlist_parquet(csv_path)
        elif target.is_file():
            rows=articles=0;seen=set()
            for row in shortlist_parquet_rows(target):
                if row['behaviour_id'] in seen:raise ValueError('Duplicate Parquet request')
                seen.add(row['behaviour_id']);rows+=1;articles+=len(row['article_ids'])
            converted[split]={'path':str(target),'scores_path':str(score_path) if score_path.is_file() else None,
                              'rows':rows,'article_ids':articles,'schema':'behaviour_id:string,article_ids:list<int64>'}
    if not converted:raise FileNotFoundError('No completed shortlist files under WORK/rankings')
    record_path=work/'run_config.json'
    if record_path.is_file():
        record=json.loads(record_path.read_text());record['export_schema_version']=7
        record['shortlist_score_column']=None;record['score_companion_column']='similarity_scores'
        record.pop('csv_shortlist_score_column',None)
        record['parquet_submission_schema']={'behaviour_id':'string','article_ids':'ordered list<int64>','rows':'one per request'}
        for split,info in converted.items():
            entry=record.setdefault('exports',{}).setdefault(split,{})
            entry['shortlist']=f'rankings/{split}/submission_top_200.parquet'
            if info['scores_path']:entry['shortlist_scores']=f'rankings/{split}/submission_top_200_scores.parquet'
            # Preserve any legacy CSV reference when converting an older bundle.
        save_json(record_path,record)
    save_json(work/'parquet_submissions.json',converted)
    if persistent_location:
        destination=Path(persistent_location)
        if not destination.is_dir():raise FileNotFoundError('Persistent run folder does not exist')
        if destination.resolve()!=work.resolve():
            for info in converted.values():
                for name in ['path','scores_path']:
                    if not info.get(name):continue
                    source=Path(info[name]);target=destination/source.relative_to(work)
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            for name in ['run_config.json','parquet_submissions.json']:
                if (work/name).is_file():shutil.copy2(work/name,destination/name)
    return converted
