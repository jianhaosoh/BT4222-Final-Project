"""Frozen text embeddings + incremental logistic relevance, export schema v3.

Only raw shared feature columns with a declared catalogue reconstruction are used.
No API requests, catalogue negative labels, or test-based selection are performed.
"""
from __future__ import annotations
import csv
from functools import lru_cache
from contextlib import closing, ExitStack
import hashlib
import heapq
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SPLITS = ('train', 'validation', 'test')
KEYS = ['user_id', 'impression_id', 'candidate_article_id', 'impression_time']
REQUEST = ['hour', 'day_of_week', 'is_weekend', 'is_subscriber',
           'is_sso_user', 'is_front_page', 'history_size']
ARTICLE = ['premium', 'body_word_count', 'topic_count', 'has_topics',
           'article_age_h', 'article_age_missing']
SEMANTIC = ['text_cosine', 'candidate_text_missing', 'history_text_missing',
            'history_text_coverage']
FEATURES = REQUEST + ARTICLE + SEMANTIC
POLICY = {'name': 'published_at_or_before_request_v1',
          'publication_field': 'published_time', 'missing_publication': 'exclude',
          'age_limit_hours': None, 'access_filter': None,
          'snapshot': 'ordered union of declared catalogues; first snapshot wins'}


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


def resolve_inputs(config):
    root=Path(config['drive_root'])
    # Explicit overrides take precedence. Ambiguous automatic discoveries fail.
    def resolve(name, choices):
        override=config.get('paths', {}).get(name)
        if override:
            p=Path(override); p=p if p.is_absolute() else root/p
            if not p.is_file(): raise FileNotFoundError(p)
            return p
        found=list(dict.fromkeys((root/p).resolve() for p in choices if (root/p).is_file()))
        if len(found)!=1:
            raise FileNotFoundError(f'{name}: expected exactly one input, found {found}. Set config["paths"]["{name}"]. Tried: {choices}')
        return found[0]
    inputs={'vocabulary': resolve('vocabulary',['feature_engineering/feature_vocabulary.json'])}
    inputs['catalogue_train_val']=resolve('catalogue_train_val',['train_val/articles.parquet'])
    inputs['catalogue_test']=resolve('catalogue_test',['test/articles.parquet'])
    for split in SPLITS:
        inputs[f'features_{split}']=resolve(f'features_{split}',[f'feature_engineering/{split}_features.parquet'])
        prefix=f'train_val/{split}' if split!='test' else 'test'
        inputs[f'history_{split}']=resolve(f'history_{split}',[f'{prefix}/history.parquet'])
    for part in ['train_val','test']:
        inputs[f'embeddings_{part}']=resolve(f'embeddings_{part}',[
            f'{part}/article embeddings/text_embeddings.parquet',
            f'embeddings/{part}/text_embeddings.parquet'])
    return inputs


def load_catalogue(paths):
    parts=[]
    for p in paths:
        f=pd.read_parquet(p, columns=['article_id','published_time','premium','body','topics'])
        if f.article_id.isna().any() or f.article_id.duplicated().any():
            raise ValueError(f'Null/duplicate article IDs in {p}')
        parts.append(f)
    combined=pd.concat(parts, ignore_index=True)
    overlaps=int(combined.article_id.duplicated().sum())
    combined=combined.drop_duplicates('article_id',keep='first').sort_values('article_id').reset_index(drop=True)
    combined['published_time']=pd.to_datetime(combined.published_time)
    combined['body_word_count']=combined.body.map(lambda v:len(v.split()) if isinstance(v,str) and v.strip() else np.nan)
    def topics(v):
        if v is None or (isinstance(v,float) and np.isnan(v)): return 0
        if not isinstance(v,(list,tuple,np.ndarray)): raise ValueError('Undecodable article topics')
        return len(set(str(t) for t in v if t is not None and not pd.isna(t)))
    combined['topic_count']=combined.topics.map(topics)
    combined['has_topics']=combined.topic_count.gt(0)
    return combined.drop(columns=['body','topics']).set_index('article_id',drop=False), overlaps


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


def index_shared(path, split, db_path, seed, expected_request_limit=None, catalogue=None):
    with closing(sqlite3.connect(db_path)) as db:
        return _index_shared(db,path,split,db_path,seed,expected_request_limit,catalogue)


def _index_shared(db, path, split, db_path, seed, expected_request_limit=None, catalogue=None):
    """Disk index for composite keys, complete slates, labels and request context."""
    db.execute('PRAGMA cache_size=-32768')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS requests (
        behaviour_id TEXT PRIMARY KEY, user_id INTEGER, impression_id INTEGER,
        time TEXT NOT NULL, context TEXT NOT NULL, priority TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS candidates (
        behaviour_id TEXT, article_id INTEGER, clicked INTEGER,
        PRIMARY KEY (behaviour_id,article_id));
    ''')
    if db.execute('SELECT count(*) FROM candidates').fetchone()[0]:
        db.close(); raise FileExistsError(f'Index already populated: {db_path}')
    available=set(pq.ParquetFile(path).schema_arrow.names)
    required=KEYS+REQUEST+ARTICLE+['split']
    if split!='test': required+=['clicked']
    elif 'clicked' in available: required+=['clicked']
    if not set(required)<=available: raise ValueError(f'{path} lacks {sorted(set(required)-available)}')
    rows=0
    for batch in pq.ParquetFile(path).iter_batches(columns=list(dict.fromkeys(required)),batch_size=8192):
        if catalogue is not None:
            data=batch.to_pandas()
            rebuilt=article_values(data.candidate_article_id.to_numpy(),data.impression_time.to_numpy(),catalogue)
            supplied=data[ARTICLE].apply(pd.to_numeric,errors='raise').to_numpy(dtype='float32',na_value=np.nan)
            if not np.allclose(rebuilt,supplied,equal_nan=True,atol=1e-4,rtol=1e-5):
                raise ValueError(f'{split}: shared article columns disagree with declared catalogue reconstruction')
        requests={}; candidates=[]
        for r in batch.to_pylist():
            if r['split']!=split: raise ValueError('Shared feature split column disagrees with file')
            key=identity(r['user_id'],r['impression_id'])
            if not isinstance(r['candidate_article_id'],int): raise ValueError('Candidate IDs must be original integers')
            label=r.get('clicked')
            if split=='test':
                if label is not None: raise ValueError('Test targets must be unavailable; test is never a tuning signal')
            elif label not in (0,1): raise ValueError('Training/validation labels must be genuine binary clicks')
            context=json.dumps([plain(r[c]) for c in REQUEST],allow_nan=False)
            time_value=stamp(r['impression_time'])
            item=(key,int(r['user_id']),int(r['impression_id']),time_value,context,
                  hashlib.blake2b(f'{seed}:{split}:{key}'.encode(),digest_size=8).hexdigest())
            if key in requests and requests[key]!=item: raise ValueError('Request features/time vary within a slate')
            requests[key]=item
            candidates.append((key,int(r['candidate_article_id']),label))
        for item in requests.values():
            previous=db.execute('SELECT time,context FROM requests WHERE behaviour_id=?',(item[0],)).fetchone()
            if previous and previous!=(item[3],item[4]): raise ValueError('Request changes across feature batches')
            db.execute('INSERT OR IGNORE INTO requests VALUES (?,?,?,?,?,?)',item)
        try: db.executemany('INSERT INTO candidates VALUES (?,?,?)',candidates)
        except sqlite3.IntegrityError as e: raise ValueError('Duplicate source candidate keys: investigate instead of deduplicating') from e
        db.commit(); rows+=len(candidates)
    if not rows: raise ValueError('Empty shared feature table')
    total=db.execute('SELECT count(*) FROM requests').fetchone()[0]
    start=db.execute('SELECT min(time) FROM requests').fetchone()[0]
    db.close()
    return {'candidate_rows':rows,'requests':total,'split_start':start,
            'sampling_rule':'Smallest BLAKE2b-64(seed:split:behaviour_id); all candidates retained'}


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


class ProfileLookup:
    """Bounded memory lookup for persistent, fixed history profiles."""
    def __init__(self,path,dimension):
        self.path=str(path);self.dimension=dimension
        self.connection=sqlite3.connect(self.path)
        self.get=lru_cache(maxsize=1024)(self._get)
    def _get(self,user,default=(None,0.0)):
        row=self.connection.execute('SELECT vector,coverage FROM profiles WHERE user_id=?',(int(user),)).fetchone()
        if row is None:return default
        return (np.frombuffer(row[0],dtype='float32') if row[0] is not None else None,float(row[1]))
    def close(self):self.get.cache_clear();self.connection.close()


def history_profiles(path, needed_users, cutoff, vectors, output_db=None):
    if output_db is None:return _history_profiles(path,needed_users,cutoff,vectors)
    with closing(sqlite3.connect(output_db)) as store:
        return _history_profiles(path,needed_users,cutoff,vectors,output_db,store)


def _history_profiles(path, needed_users, cutoff, vectors, output_db=None, store=None):
    profiles={}; seen=set(); excluded_missing_time=0
    if output_db is not None:
        store.execute('CREATE TABLE IF NOT EXISTS profiles (user_id INTEGER PRIMARY KEY, vector BLOB,coverage REAL)')
        if store.execute('SELECT count(*) FROM profiles').fetchone()[0]:raise FileExistsError('History store exists')
    dim=len(next(iter(vectors.values())))
    for batch in pq.ParquetFile(path).iter_batches(columns=['user_id','article_id_fixed','impression_time_fixed'],batch_size=64):
        for row in batch.to_pylist():
            uid=int(row['user_id'])
            if uid not in needed_users: continue
            if uid in seen: raise ValueError('Duplicate history users')
            seen.add(uid)
            ids,times=row['article_id_fixed'],row['impression_time_fixed']
            if ids is None or times is None or len(ids)!=len(times): raise ValueError('Misaligned history arrays')
            total=available=0; summed=np.zeros(dim,dtype='float64')
            for aid,when in zip(ids,times):
                if when is None: excluded_missing_time+=1; continue
                if pd.Timestamp(when)>=pd.Timestamp(cutoff):
                    raise ValueError('History reaches the behavior split; shared fixed-snapshot temporal audit failed')
                if aid is None: raise ValueError('Null historical article ID')
                total+=1
                if int(aid) in vectors: summed+=vectors[int(aid)]; available+=1
            norm=np.linalg.norm(summed)
            profile=(summed/norm).astype('float32') if norm>1e-8 else None
            coverage=available/max(total,1)
            if store is not None:
                store.execute('INSERT INTO profiles VALUES (?,?,?)',(uid,profile.tobytes() if profile is not None else None,coverage))
            else:profiles[uid]=(profile,coverage)
        if store is not None:store.commit()
    # An absent history row is explicit unavailable history, not an invented vector.
    if store is not None:
        store.close();profiles=ProfileLookup(output_db,dim)
    return profiles, {'users_requested':len(needed_users),'users_found':len(seen),
                      'missing_history_users':len(needed_users-seen),
                      'events_missing_time_excluded':excluded_missing_time}


def semantic_values(ids, user, profiles, vectors):
    profile,coverage=profiles.get(user,(None,0.0))
    result=np.zeros((len(ids),4),dtype='float32')
    for i,aid in enumerate(ids):
        vector=vectors.get(int(aid))
        result[i]=[float(vector@profile) if vector is not None and profile is not None else 0,
                   vector is None,profile is None,coverage]
    result[:,0]=np.clip(result[:,0],-1,1)
    return result


def article_values(ids, when, catalogue):
    meta=catalogue.reindex(ids)
    times=np.asarray(pd.to_datetime(when),dtype='datetime64[ns]')
    age=(times-meta.published_time.to_numpy(dtype='datetime64[ns]'))/np.timedelta64(1,'h')
    age=np.where(age>=0,age,np.nan)
    premium=pd.to_numeric(meta.premium,errors='coerce').to_numpy(dtype='float32',na_value=np.nan)
    body=meta.body_word_count.to_numpy(dtype='float32',na_value=np.nan)
    topics=meta.topic_count.fillna(0).to_numpy(dtype='float32')
    return np.column_stack([premium,body,topics,topics>0,age,np.isnan(age)]).astype('float32')


def catalogue_features(request, ids, catalogue, profiles, vectors):
    _,uid,_,when,context=request
    context=np.asarray(json.loads(context),dtype='float32')
    return np.column_stack([np.tile(context,(len(ids),1)),article_values(ids,when,catalogue),
                           semantic_values(ids,int(uid),profiles,vectors)]).astype('float32')


def selected_record_batches(path, selected, profiles, vectors, catalogue, batch_size=8192, audit=True):
    """Read existing expanded pairs; never re-explode or invent negative labels."""
    read=KEYS+REQUEST+ARTICLE+['clicked']
    for batch in pq.ParquetFile(path).iter_batches(columns=read,batch_size=batch_size):
        frame=batch.to_pandas()
        mask=[(selected is None or identity(u,i) in selected) for u,i in zip(frame.user_id,frame.impression_id)]
        frame=frame.loc[mask]
        if frame.empty: continue
        raw=frame[REQUEST+ARTICLE].apply(pd.to_numeric,errors='raise').to_numpy(dtype='float32',na_value=np.nan)
        extra=np.empty((len(frame),4),dtype='float32')
        for uid,positions in frame.reset_index(drop=True).groupby('user_id',sort=False).indices.items():
            extra[positions]=semantic_values(frame.candidate_article_id.iloc[positions].to_numpy(),int(uid),profiles,vectors)
        if audit:
            # Check exact reconstruction of the used article feature definitions.
            for positions in frame.reset_index(drop=True).groupby(['user_id','impression_id'],sort=False).indices.values():
                first=frame.iloc[int(positions[0])]
                req=(None,int(first.user_id),int(first.impression_id),first.impression_time,
                     json.dumps([plain(first[c]) for c in REQUEST]))
                recreated=catalogue_features(req,frame.candidate_article_id.iloc[positions].to_numpy(),catalogue,profiles,vectors)
                if not np.allclose(raw[positions,len(REQUEST):],recreated[:,len(REQUEST):len(REQUEST)+len(ARTICLE)],equal_nan=True,rtol=1e-5,atol=1e-4):
                    raise ValueError('Shared article features differ from catalogue reconstruction. Fix the source/version; do not fill unavailable fields with zeros.')
        y=frame.clicked.to_numpy(dtype='int8')
        if not np.isin(y,[0,1]).all(): raise ValueError('Invalid click label')
        yield np.column_stack([raw,extra]),y


def fill(raw, means):
    missing=~np.isfinite(raw)
    return np.column_stack([np.where(missing,means,raw),missing.astype('float32')])


def fit_transforms(batches):
    sums=np.zeros(len(FEATURES)); counts=np.zeros(len(FEATURES),dtype='int64');label_counts=np.zeros(2,dtype='int64')
    for x,y in batches():
        label_counts+=np.bincount(y,minlength=2)
        valid=np.isfinite(x); sums+=np.where(valid,x,0).sum(axis=0,dtype='float64'); counts+=valid.sum(axis=0)
    if not counts.any(): raise ValueError('No training features selected')
    if (label_counts==0).any():raise ValueError('Training requires both genuine click labels')
    means=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
    scaler=StandardScaler()
    for x,y in batches(): scaler.partial_fit(fill(x,means))
    return means,scaler,counts.tolist()


def predict(bundle, raw):
    x=bundle['scaler'].transform(fill(raw,bundle['imputation_means']))
    score=bundle['classifier'].decision_function(x)
    probability=bundle['classifier'].predict_proba(x)[:,1]
    return score,probability


def eligible_ids(catalogue, when):
    mask=catalogue.published_time.notna() & catalogue.published_time.le(pd.Timestamp(when))
    return catalogue.index[mask].to_numpy(dtype='int64')


def retrieval_metrics(ranking, known):
    if not known: return None
    hits=[int(a in known) for a in ranking]
    recall=sum(hits)/len(known)
    ideal=sum(1/math.log2(i+2) for i in range(min(100,len(known))))
    dcg=sum(v/math.log2(i+2) for i,v in enumerate(hits))
    return recall,dcg/ideal


def score_request(bundle, request, catalogue, profiles, vectors, batch_size, write=None, stats=None):
    ids=eligible_ids(catalogue,request[3]); top=[]
    for start in range(0,len(ids),batch_size):
        part=ids[start:start+batch_size]
        raw=catalogue_features(request,part,catalogue,profiles,vectors)
        scores,probs=predict(bundle,raw)
        if not np.isfinite(scores).all() or not np.isfinite(probs).all(): raise ValueError('Nonfinite output')
        if write:
            write.writerows((request[0],int(a),float(s),float(p)) for a,s,p in zip(part,scores,probs))
        local=sorted(zip(scores.tolist(),part.tolist()),key=lambda z:(-z[0],z[1]))[:101]
        top=sorted(top+local,key=lambda z:(-z[0],z[1]))[:101]
    if stats is not None:stats['ties_crossing_100']=len(top)>100 and top[99][0]==top[100][0]
    return [int(a) for _,a in top[:100]],len(ids)


def evaluate_catalogue(bundle, reqs, db_path, catalogue, profiles, vectors, batch_size):
    values=[]; eligible_recalls=[]; without=empty=excluded=total_known=ties=0
    exclusion_reasons={'missing_article':0,'missing_publication':0,'future_publication':0}
    with closing(sqlite3.connect(db_path)) as db:
        for req in reqs:
            known={r[0] for r in db.execute('SELECT article_id FROM candidates WHERE behaviour_id=? AND clicked=1',(req[0],))}
            stats={}
            ranking,count=score_request(bundle,req,catalogue,profiles,vectors,batch_size,stats=stats)
            ties+=stats['ties_crossing_100']
            empty+=count==0
            if not known: without+=1; continue
            metric=retrieval_metrics(ranking,known);values.append(metric)
            elig=set(eligible_ids(catalogue,req[3]));excluded+=len(known-elig);total_known+=len(known)
            if known&elig:eligible_recalls.append(len(set(ranking)&known)/len(known&elig))
            for aid in known-elig:
                reason='missing_article' if aid not in catalogue.index else ('missing_publication' if pd.isna(catalogue.loc[aid,'published_time']) else 'future_publication')
                exclusion_reasons[reason]+=1
    if not values: raise ValueError('Selection sample has no requests with known clicks')
    return {'known_click_recall@100':float(np.mean([v[0] for v in values])),
            'observed_click_ndcg@100':float(np.mean([v[1] for v in values])),
            'evaluated_requests':len(values),'requests_without_known_clicks':without,
            'empty_eligible_pools':empty,'known_clicks_excluded_by_eligibility':excluded,
            'known_clicks':total_known,'eligible_only_recall@100':float(np.mean(eligible_recalls)) if eligible_recalls else None,
            'known_click_exclusion_reasons':exclusion_reasons,'requests_with_ties_crossing_100':ties,'known_click_eligibility_coverage':1-excluded/max(total_known,1),
            'ties':'descending score, ascending original article ID; exact top-100 membership',
            'outcomes':'unshown outcomes unknown; zero observed gain is not a negative label'}


def budget(reqs, catalogue, bytes_per_pair=130):
    published=np.sort(catalogue.published_time.dropna().to_numpy(dtype='datetime64[ns]'))
    counts=[int(np.searchsorted(published,np.datetime64(pd.Timestamp(r[3]),'ns'),side='right')) for r in reqs]
    pairs=sum(counts)
    return {'requests':len(counts),'eligible_pairs':pairs,'empty_pools':sum(c==0 for c in counts),
            'estimated_csv_bytes':pairs*bytes_per_pair,'bytes_per_pair_assumption':bytes_per_pair,
            'catalogue_articles':len(catalogue),'missing_publication_excluded':int(catalogue.published_time.isna().sum()),
            'runtime_seconds':None,'runtime_note':'Measure a bounded sample; no reliable timing estimate before scoring.'}


def export_split(bundle, reqs, db_path, split, out, catalogue, profiles, vectors, batch_size):
    paths={kind:out/f'{kind}/{split}' for kind in ['predictions','rankings']}
    for p in paths.values():p.mkdir(parents=True,exist_ok=True)
    diag=out/'diagnostics/recorded_slates';diag.mkdir(parents=True,exist_ok=True)
    manifest=out/'eligibility';manifest.mkdir(exist_ok=True)
    diagnostics=[]; missing_recorded=known_elig=known_total=request_count=0
    # Test labels are never queried for evaluation or diagnostic metrics.
    with closing(sqlite3.connect(db_path)) as db, \
         (paths['predictions']/'submission_click_probabilities.csv').open('w',newline='') as pf, \
         (paths['rankings']/'submission_top_100.csv').open('w',newline='') as rf, \
         (diag/f'{split}_scores.csv').open('w',newline='') as df, \
         (manifest/f'{split}_requests.csv').open('w',newline='') as ef:
        pw,rw,dw,ew=map(csv.writer,[pf,rf,df,ef])
        pw.writerow(['behaviour_id','article_id','score','click_probability'])
        rw.writerow(['behaviour_id','article_ids'])
        dw.writerow(['behaviour_id','article_id','score','click_probability'])
        ew.writerow(['behaviour_id','impression_time','eligible_article_count','recorded_candidates_outside_pool','ties_crossing_100'])
        for req in reqs:
            request_count+=1;stats={}
            ranking,count=score_request(bundle,req,catalogue,profiles,vectors,batch_size,pw,stats)
            rw.writerow([req[0],json.dumps(ranking)])
            rows=db.execute('SELECT article_id,clicked FROM candidates WHERE behaviour_id=? ORDER BY article_id',(req[0],)).fetchall()
            ids=np.array([r[0] for r in rows],dtype='int64')
            raw=catalogue_features(req,ids,catalogue,profiles,vectors)
            scores,probs=predict(bundle,raw)
            dw.writerows((req[0],int(a),float(s),float(p)) for a,s,p in zip(ids,scores,probs))
            excluded=sum(a not in catalogue.index or pd.isna(catalogue.loc[a,'published_time']) or catalogue.loc[a,'published_time']>pd.Timestamp(req[3]) for a in ids)
            missing_recorded+=excluded; ew.writerow([req[0],req[3],count,excluded,stats['ties_crossing_100']])
            if split!='test':
                y=np.array([r[1] for r in rows]); known=set(ids[y==1])
                if len(np.unique(y))==2: diagnostics.append(float(roc_auc_score(y,scores)))
                eligible=set(eligible_ids(catalogue,req[3])); known_total+=len(known); known_elig+=len(known&eligible)
    return {'recorded_slate_mean_impression_auc':float(np.mean(diagnostics)) if diagnostics else None,
            'auc_evaluable_requests':len(diagnostics),'requests_exported':request_count,
            'recorded_candidates_outside_pool':missing_recorded,
            'test_metrics_available':False if split=='test' else None,
            'known_click_eligibility_coverage':known_elig/known_total if known_total else None}



def evaluate_exported_validation(out, reqs, db_path, catalogue):
    """Evaluate already exported shortlists, without rescoring the catalogue."""
    values=[];eligible_values=[];without=empty=excluded=total_known=ties=0
    reasons={'missing_article':0,'missing_publication':0,'future_publication':0}
    with closing(sqlite3.connect(db_path)) as db, \
         (out/'rankings/validation/submission_top_100.csv').open(newline='') as f, \
         (out/'eligibility/validation_requests.csv').open(newline='') as ef:
        rankings=iter(csv.DictReader(f));eligibility=iter(csv.DictReader(ef))
        for req in reqs:
            row=next(rankings);detail=next(eligibility)
            if row['behaviour_id']!=req[0] or detail['behaviour_id']!=req[0]:raise ValueError('Export metric request alignment failure')
            ranking=json.loads(row['article_ids']);empty+=int(detail['eligible_article_count'])==0
            ties+=detail['ties_crossing_100']=='True'
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
    return {'known_click_recall@100':float(np.mean([v[0] for v in values])) if values else None,
            'observed_click_ndcg@100':float(np.mean([v[1] for v in values])) if values else None,
            'evaluated_requests':len(values),'requests_without_known_clicks':without,
            'empty_eligible_pools':empty,'known_clicks_excluded_by_eligibility':excluded,
            'known_clicks':total_known,'known_click_eligibility_coverage':1-excluded/max(total_known,1),
            'eligible_only_recall@100':float(np.mean(eligible_values)) if eligible_values else None,
            'known_click_exclusion_reasons':reasons,'requests_with_ties_crossing_100':ties,
            'source':'validated exported catalogue shortlists; original known-click denominators'}

def validate_export(out, split, reqs, catalogue, db_path, expected_pairs):
    """Stream-check every pair, probability, deterministic top-100 and empty pool."""
    score_path=out/f'predictions/{split}/submission_click_probabilities.csv'
    rank_path=out/f'rankings/{split}/submission_top_100.csv'
    pair_count=request_count=0
    with score_path.open(newline='') as sf,rank_path.open(newline='') as rf:
        scores=iter(csv.DictReader(sf)); current=next(scores,None)
        rankings=iter(csv.DictReader(rf))
        for req in reqs():
            request_count+=1
            ranked=next(rankings,None)
            if ranked is None or ranked['behaviour_id']!=req[0]:raise ValueError('Ranking request coverage/order mismatch')
            ids=json.loads(ranked['article_ids'])
            eligible=eligible_ids(catalogue,req[3]).tolist(); seen=[]; top=[]
            while current and current['behaviour_id']==req[0]:
                aid=int(current['article_id']);score=float(current['score']);prob=float(current['click_probability'])
                if not np.isfinite(score) or not np.isfinite(prob) or not 0<=prob<=1:raise ValueError('Invalid exported score/probability')
                seen.append(aid)
                entry=(score,-aid)
                if len(top)<100:heapq.heappush(top,entry)
                elif entry>top[0]:heapq.heapreplace(top,entry)
                pair_count+=1;current=next(scores,None)
            if seen!=eligible:raise ValueError('Full score export does not match exact eligible pool')
            if ids!=[-negative_id for _,negative_id in sorted(top,reverse=True)] or len(ids)!=min(100,len(eligible)):raise ValueError('Top-100 export mismatch')
        if current is not None or next(rankings,None) is not None:raise ValueError('Unexpected extra requests')
    if pair_count!=expected_pairs:raise ValueError('Exported pair count differs from budget')
    with closing(sqlite3.connect(db_path)) as db:
        expected=sum(db.execute('SELECT count(*) FROM candidates WHERE behaviour_id=?',(r[0],)).fetchone()[0] for r in reqs())
        path=out/f'diagnostics/recorded_slates/{split}_scores.csv'
        count=0
        with path.open(newline='') as f:
            reader=iter(csv.DictReader(f))
            for req in reqs():
                for (aid,) in db.execute('SELECT article_id FROM candidates WHERE behaviour_id=? ORDER BY article_id',(req[0],)):
                    row=next(reader,None)
                    if row is None or row['behaviour_id']!=req[0] or int(row['article_id'])!=aid:raise ValueError('Recorded-slate coverage mismatch')
                    count+=1
            if next(reader,None) is not None:raise ValueError('Extra recorded-slate rows')
        if count!=expected:raise ValueError('Recorded-slate diagnostic count mismatch')
    return {'catalogue_pairs_verified':pair_count,'requests_verified':request_count,'recorded_pairs_verified':expected}


def prepare(config, work):
    """Read real schemas/vectors and disk-index the shared recorded rows."""
    if config.get('feature_case','raw')!='raw':
        raise ValueError('This experiment deliberately uses the raw shared case and its own train-fitted transforms')
    if config.get('scope','development') not in ('development','final'):
        raise ValueError('Scope must explicitly be development or final')
    work=Path(work);(work/'indexes').mkdir(parents=True,exist_ok=True)
    inputs=resolve_inputs(config)
    vocabulary=json.loads(inputs['vocabulary'].read_text())
    catalogue,overlap=load_catalogue([inputs['catalogue_train_val'],inputs['catalogue_test']])
    vectors,embedding_info=load_embeddings([inputs['embeddings_train_val'],inputs['embeddings_test']],
        config.get('embedding_column','text_embedding'),config.get('embedding_dimension'))
    fingerprints={name:fingerprint(path,config.get('hash_inputs',False)) for name,path in inputs.items()}
    estimated_index_bytes=sum(fingerprints[f'features_{split}']['rows'] for split in SPLITS)*320
    if estimated_index_bytes>config.get('max_index_gb',20)*1e9 or estimated_index_bytes>shutil.disk_usage(work).free*0.7:
        save_json(work/'index_budget.json',{'estimated_index_bytes':estimated_index_bytes,'assumption':'320 bytes per input candidate, conservative planning estimate'})
        raise RuntimeError('Disk index estimate exceeds configured/local budget. See index_budget.json; use a machine with sufficient storage before full indexing.')
    previous_audit=work/'input_audit.json'
    if previous_audit.exists():
        prior=json.loads(previous_audit.read_text())
        if prior['inputs']!=fingerprints or prior.get('seed')!=config['seed']:
            raise ValueError('Input versions or sampling seed changed; use a new run ID')
    index_info={}; dbs={}
    for split in SPLITS:
        dbs[split]=work/f'indexes/{split}.sqlite'
        state=work/f'indexes/{split}.json'
        current=fingerprints[f'features_{split}']
        if state.exists():
            previous=json.loads(state.read_text())
            if previous['input']!=current:raise ValueError('Indexed input changed; use a new run folder')
            index_info[split]=previous['counts']
        else:
            if dbs[split].exists():dbs[split].unlink()  # only an unfinished generated index
            index_info[split]=index_shared(inputs[f'features_{split}'],split,dbs[split],config['seed'],catalogue=catalogue)
            save_json(state,{'input':current,'counts':index_info[split]})
    with closing(sqlite3.connect(dbs['train'])) as db:
        train_max=db.execute('SELECT max(time) FROM requests').fetchone()[0]
    if train_max>=index_info['validation']['split_start']:
        raise ValueError('Shared training and validation periods overlap')
    limits=config.get('export_requests',{'train':25,'validation':25,'test':25})
    if config.get('scope','development')=='final':
        if any(limits.get(s) is not None for s in SPLITS) or config.get('train_requests') is not None:
            raise ValueError('Final scope requires full shared training and all requests in every export split')
        if not config.get('owner'):raise ValueError('Record the experiment owner for final handoff')
        if not config.get('input_version'):raise ValueError('Declare the shared input release/version before a final run')
        if not isinstance(config.get('embedding_source'),dict) or not config['embedding_source'].get('model'):
            raise ValueError('Confirm the actual shared embedding encoder identity before a final run')
    budgets={s:budget(request_iter(dbs[s],limits.get(s)),catalogue) for s in SPLITS}
    full_budgets={s:budget(request_iter(dbs[s]),catalogue) for s in SPLITS}
    selection=requests(dbs['validation'],config.get('selection_requests',100))
    evidence={'inputs':fingerprints,'seed':config['seed'],'input_version':config.get('input_version'),
        'vocabulary_version':vocabulary.get('version'), 'catalogue_overlap_first_wins':overlap,
        'catalogue_articles':len(catalogue),'embedding_info':embedding_info,
        'catalogue_articles_with_vectors':len(set(catalogue.index)&set(vectors)),
        'catalogue_articles_without_vectors':len(set(catalogue.index)-set(vectors)),
        'request_counts':index_info,'planned_exports':budgets,'full_export_costs':full_budgets,
        'estimated_index_bytes':estimated_index_bytes,'eligibility_policy':POLICY,'selection_requests':[r[0] for r in selection],
        'hash_note':'sha256 content hashes recorded' if config.get('hash_inputs') else 'File size, modification time and schema identify inputs; these are not content hashes.'}
    save_json(work/'input_audit.json',evidence)
    return {'config':config,'work':work,'inputs':inputs,'catalogue':catalogue,'vectors':vectors,
            'dbs':dbs,'index_info':index_info,'budgets':budgets,'selection_requests':selection,'audit':evidence}


def profiles_for(prepared, split, include_training=False):
    config=prepared['config'];db=prepared['dbs'][split]
    limits=[config.get('export_requests',{}).get(split)]
    if split=='validation':limits.append(config.get('selection_requests',100))
    if split=='train' and include_training:limits.append(config.get('train_requests',5000))
    users=set()
    for limit in limits:
        users.update(int(r[1]) for r in request_iter(db,limit))
    profile_path=prepared['work']/f'indexes/{split}_profiles.sqlite'
    state=profile_path.with_suffix('.json')
    key={'users_sha256':hashlib.sha256(','.join(map(str,sorted(users))).encode()).hexdigest(),
         'history':prepared['audit']['inputs'][f'history_{split}'],
         'cutoff':prepared['index_info'][split]['split_start']}
    if state.exists():
        saved=json.loads(state.read_text())
        if saved['key']!=key:raise ValueError('History cache configuration changed; use a new run')
        return ProfileLookup(profile_path,prepared['audit']['embedding_info']['dimension']),saved['counts']
    if profile_path.exists():profile_path.unlink()
    lookup,counts=history_profiles(prepared['inputs'][f'history_{split}'],users,key['cutoff'],
                                   prepared['vectors'],profile_path)
    save_json(state,{'key':key,'counts':counts})
    return lookup,counts


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


def train(prepared):
    with ExitStack() as resources:
        try:return _train(prepared,resources)
        except BaseException as error:
            work=prepared['work'];current=work/'tuning/current_trial.json'
            failure=json.loads(current.read_text()) if current.exists() else {'status':'setup_failed'}
            failure.update(status='failed',error_type=type(error).__name__,message=str(error))
            path=work/'tuning/trials.json'
            records=json.loads(path.read_text()) if path.exists() else []
            records.append(failure);save_json(path,records)
            pd.json_normalize(records,sep='.').to_csv(work/'tuning/trials.csv',index=False)
            raise


def _train(prepared, resources):
    config=prepared['config'];work=prepared['work']
    if (work/'model/selected.joblib').exists():raise FileExistsError('Selected model exists; use load-only mode or a new run ID')
    train_profiles,train_profile_info=profiles_for(prepared,'train',True)
    resources.callback(train_profiles.close)
    val_profiles,val_profile_info=profiles_for(prepared,'validation')
    resources.callback(val_profiles.close)
    selected=None if config.get('train_requests',5000) is None else {r[0] for r in requests(prepared['dbs']['train'],config.get('train_requests',5000))}
    if selected is not None:save_json(work/'training_request_ids.json',sorted(selected))
    def batches():
        return selected_record_batches(prepared['inputs']['features_train'],selected,train_profiles,
            prepared['vectors'],prepared['catalogue'],config.get('batch_size',8192),audit=False)
    means,scaler,counts=fit_transforms(batches)
    attempts=[];best=None;best_key=(-1,-1);evaluations=0
    # A small, declared search. No estimator fit uses validation labels.
    alphas=config.get('alphas',[0.0001,0.001]);epochs=config.get('epochs',3)
    if epochs<1 or not alphas or any(a<=0 for a in alphas):raise ValueError('Use positive epochs/alphas')
    min_gain=config.get('minimum_recall_improvement',0.0);patience=config.get('patience',2)
    if patience<1:raise ValueError('Patience must be positive')
    start=time.perf_counter()
    for trial,alpha in enumerate(alphas):
        classifier=SGDClassifier(loss='log_loss',alpha=alpha,average=True,random_state=config['seed'])
        trial_key=(-1,-1);stale=0
        for epoch in range(epochs):
            save_json(work/'tuning/current_trial.json',{'trial':trial,'alpha':alpha,'epoch':epoch+1,'status':'running'})
            rng=np.random.default_rng(config['seed']+epoch)
            loss_sum=rows=0
            for raw,y in batches():
                x=scaler.transform(fill(raw,means));order=rng.permutation(len(y))
                classifier.partial_fit(x[order],y[order],classes=np.array([0,1],dtype='int8'))
                p=np.clip(classifier.predict_proba(x)[:,1],1e-12,1-1e-12)
                loss_sum+=float(-(y*np.log(p)+(1-y)*np.log(1-p)).sum());rows+=len(y)
            if not rows:raise ValueError('No training rows')
            bundle={'classifier':classifier,'scaler':scaler,'imputation_means':means,
                    'feature_columns':FEATURES,'missingness_columns':[c+'__missing' for c in FEATURES],
                    'frozen_components':['shared text embedding encoder'],
                    'trained_components':['imputation means','StandardScaler','SGD logistic classifier'],
                    'alpha':alpha,'epoch':epoch+1,'seed':config['seed']}
            eval_start=time.perf_counter()
            metrics=evaluate_catalogue(bundle,prepared['selection_requests'],prepared['dbs']['validation'],
                prepared['catalogue'],val_profiles,prepared['vectors'],config.get('catalogue_batch_size',4096))
            elapsed=time.perf_counter()-eval_start;evaluations+=1
            key=(metrics['known_click_recall@100'],metrics['observed_click_ndcg@100'])
            record={'trial':trial,'alpha':alpha,'epoch':epoch+1,'training_rows':rows,
                    'training_loss_after_batch_updates':loss_sum/rows,'validation':metrics,
                    'selection_seconds':elapsed,'status':'complete'}
            attempts.append(record)
            print(f'Trial {trial}, epoch {epoch+1}: Recall@100={key[0]:.6f}, observed nDCG@100={key[1]:.6f}',flush=True)
            # Earlier checkpoints win exact ties. Secondary nDCG breaks equal recall.
            if key>best_key:
                best_key=key;best=record.copy()
                model_dir=work/'model';model_dir.mkdir(exist_ok=True)
                tmp=model_dir/'selected.tmp.joblib';joblib.dump(bundle,tmp);tmp.replace(model_dir/'selected.joblib')
                if config.get('persistent_root'):
                    checkpoint=Path(config['persistent_root'])/config['experiment_id']/(config['run_id']+'_in_progress')
                    checkpoint.mkdir(parents=True,exist_ok=True)
                    temporary=checkpoint/'selected.tmp.joblib'
                    shutil.copy2(model_dir/'selected.joblib',temporary)
                    temporary.replace(checkpoint/'selected.joblib')
                    shutil.copy2(__file__,checkpoint/'catalogue_pipeline.py')
                    save_json(checkpoint/'checkpoint_config.json',{'config':config,'selected_so_far':record,
                        'input_audit':prepared['audit'],'status':'incomplete inference checkpoint, not a complete run bundle',
                        'exact_training_resumption_supported':False})

            improves=(key[0]>trial_key[0]+min_gain) or (key[0]==trial_key[0] and key[1]>trial_key[1])
            if improves:trial_key=key;stale=0
            else:stale+=1
            save_json(work/'tuning/trials.json',attempts)
            pd.json_normalize(attempts,sep='.').to_csv(work/'tuning/trials.csv',index=False)
            if stale>=patience:break
    trained={'selected':best,'objective':'maximize catalogue known-click Recall@100',
             'secondary':'observed-click nDCG@100','tie_policy':'earlier evaluation',
             'search_space':{'alphas':alphas,'epochs':epochs,'patience':patience,
                             'minimum_recall_improvement':min_gain},
             'selection_evaluations':evaluations,'training_imputation_nonmissing_counts':counts,
             'history_coverage':{'train':train_profile_info,'validation':val_profile_info},
             'train_seconds':time.perf_counter()-start,
             'training_curve_note':'Average after-update batch loss is an in-sample optimisation diagnostic.',
             'resumption_supported':False,'validation_role':'development, repeatedly used for checkpoint selection'}
    save_json(work/'training_summary.json',trained)
    train_profiles.close();val_profiles.close()
    return trained


def persist_and_reload(prepared,bundle,probe,original):
    config=prepared['config'];work=prepared['work']
    destination=config.get('persistent_root')
    if not destination:
        return {'persistent_copy_complete':False,'reload_from_persistent_copy':False,
                'reason':'Set persistent_root to your shared Drive experiment location.',
                'transfer_instructions':'Copy this entire run folder, preserving its model, code, manifests and exports.'}
    target=Path(destination)/config['experiment_id']/config['run_id']
    if target.resolve()==work.resolve() or work.resolve() in target.resolve().parents:
        raise ValueError('Persistent destination must be outside the local working run')
    if target.exists():raise FileExistsError(f'Persistent run already exists: {target}')
    target.parent.mkdir(parents=True,exist_ok=True)
    # Do not copy temporary disk indexes. Their source paths/hashes and code are in the manifest.
    shutil.copytree(work,target,ignore=shutil.ignore_patterns('indexes','*.tmp'))
    local_files={str(p.relative_to(work)):p.stat().st_size for p in work.rglob('*') if p.is_file() and 'indexes' not in p.relative_to(work).parts}
    for relative,size in local_files.items():
        other=target/relative
        if not other.is_file() or other.stat().st_size!=size:raise RuntimeError(f'Incomplete persistent copy: {relative}')
    loaded=joblib.load(target/'model/selected.joblib')
    actual=predict(loaded,probe)
    for a,b in zip(original,actual):np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-8)
    return {'persistent_copy_complete':True,'persistent_location':str(target),
            'copied_files_verified_by_size':len(local_files),'reload_from_persistent_copy':True,
            'reload_probe_rows':len(probe),'rtol':1e-7,'atol':1e-8,'retrained_for_check':False}


def execute(prepared,train_model=True,export=True):
    with ExitStack() as resources:
        return _execute(prepared,resources,train_model,export)


def _execute(prepared,resources,train_model=True,export=True):
    config=prepared['config'];work=prepared['work']
    if train_model:training=train(prepared)
    else:
        if not (work/'model/selected.joblib').exists():raise FileNotFoundError('Load-only mode needs a complete saved selected model')
        training=json.loads((work/'training_summary.json').read_text())
    bundle=joblib.load(work/'model/selected.joblib')
    profiles={};history_counts={}
    for split in SPLITS:
        profiles[split],history_counts[split]=profiles_for(prepared,split,split=='train')
        resources.callback(profiles[split].close)
    validation=evaluate_catalogue(bundle,prepared['selection_requests'],prepared['dbs']['validation'],
        prepared['catalogue'],profiles['validation'],prepared['vectors'],config.get('catalogue_batch_size',4096))
    checks={};diagnostics={};cost=None
    try:
        if export:
            cost=check_cost(prepared)
            selection_pairs=budget(prepared['selection_requests'],prepared['catalogue'])['eligible_pairs']
            measured=training['selected'].get('selection_seconds')
            cost['selection_benchmark_pairs']=selection_pairs
            cost['selection_benchmark_seconds']=measured
            cost['estimated_scoring_seconds_excluding_csv_io']=measured*cost['pairs']/selection_pairs if measured and selection_pairs else None
            cost['runtime_note']='Scaled from measured catalogue selection; excludes CSV I/O, validation passes, indexing and persistence overhead.'
            if cost['estimated_scoring_seconds_excluding_csv_io'] is not None and cost['estimated_scoring_seconds_excluding_csv_io']>config.get('max_scoring_seconds',3600):
                save_json(work/'export_budget.json',cost)
                raise RuntimeError('Measured scoring estimate exceeds declared runtime budget; inspect export_budget.json')
            for split in SPLITS:
                limit=config.get('export_requests',{}).get(split)
                diagnostics[split]=export_split(bundle,request_iter(prepared['dbs'][split],limit),prepared['dbs'][split],split,work,
                    prepared['catalogue'],profiles[split],prepared['vectors'],config.get('catalogue_batch_size',4096))
                checks[split]=validate_export(work,split,lambda s=split,n=limit:request_iter(prepared['dbs'][s],n),
                    prepared['catalogue'],prepared['dbs'][split],prepared['budgets'][split]['eligible_pairs'])
        else:raise RuntimeError('Export was disabled: handoff remains incomplete')
        exported_validation=evaluate_exported_validation(work,request_iter(prepared['dbs']['validation'],config.get('export_requests',{}).get('validation')),
            prepared['dbs']['validation'],prepared['catalogue'])
        sample=prepared['selection_requests'][0]
        ids=eligible_ids(prepared['catalogue'],sample[3])[:10]
        if not len(ids):
            with closing(sqlite3.connect(prepared['dbs']['validation'])) as db:
                ids=np.array([r[0] for r in db.execute('SELECT article_id FROM candidates WHERE behaviour_id=? LIMIT 10',(sample[0],))])
        probe=catalogue_features(sample,ids,prepared['catalogue'],profiles['validation'],prepared['vectors'])
        original=predict(bundle,probe)
        np.savez_compressed(work/'model/reload_probe.npz',raw=probe,score=original[0],probability=original[1])
        versions={n:version(n) for n in ['numpy','pandas','pyarrow','scikit-learn','joblib']}
        (work/'requirements.txt').write_text(''.join(f'{n}=={v}\n' for n,v in versions.items()))
        shutil.copy2(__file__,work/'model/catalogue_pipeline.py')
        feature_names=FEATURES+[c+'__missing' for c in FEATURES]
        pd.DataFrame({'feature':feature_names,'standardized_coefficient':bundle['classifier'].coef_[0]}).to_csv(work/'model/coefficients.csv',index=False)
        ranking,_=score_request(bundle,sample,prepared['catalogue'],profiles['validation'],prepared['vectors'],config.get('catalogue_batch_size',4096))
        explained_ids=np.asarray(ranking[:5],dtype='int64')
        explained_raw=catalogue_features(sample,explained_ids,prepared['catalogue'],profiles['validation'],prepared['vectors'])
        transformed=bundle['scaler'].transform(fill(explained_raw,bundle['imputation_means']))
        contributions=transformed*bundle['classifier'].coef_[0]
        explanation=[]
        for i,aid in enumerate(explained_ids):
            strongest=np.argsort(np.abs(contributions[i]))[-5:][::-1]
            explanation.append({'article_id':int(aid),'text_cosine':float(explained_raw[i,len(REQUEST)+len(ARTICLE)]),
                'logit':float(contributions[i].sum()+bundle['classifier'].intercept_[0]),
                'largest_absolute_contributions':{feature_names[j]:float(contributions[i,j]) for j in strongest}})
        save_json(work/'validation_explanation.json',{'behaviour_id':sample[0],'top_five':explanation,
            'note':'Exact additive logit contributions, not causal feature importance.'})
        save_json(work/'model/model_config.json',{'family':'SGD logistic regression over frozen text similarity',
            'features':FEATURES,'missingness_columns':[c+'__missing' for c in FEATURES],
            'intercept':float(bundle['classifier'].intercept_[0]),'alpha':bundle['alpha'],
            'selected_epoch':bundle['epoch'],'recovery_checkpoint':None,'resumption_supported':False})
        record={'config':config,'export_schema_version':3,'prediction_pool':'time_eligible_catalogue',
            'shortlist_size':100,'final_display_size':5,'score_semantics':'logistic decision logit',
            'probability_semantics':'uncalibrated sigmoid click estimates trained on logged shown pairs; not catalogue-wide calibrated CTR',
            'eligibility_policy':POLICY,'feature_case':'raw','ordered_features':FEATURES,
            'removed_features':'All unlisted engineered columns, including pool-dependent slate_size and slate_relative_age_h; model retrained on declared subset.',
            'inputs':prepared['audit']['inputs'],'embedding_source':config.get('embedding_source','See shared embedding metadata and input manifest'),
            'embedding_dimensions':prepared['audit']['embedding_info']['dimension'],
            'frozen':['shared text encoder'],'trained':['train-only imputer/scaler','logistic classifier'],
            'train_predictions':'in-sample, not out-of-fold; not evidence of generalisation',
            'sample_scope':config.get('scope','development'),'input_audit':'input_audit.json',
            'history_coverage':history_counts,'export_cost':cost,'export_checks':checks,
            'all_agreed_requests_exported':all(config.get('export_requests',{}).get(s) is None for s in SPLITS),
            'python':sys.version,'packages':versions,'code_provenance':config.get('code_provenance',{}),
            'exports':{s:{'scores':f'predictions/{s}/submission_click_probabilities.csv',
                'shortlist':f'rankings/{s}/submission_top_100.csv',
                'recorded':f'diagnostics/recorded_slates/{s}_scores.csv',
                'eligibility':f'eligibility/{s}_requests.csv'} for s in SPLITS},
            'limitations':['Unshown outcomes unknown; retrieval metrics are observed-click recovery measures, not unbiased catalogue relevance.',
                'Publication-only policy and combined-snapshot version must match the group for cross-model comparison.',
                'Supplied article snapshots/encoder pretraining may postdate historical requests.',
                'Sampled development exports do not constitute complete final-split exports.']}
        save_json(work/'run_config.json',record)
        save_json(work/'metrics.json',{'validation':exported_validation if config.get('scope')=='final' else validation,
            'selection_validation':validation,'exported_validation':exported_validation,'recorded_slate_diagnostics':diagnostics,
            'selection':training['selected'],'test_metrics':None,'scope':config.get('scope','development')})
        record['handoff_complete']=False
        record['persistence']={'persistent_copy_complete':False,'reload_from_persistent_copy':False,'status':'not yet checked'}
        save_json(work/'run_config.json',record)
        try:persistence=persist_and_reload(prepared,bundle,probe,original)
        except BaseException as error:
            record['persistence'].update(status='failed or incomplete',error_type=type(error).__name__,message=str(error))
            save_json(work/'run_config.json',record)
            save_json(work/'reload_check.json',record['persistence'])
            raise

        record['persistence']=persistence
        record['handoff_complete']=persistence['persistent_copy_complete'] and record['all_agreed_requests_exported'] and config.get('scope')=='final'
        save_json(work/'run_config.json',record)
        if persistence['persistent_copy_complete']:
            target=Path(persistence['persistent_location'])
            shutil.copy2(work/'run_config.json',target/'run_config.json')
            save_json(target/'reload_check.json',persistence)
        save_json(work/'reload_check.json',persistence)
        return {'metrics':exported_validation if config.get('scope')=='final' else validation,
                'selection_validation':validation,'exported_validation':exported_validation,'diagnostics':diagnostics,'checks':checks,
                'persistence':persistence,'handoff_complete':record['handoff_complete']}
    finally:
        for p in profiles.values():p.close()


def load_trained_run(prepared, source_run):
    """Import an existing persistent selected model for a new load-only run."""
    source=Path(source_run);work=prepared['work']
    record=json.loads((source/'run_config.json').read_text())
    if record['ordered_features']!=FEATURES or record['eligibility_policy']!=POLICY:
        raise ValueError('Saved model feature/pool contract differs')
    if record['inputs']!=prepared['audit']['inputs']:
        raise ValueError('Shared source versions/paths changed since training; verify hashes and declare a separate migration')
    if (work/'model').exists():raise FileExistsError('Local model already exists')
    shutil.copytree(source/'model',work/'model')
    shutil.copy2(source/'training_summary.json',work/'training_summary.json')
    save_json(work/'imported_model.json',{'source_run':str(source.resolve()),'retrained':False})
    return joblib.load(work/'model/selected.joblib')
