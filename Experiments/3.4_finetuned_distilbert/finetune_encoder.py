"""Fine-tune a shared DistilBERT sentence encoder for history-conditioned retrieval.

The shared encoder receives gradients from BOTH history and candidate article text.
There is no added recommendation projection layer and no test-label training.
"""
from contextlib import closing, nullcontext
import hashlib
import heapq
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = 'distilbert_click_finetuning_v1'


def text_for_article(row):
    fields=[]
    for field in ['title','subtitle']:
        value=row.get(field)
        if isinstance(value,str) and value.strip():fields.append(f'{field}: {value.strip()}')
    return '\n'.join(fields) or None


def resolve(config,key,relative):
    value=config.get('paths',{}).get(key,relative)
    path=Path(value)
    if not path.is_absolute():path=Path(config['drive_root'])/path
    if not path.is_file():raise FileNotFoundError(path)
    return path


def history_ids(path,users,cutoff,texts,limit):
    result={};seen=set();counts={'selected_events':0,'events_with_text':0,'missing_history_users':0,'users_without_usable_history':0}
    for batch in pq.ParquetFile(path).iter_batches(columns=['user_id','article_id_fixed','impression_time_fixed'],batch_size=128):
        for row in batch.to_pylist():
            user=int(row['user_id'])
            if user not in users:continue
            if user in seen:raise ValueError('Duplicate history user')
            seen.add(user);ids=row['article_id_fixed'];times=row['impression_time_fixed']
            if ids is None or times is None or len(ids)!=len(times):raise ValueError('Misaligned histories')
            events=[]
            for position,(aid,stamp) in enumerate(zip(ids,times)):
                if stamp is None:continue
                if pd.Timestamp(stamp)>=pd.Timestamp(cutoff):raise ValueError('History reaches the behaviour split')
                if aid is None:raise ValueError('Missing historical article ID')
                events.append((pd.Timestamp(stamp),position,int(aid)))
            selected=heapq.nlargest(limit,events)
            usable=[aid for _,_,aid in selected if texts.get(aid)]
            counts['selected_events']+=len(selected);counts['events_with_text']+=len(usable)
            result[user]=usable
    counts['missing_history_users']=len(users-seen)
    counts['users_without_usable_history']=sum(not result.get(user) for user in users)
    return result,counts


def prepare(config,work,cp):
    """Read only train/validation behaviour labels, with unchanged catalogue eligibility."""
    config=json.loads(json.dumps(config,default=str))
    for field in ['train_requests','validation_requests','history_limit','negatives_per_positive']:
        if type(config[field]) is not int or config[field]<1:raise ValueError(f'{field} must be positive')
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    sources={}
    for part in ['train_val','test']:
        sources[f'catalogue_{part}']=resolve(config,f'catalogue_{part}',f'{part}/articles.parquet')
    for split in ['train','validation']:
        sources[f'features_{split}']=resolve(config,f'features_{split}',f'feature_engineering/{split}_features.parquet')
        sources[f'history_{split}']=resolve(config,f'history_{split}',f'train_val/{split}/history.parquet')
    for i,path in enumerate(config.get('additional_text_catalogues',[])):
        candidate=Path(path)
        if not candidate.is_absolute():candidate=Path(config['drive_root'])/candidate
        if not candidate.is_file():raise FileNotFoundError(candidate)
        sources[f'additional_text_{i}']=candidate
    audit={'version':VERSION,'config':config,'sources':{key:cp.fingerprint(path,config.get('hash_inputs',False)) for key,path in sources.items()}}
    state=work/'finetuning_inputs.json'
    if state.exists() and json.loads(state.read_text())!=audit:raise ValueError('Sources/config changed; use a new run ID')
    cp.save_json(state,audit)
    texts={}
    # First catalogue snapshot wins. Additional sources only fill previously absent IDs.
    for key,path in sources.items():
        if not (key.startswith('catalogue_') or key.startswith('additional_text_')):continue
        seen=set()
        for batch in pq.ParquetFile(path).iter_batches(columns=['article_id','title','subtitle'],batch_size=512):
            for row in batch.to_pylist():
                aid=int(row['article_id'])
                if aid in seen:raise ValueError(f'Duplicate article ID in {path}')
                seen.add(aid)
                if aid not in texts:texts[aid]=text_for_article(row)
    catalogue,_=cp.load_catalogue([sources['catalogue_train_val'],sources['catalogue_test']])
    estimate=sum(audit['sources'][f'features_{s}']['rows'] for s in ['train','validation'])*320
    if estimate>config.get('max_index_gb',20)*1e9 or estimate>shutil.disk_usage(work).free*.7:
        raise RuntimeError('Training/validation index estimate exceeds disk budget')
    index=work/'indexes';index.mkdir(exist_ok=True);dbs={};index_info={}
    for split in ['train','validation']:
        dbs[split]=index/f'{split}.sqlite';metadata=index/f'{split}.json'
        if metadata.exists():index_info[split]=json.loads(metadata.read_text())
        else:
            if dbs[split].exists():dbs[split].unlink()
            print('Indexing:',split,flush=True)
            index_info[split]=cp.index_shared(sources[f'features_{split}'],split,dbs[split],config['seed'])
            cp.save_json(metadata,index_info[split])
    with closing(sqlite3.connect(dbs['train'])) as db:
        if db.execute('SELECT max(time) FROM requests').fetchone()[0]>=index_info['validation']['split_start']:
            raise ValueError('Training and validation periods overlap')
    requests={split:list(cp.request_iter(dbs[split],config[f'{split}_requests'])) for split in ['train','validation']}
    histories={};coverage={}
    for split in requests:
        histories[split],coverage[split]=history_ids(sources[f'history_{split}'],{int(r[1]) for r in requests[split]},
            index_info[split]['split_start'],texts,config['history_limit'])
    pub=dict(zip(catalogue.index,catalogue.published_time))
    def usable(aid,when):
        stamp=pub.get(aid)
        return bool(texts.get(aid)) and stamp is not None and not pd.isna(stamp) and stamp<=pd.Timestamp(when)
    examples=[];skipped={'no_history':0,'no_positive':0,'no_usable_negative':0,'positive_missing_text_or_ineligible':0}
    with closing(sqlite3.connect(dbs['train'])) as db:
        for req in requests['train']:
            key,user,_,when,_=req
            rows=db.execute('SELECT article_id,clicked FROM candidates WHERE behaviour_id=? ORDER BY article_id',(key,)).fetchall()
            positive={int(a) for a,label in rows if label==1}
            if not positive:skipped['no_positive']+=1;continue
            if not histories['train'].get(int(user)):skipped['no_history']+=1;continue
            negative=[int(a) for a,label in rows if label==0 and a not in positive and usable(int(a),when)]
            if not negative:skipped['no_usable_negative']+=1;continue
            for aid in sorted(positive):
                if not usable(aid,when):skipped['positive_missing_text_or_ineligible']+=1;continue
                def priority(other):
                    return hashlib.blake2b(f'{config["seed"]}:{key}:{aid}:{other}'.encode(),digest_size=8).digest(),other
                sampled=heapq.nsmallest(config['negatives_per_positive'],negative,key=priority)
                examples.append({'behaviour_id':key,'user_id':int(user),'positive_article_id':aid,'negative_article_ids':sampled})
    if not examples:raise ValueError('No usable training examples; check histories, text coverage and recorded slates')
    pq.write_table(pa.Table.from_pylist(examples),work/'finetuning_examples.parquet')
    summary={'training_examples':len(examples),'training_requests_selected':len(requests['train']),
        'validation_requests_selected':len(requests['validation']),'history_coverage':coverage,'skipped':skipped,
        'catalogue_articles':len(catalogue),'text_lookup_articles':len(texts),'text_variant':'title_subtitle',
        'validation_request_ids':[r[0] for r in requests['validation']],
        'test_behaviour_labels_read':False,'eligibility_policy':cp.POLICY}
    cp.save_json(work/'dataset_audit.json',summary)
    return dict(config=config,work=work,sources=sources,dbs=dbs,index_info=index_info,texts=texts,
                catalogue=catalogue,requests=requests,histories=histories,examples=examples,audit=summary)


def encode_articles(model,texts,chunk_batch=16):
    """Differentiable whole-text chunk pooling. Do NOT use model.encode() in training."""
    import torch
    if chunk_batch<1:raise ValueError('chunk_batch must be positive')
    tokenizer=model.tokenizer;pending=[];parts=[[] for _ in texts];weights=[[] for _ in texts]
    def flush():
        if not pending:return
        features={}
        for name in tokenizer.model_input_names:
            if name in pending[0][2]:
                pad=(getattr(tokenizer,'pad_token_id',0) or 0) if name=='input_ids' else 0
                features[name]=torch.nn.utils.rnn.pad_sequence([r[2][name] for r in pending],batch_first=True,padding_value=pad).to(model.device)
        output=torch.nn.functional.normalize(model(features)['sentence_embedding'].float(),dim=1)
        for (owner,weight,_),vector in zip(pending,output):
            parts[owner].append(vector);weights[owner].append(weight)
        pending.clear()
    for owner,text in enumerate(texts):
        if not text:continue
        chunks=tokenizer(text,truncation=True,max_length=model.max_seq_length,stride=0,
                         return_overflowing_tokens=True,padding=True,return_special_tokens_mask=True,return_tensors='pt')
        count=(chunks['attention_mask']*(1-chunks['special_tokens_mask'])).sum(1)
        for i,weight in enumerate(count.tolist()):
            if weight<=0:continue
            length=int(chunks['attention_mask'][i].sum())
            features={name:chunks[name][i,:length] for name in tokenizer.model_input_names if name in chunks}
            pending.append((owner,weight,features))
            if len(pending)>=chunk_batch:flush()
    flush();result=[]
    for text,rows,counts in zip(texts,parts,weights):
        if not text:result.append(None);continue
        if not rows:raise ValueError('Nonempty article produced no usable chunks')
        matrix=torch.stack(rows);w=matrix.new_tensor(counts)
        total=(matrix*w[:,None]).sum(0)
        if not bool(torch.isfinite(total).all()) or float(total.detach().norm())<=1e-12:
            raise ValueError('Encoder produced invalid or zero article embedding')
        result.append(torch.nn.functional.normalize(total,dim=0))
    return result


def configure_trainable(model,last_layers=2):
    """Unfreeze actual pretrained transformer blocks, plus the existing dense encoder module."""
    transformer=model[0].auto_model
    if getattr(transformer.config,'model_type',None)!='distilbert':
        raise ValueError('This notebook specifically fine-tunes the DistilBERT sentence encoder')
    if last_layers is None:
        for parameter in model.parameters():parameter.requires_grad=True
    else:
        layers=transformer.transformer.layer
        if type(last_layers) is not int or not 1<=last_layers<=len(layers):raise ValueError('Invalid transformer layer count')
        for parameter in model.parameters():parameter.requires_grad=False
        for block in layers[-last_layers:]:
            for parameter in block.parameters():parameter.requires_grad=True
        # These are existing pretrained encoder weights, not newly added user/article towers.
        for module in list(model)[1:]:
            for parameter in module.parameters():parameter.requires_grad=True
    return {'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_parameters':sum(p.numel() for p in model.parameters()),'last_transformer_layers':last_layers}


def request_loss(model,batch,histories,text_lookup,temperature=.05,chunk_batch=16):
    import torch
    if temperature<=0:raise ValueError('temperature must be positive')
    needed=set()
    for example in batch:
        needed.update(histories[example['user_id']]);needed.add(example['positive_article_id']);needed.update(example['negative_article_ids'])
    ids=sorted(needed)
    vectors=dict(zip(ids,encode_articles(model,[text_lookup[a] for a in ids],chunk_batch)))
    losses=[]
    for example in batch:
        profile=torch.nn.functional.normalize(torch.stack([vectors[a] for a in histories[example['user_id']]]).mean(0),dim=0)
        candidates=[example['positive_article_id']]+example['negative_article_ids']
        scores=torch.stack([profile@vectors[a] for a in candidates])/temperature
        # Actual candidate count is used: no padding or artificial negative articles.
        losses.append(torch.nn.functional.cross_entropy(scores[None,:],torch.zeros(1,dtype=torch.long,device=scores.device)))
    return torch.stack(losses).mean()


def train_epoch(model,optimizer,prepared,epoch):
    import torch
    config=prepared['config'];batch_size=config['batch_size'];accum=config['gradient_accumulation']
    if batch_size<1 or accum<1:raise ValueError('Training batch sizes must be positive')
    use_amp=bool(config.get('amp',True) and model.device.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    model.train();order=np.random.default_rng(config['seed']+epoch).permutation(len(prepared['examples']))
    size=batch_size*accum;total=0.;seen=0;started=time.perf_counter();updates=0
    for start in range(0,len(order),size):
        window=order[start:start+size];optimizer.zero_grad(set_to_none=True)
        for offset in range(0,len(window),batch_size):
            selected=window[offset:offset+batch_size]
            batch=[prepared['examples'][int(i)] for i in selected]
            context=torch.autocast(device_type='cuda',dtype=torch.float16) if use_amp else nullcontext()
            with context:
                loss=request_loss(model,batch,prepared['histories']['train'],prepared['texts'],
                    config['temperature'],config['chunk_batch_size'])
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite fine-tuning loss')
            scaler.scale(loss*len(batch)/len(window)).backward()
            total+=float(loss.detach())*len(batch);seen+=len(batch)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        scaler.step(optimizer);scaler.update();updates+=1
        if updates%25==0:print(f'Epoch {epoch}: {seen:,}/{len(order):,} examples; loss={total/seen:.4f}',flush=True)
    return {'epoch':epoch,'training_loss':total/seen,'examples':seen,'optimizer_updates':updates,'seconds':time.perf_counter()-started}


def evaluate_checkpoint(model,prepared,name,cp):
    """Regenerate the SAME encoder's candidate and history vectors, then score full pools."""
    import torch
    config=prepared['config'];out=prepared['work']/'checkpoints'/name
    if out.exists():raise FileExistsError(f'{out} exists; use a fresh run or load its completed metrics')
    out.mkdir(parents=True);model.eval();model.save(str(out/'encoder'))
    vectors={};ids=sorted(prepared['texts']);started=time.perf_counter()
    schema=pa.schema([('article_id',pa.int64()),('text_embedding',pa.list_(pa.float32())),('has_text_embedding',pa.bool_())])
    with pq.ParquetWriter(out/'text_embeddings.parquet',schema) as writer,torch.inference_mode():
        for start in range(0,len(ids),config.get('encoding_article_batch',64)):
            block=ids[start:start+config.get('encoding_article_batch',64)]
            encoded=encode_articles(model,[prepared['texts'][a] for a in block],config['chunk_batch_size'])
            rows=[]
            for aid,vector in zip(block,encoded):
                value=vector.float().cpu().numpy() if vector is not None else None
                if value is not None:vectors[aid]=value
                rows.append({'article_id':aid,'text_embedding':value.tolist() if value is not None else None,'has_text_embedding':value is not None})
            writer.write_table(pa.Table.from_pylist(rows,schema=schema))
            if start%2048==0:print(f'{name}: encoded {min(start+len(block),len(ids)):,}/{len(ids):,} article texts',flush=True)
    encoding_seconds=time.perf_counter()-started
    profiles=cp.HistoryProfiles(aggregation='mean_profile')
    for user,history in prepared['histories']['validation'].items():
        available=[vectors[a] for a in history if a in vectors]
        summed=np.asarray(available,dtype='float64').sum(axis=0) if available else np.zeros(model.get_sentence_embedding_dimension())
        norm=np.linalg.norm(summed)
        profiles[user]=((summed/norm).astype('float32') if norm>1e-8 else None,len(available)/max(len(history),1))
    lookup=cp.SimilarityVectors(vectors,config.get('similarity_device','auto'),256)
    specification=cp.model_spec({'history_limit':config['history_limit'],'history_aggregation':'mean_profile'})
    started=time.perf_counter()
    metrics=cp.evaluate_catalogue(specification,prepared['requests']['validation'],prepared['dbs']['validation'],
        prepared['catalogue'],profiles,lookup,config.get('catalogue_batch_size',4096))
    result={'name':name,'metrics':metrics,'encoding_seconds':encoding_seconds,'validation_seconds':time.perf_counter()-started,
            'encoder_path':str(out/'encoder'),'embeddings_path':str(out/'text_embeddings.parquet'),
            'frozen_baseline':name=='frozen','history_aggregation':'mean_profile',
            'validation_request_ids':prepared['audit']['validation_request_ids'],'compute':lookup.info}
    cp.save_json(out/'validation_metrics.json',result)
    return result


def select_checkpoint(results):
    # Original order preserves the frozen baseline/earlier epoch in exact metric ties.
    return max(results,key=lambda r:(r['metrics']['known_click_recall@200'],r['metrics']['observed_click_ndcg@200']))


def save_selected(prepared,results,training_log,cp):
    selected=select_checkpoint(results);work=prepared['work']
    record={'version':VERSION,'config':prepared['config'],'selected':selected,'training':training_log,
            'comparisons':results,'test_labels_used':False,'encoder_was_finetuned':bool(training_log),
            'selected_is_frozen_baseline':selected['frozen_baseline'],
            'selection':'maximum validation Recall@200, then nDCG@200, preserve earlier result in exact ties',
            'eligibility_policy':cp.POLICY,'text_variant':'title_subtitle',
            'scope':'development; matched frozen/finetuned comparison on fixed validation requests'}
    cp.save_json(work/'finetuning_summary.json',record)
    return record
