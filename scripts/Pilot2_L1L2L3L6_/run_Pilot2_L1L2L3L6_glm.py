#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pilot2 local GLM measurement for L1/L2/L3/L6; reads supplied Excel and writes audited outputs."""
import os
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import argparse,csv,json,math,hashlib,sys
from pathlib import Path
from datetime import datetime
CONCEPTS=['L1','L2','L3','L6']
CONDITIONS=['positive_known','positive_variant','neutral_mention','self_or_behavior','quote_critique','quote_neutral','context_attack','context_nonattack']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read_xlsx(p):
 import openpyxl
 wb=openpyxl.load_workbook(p,data_only=True)
 ps=wb['概念定义与Prompt']; prompts={}
 for row in ps.iter_rows(min_row=2,values_only=True):
  if row[0] and row[5]:prompts[str(row[0]).strip()]=str(row[5]).strip()
 ss=wb['Pilot2对照样本']; headers=[str(c.value).strip() if c.value is not None else '' for c in ss[1]]; ix={h:i for i,h in enumerate(headers)}
 required=['sample_id','concept_id','condition','text']; assert all(k in ix for k in required),f'Missing columns {required}'
 samples=[]
 for r in ss.iter_rows(min_row=2,values_only=True):
  if r[ix['sample_id']]:samples.append({h:(str(r[i]).strip() if r[i] is not None else '') for h,i in ix.items()})
 if len(samples)!=128:raise ValueError(f'Expected 128 samples, got {len(samples)}')
 if len(set(s['sample_id'] for s in samples))!=len(samples):raise ValueError('Duplicate sample_id')
 for c in CONCEPTS:
  for cond in CONDITIONS:
   n=sum(s['concept_id']==c and s['condition']==cond for s in samples)
   if n!=4:raise ValueError(f'{c}/{cond}: expected 4, got {n}')
 if set(prompts)!=set(CONCEPTS):raise ValueError(f'Prompt concept IDs mismatch {prompts.keys()}')
 return samples,prompts
def insert_text(template,text,uncertain):
 template=template.replace('存疑',uncertain)
 marker='判断问题：'
 if marker not in template:raise ValueError('Prompt marker 判断问题： missing')
 return template.replace(marker,'给定文本：\n'+text+'\n\n'+marker,1)
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--model_path',required=True); ap.add_argument('--pilot_xlsx',default=str(Path(__file__).with_name('ToxiCN_Pilot2_L1L2L3L6_128条材料.xlsx')))
 ap.add_argument('--output_dir',default='Pilot2_L1L2L3L6_outputs'); ap.add_argument('--gpu_memory_utilization',type=float,default=.80)
 ap.add_argument('--max_model_len',type=int,default=2048); ap.add_argument('--max_num_seqs',type=int,default=12); ap.add_argument('--batch_size',type=int,default=16)
 ap.add_argument('--yes_token',default='是');ap.add_argument('--no_token',default='否');ap.add_argument('--uncertain_token',default='存疑');ap.add_argument('--overwrite',action='store_true');args=ap.parse_args()
 if not 0<args.gpu_memory_utilization<1:raise ValueError('gpu_memory_utilization must be between 0 and 1')
 model=Path(args.model_path).resolve(); xlsx=Path(args.pilot_xlsx).resolve()
 if not model.exists():raise FileNotFoundError(model)
 if not xlsx.exists():raise FileNotFoundError(xlsx)
 samples,prompts=read_xlsx(xlsx);out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
 outputs=[out/'pilot2_probabilities.csv',out/'pilot2_firsttoken_audit.csv',out/'pilot2_condition_summary.csv',out/'pilot2_run_meta.json']
 if not args.overwrite and any(p.exists() for p in outputs):raise FileExistsError('Output files exist; use --overwrite to rerun')
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 tok=AutoTokenizer.from_pretrained(str(model),trust_remote_code=True,local_files_only=True)
 verbal={'yes':args.yes_token,'no':args.no_token,'uncertain':args.uncertain_token};ids={k:tok.encode(v,add_special_tokens=False) for k,v in verbal.items()}
 print('Verbalizer audit:',{k:{'phrase':verbal[k],'ids':ids[k],'decoded':[tok.decode([i]) for i in ids[k]]} for k in ids})
 if any(len(v)!=1 for v in ids.values()):raise RuntimeError('All 3 verbalizers must be single tokens; change --uncertain_token if needed')
 token={k:v[0] for k,v in ids.items()}
 if len(set(token.values()))<3:raise RuntimeError('Verbalizer tokens must be distinct')
 sampling=SamplingParams(max_tokens=1,temperature=0,logprobs=50)
 template=getattr(tok,'chat_template',None) or ''; kwargs={'enable_thinking':False} if 'enable_thinking' in template else {}
 llm=LLM(model=str(model),trust_remote_code=True,dtype='auto',gpu_memory_utilization=args.gpu_memory_utilization,enable_prefix_caching=True,max_model_len=args.max_model_len,max_num_seqs=args.max_num_seqs,max_num_batched_tokens=max(1024,args.max_model_len*min(args.max_num_seqs,8)),enforce_eager=True,max_logprobs=50)
 prompts_in=[]
 for s in samples:
  raw=insert_text(prompts[s['concept_id']],s['text'],args.uncertain_token)
  formatted=tok.apply_chat_template([{'role':'user','content':raw}],tokenize=False,add_generation_prompt=True,**kwargs)
  if not formatted.endswith('\n'):formatted+='\n'
  prompts_in.append(formatted)
 rows=[];aud=[]
 for start in range(0,len(prompts_in),args.batch_size):
  end=min(start+args.batch_size,len(prompts_in)); outs=llm.generate(prompts_in[start:end],sampling,use_tqdm=True)
  if len(outs)!=end-start:raise RuntimeError('batch output size mismatch')
  for j,obj in enumerate(outs,start=start):
   sample=samples[j]
   if not obj.outputs or not obj.outputs[0].logprobs or not obj.outputs[0].logprobs[0]:raise RuntimeError(f'Missing logprobs {sample["sample_id"]}')
   lp=obj.outputs[0].logprobs[0]; raw={k:(math.exp(lp[token[k]].logprob) if token[k] in lp else 0.0) for k in token}; mass=sum(raw.values())
   if mass<=0:raise RuntimeError('candidate mass zero '+sample['sample_id'])
   cond={k:v/mass for k,v in raw.items()}; gen=obj.outputs[0].token_ids; gen_text=tok.decode(gen,skip_special_tokens=False) if gen else ''
   r={**{k:sample.get(k,'') for k in ['sample_id','concept_id','condition','condition_description','expected_phenomenon','text']},'p_yes_raw':raw['yes'],'p_no_raw':raw['no'],'p_uncertain_raw':raw['uncertain'],'candidate_mass_3':mass,'p_yes_cond3':cond['yes'],'p_no_cond3':cond['no'],'p_uncertain_cond3':cond['uncertain'],'generated_first_token':gen_text}
   rows.append(r);aud.append({**r,'prompt_sha256':hashlib.sha256(prompts_in[j].encode()).hexdigest(),'token_id_yes':token['yes'],'token_id_no':token['no'],'token_id_uncertain':token['uncertain'],'verbalizer_yes':args.yes_token,'verbalizer_no':args.no_token,'verbalizer_uncertain':args.uncertain_token,'lp_yes':lp[token['yes']].logprob if token['yes'] in lp else None,'lp_no':lp[token['no']].logprob if token['no'] in lp else None,'lp_uncertain':lp[token['uncertain']].logprob if token['uncertain'] in lp else None})
  print(f'Completed {end}/{len(prompts_in)}')
 def write(path,data):
  with open(path,'w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 write(outputs[0],rows);write(outputs[1],aud)
 summ=[]
 for c in CONCEPTS:
  for cond in CONDITIONS:
   ss=[r for r in rows if r['concept_id']==c and r['condition']==cond]
   summ.append({'concept_id':c,'condition':cond,'n':len(ss),'mean_p_yes_cond3':sum(r['p_yes_cond3'] for r in ss)/len(ss),'mean_p_no_cond3':sum(r['p_no_cond3'] for r in ss)/len(ss),'mean_p_uncertain_cond3':sum(r['p_uncertain_cond3'] for r in ss)/len(ss),'mean_candidate_mass_3':sum(r['candidate_mass_3'] for r in ss)/len(ss)})
 write(outputs[2],summ)
 meta={'run_time':datetime.now().isoformat(timespec='seconds'),'model_path':str(model),'model_name':model.name,'input_xlsx':str(xlsx),'input_xlsx_sha256':sha(xlsx),'samples':len(samples),'concepts':CONCEPTS,'conditions':CONDITIONS,'token_ids':token,'verbalizers':verbal,'generation':{'temperature':0,'max_tokens':1,'logprobs':50,'unconstrained':True},'notes':['p_raw are probability masses from full next-token distribution','p_cond3 normalized over three candidate verbalizers','candidate mass is sum of raw masses for three verbalizers','constructed Pilot stimuli, condition labels are design tags not ground truth']}
 outputs[3].write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Finished; outputs:');[print(p) for p in outputs]
if __name__=='__main__':main()
