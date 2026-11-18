#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pilot3 local GLM inference for L2/L3 construct validation. Never run model remotely."""
import argparse, json, re, sys
from pathlib import Path
import pandas as pd

def parse_args():
 p=argparse.ArgumentParser()
 p.add_argument('--model_path',required=True)
 p.add_argument('--pilot_xlsx',required=True)
 p.add_argument('--output_dir',required=True)
 p.add_argument('--gpu_memory_utilization',type=float,default=.78)
 p.add_argument('--batch_size',type=int,default=8)
 p.add_argument('--max_model_len',type=int,default=2048)
 p.add_argument('--yes_token',default='是'); p.add_argument('--no_token',default='否'); p.add_argument('--uncertain_token',default='疑')
 p.add_argument('--max_samples',type=int,default=0,help='0=all; use only for smoke test')
 return p.parse_args()

def main():
 a=parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 data=pd.read_excel(a.pilot_xlsx,sheet_name='材料与条件标签')
 required={'sample_id','concept_id','condition','text'}
 if not required.issubset(data.columns): raise RuntimeError('Workbook columns missing: '+str(required-set(data.columns)))
 if data.sample_id.duplicated().any(): raise RuntimeError('Duplicate sample IDs detected')
 if len(data)!=144: raise RuntimeError(f'Expected 144 samples, found {len(data)}')
 counts=data.groupby(['concept_id','condition']).size()
 if len(counts)!=12 or not (counts==12).all(): raise RuntimeError('Condition counts invalid; expected 12 per concept-condition cell')
 if a.max_samples: data=data.head(a.max_samples)
 try:
  from transformers import AutoTokenizer
  from vllm import LLM, SamplingParams
 except Exception as e: raise RuntimeError('Install requirements: transformers, vllm, pandas, openpyxl. Import failed: '+str(e))
 tok=AutoTokenizer.from_pretrained(a.model_path,trust_remote_code=True)
 verbalizers=[a.yes_token,a.no_token,a.uncertain_token]
 tokenized={v:tok.encode(v,add_special_tokens=False) for v in verbalizers}
 if any(len(x)!=1 for x in tokenized.values()) or len({x[0] for x in tokenized.values()})!=3:
  print('TOKEN AUDIT FAILED:',tokenized,file=sys.stderr); raise RuntimeError('Verbalizers must be distinct single tokens. Change --uncertain_token and re-check.')
 v_ids={v:tokenized[v][0] for v in verbalizers}
 prompts=[]
 for r in data.itertuples():
  if r.concept_id=='L2': definition='判断文本作者是否在当前语境中使用谐音、拼写替换、拆字、缩写或其他词汇变形作为攻击/贬损表达。只出现变体词、讨论其含义、引用批评、词典/研究/中性语境，不算作者当前实施变体攻击。一般攻击但未使用词汇变体，应判否。'
  else: definition='判断文本作者是否在当前语境中把身份标签用于贬低、羞辱、排斥或攻击他人/群体。中性提及、自我身份描述、引用批评、科普或研究讨论不算当前作者实施身份攻击。一般辱骂但未使用身份标签，应判否。'
  prompts.append(f'你是中文语用概念标注员。根据以下定义判断文本中概念是否成立。\n概念定义：{definition}\n请只输出一个字符：{a.yes_token}、{a.no_token}、{a.uncertain_token}。{a.yes_token}=满足定义；{a.no_token}=不满足定义；{a.uncertain_token}=信息不足或无法确定。不要解释。\n文本：{r.text}\n回答：')
 llm=LLM(model=a.model_path,trust_remote_code=True,gpu_memory_utilization=a.gpu_memory_utilization,max_model_len=a.max_model_len,max_num_seqs=a.batch_size,max_logprobs=50)
 sp=SamplingParams(temperature=0,max_tokens=1,logprobs=50)
 outputs=llm.generate(prompts,sp)
 result=[]; audit=[]
 for row,prompt,outp in zip(data.itertuples(),prompts,outputs):
  lp=outp.outputs[0].logprobs
  first=outp.outputs[0].token_ids[0] if outp.outputs[0].token_ids else None
  raw={}; entries=[]
  if lp:
   for key,val in lp[0].items():
    # vLLM logprobs mapping is token_id -> Logprob object (key is often int).
    tid=key if isinstance(key,int) else getattr(key,'token_id',None)
    logp=getattr(val,'logprob',None)
    if tid is None or logp is None: continue
    entries.append({'token':getattr(val,'decoded_token',str(key)),'token_id':int(tid),'logprob':float(logp)})
  import math
  for v,tid in v_ids.items():
   match=[x['logprob'] for x in entries if x['token_id']==tid]
   raw[v]=math.exp(match[0]) if match else None
  mass=sum(x for x in raw.values() if x is not None)
  norm={v:(x/mass if x is not None and mass>0 else None) for v,x in raw.items()}
  chosen=next((v for v,tid in v_ids.items() if tid==first),'OTHER')
  topv=max((v for v,x in norm.items() if x is not None),key=lambda v:norm[v]) if mass>0 else None
  result.append({'sample_id':row.sample_id,'concept_id':row.concept_id,'condition':row.condition,'text':row.text,
   'p_yes_raw':raw[a.yes_token],'p_no_raw':raw[a.no_token],'p_uncertain_raw':raw[a.uncertain_token],
   'candidate_mass':mass,'p_yes_cond3':norm[a.yes_token],'p_no_cond3':norm[a.no_token],'p_uncertain_cond3':norm[a.uncertain_token],
   'first_token_id':first,'first_token_candidate':chosen,'candidate_argmax':topv,'argmax_matches_first':chosen==topv})
  audit.append({'sample_id':row.sample_id,'first_token_id':first,'first_token_candidate':chosen,'candidate_argmax':topv,'candidate_mass':mass,'candidate_logprobs':json.dumps(entries,ensure_ascii=False)})
 df=pd.DataFrame(result); df.to_csv(out/'pilot3_probabilities.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(audit).to_csv(out/'pilot3_firsttoken_audit.csv',index=False,encoding='utf-8-sig')
 summary=df.groupby(['concept_id','condition']).agg(n=('sample_id','count'),p_yes_mean=('p_yes_cond3','mean'),p_yes_median=('p_yes_cond3','median'),p_yes_sd=('p_yes_cond3','std'),yes_argmax_rate=('first_token_candidate',lambda s:(s==a.yes_token).mean()),uncertain_argmax_rate=('first_token_candidate',lambda s:(s==a.uncertain_token).mean()),candidate_mass_mean=('candidate_mass','mean'),argmax_consistency=('argmax_matches_first','mean')).reset_index()
 summary.to_csv(out/'pilot3_condition_summary.csv',index=False,encoding='utf-8-sig')
 meta={'model_path':a.model_path,'n_run':len(data),'verbalizers':verbalizers,'token_ids':v_ids,'tokenizer_audit':tokenized,'gpu_memory_utilization':a.gpu_memory_utilization,'batch_size':a.batch_size,'max_model_len':a.max_model_len}
 (out/'pilot3_run_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Completed:',len(df),'rows. Results in',out)
 print(summary.to_string(index=False))
if __name__=='__main__': main()
