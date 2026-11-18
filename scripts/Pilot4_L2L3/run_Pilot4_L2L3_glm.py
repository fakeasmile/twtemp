#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pilot4 local GLM first-token candidate probabilities. Runs locally only."""
import argparse,json,math,sys
from pathlib import Path
import pandas as pd

def parse_args():
 p=argparse.ArgumentParser(description='Pilot4 L2/L3 targeted construct validation')
 p.add_argument('--model_path',required=True); p.add_argument('--pilot_xlsx',required=True); p.add_argument('--output_dir',required=True)
 p.add_argument('--gpu_memory_utilization',type=float,default=.75); p.add_argument('--batch_size',type=int,default=4); p.add_argument('--max_model_len',type=int,default=2048)
 p.add_argument('--yes_token',default='是'); p.add_argument('--no_token',default='否'); p.add_argument('--uncertain_token',default='疑'); p.add_argument('--max_samples',type=int,default=0)
 return p.parse_args()

def main():
 a=parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 data=pd.read_excel(a.pilot_xlsx,sheet_name='材料与条件标签')
 required={'sample_id','concept_id','condition','text'}
 if not required.issubset(data.columns): raise RuntimeError(f'Missing columns: {required-set(data.columns)}')
 if data.sample_id.duplicated().any(): raise RuntimeError('Duplicate sample IDs')
 if len(data)!=144: raise RuntimeError(f'Expected 144 rows, got {len(data)}')
 counts=data.groupby(['concept_id','condition']).size()
 if len(counts)!=12 or not (counts==12).all(): raise RuntimeError('Expected 12 concept-condition cells with 12 samples each')
 if a.max_samples: data=data.head(a.max_samples)
 try:
  from transformers import AutoTokenizer
  from vllm import LLM,SamplingParams
 except Exception as e: raise RuntimeError('Install transformers, vllm, pandas, openpyxl; import error: '+str(e))
 tok=AutoTokenizer.from_pretrained(a.model_path,trust_remote_code=True)
 verbalizers=[a.yes_token,a.no_token,a.uncertain_token]; tokenized={v:tok.encode(v,add_special_tokens=False) for v in verbalizers}
 if any(len(ids)!=1 for ids in tokenized.values()) or len({ids[0] for ids in tokenized.values()})!=3:
  print('VERBALIZER TOKEN AUDIT FAILED:',tokenized,file=sys.stderr); raise RuntimeError('Verbalizers must be distinct single tokens; try --uncertain_token and re-audit')
 ids={v:tokenized[v][0] for v in verbalizers}
 def concept_def(c):
  if c=='L2': return '判断作者当前是否主动使用明确的语音、字形、拼写、拆字、缩写或替换形式，借变形后的词汇表达实施攻击/贬损。必须同时满足：变形机制明确；作者当前使用（非仅引用/讨论）；攻击性指向清晰。普通辱骂/俗语/反讽但无明确变形机制不算；中性提及、研究/报道、引用批评不算；一般攻击但未使用变体机制不算。'
  return '判断作者当前是否将身份标签/身份称谓用于贬低、羞辱、排斥或攻击身份群体/对象。必须判断作者立场与当前语用行为；不能仅因身份标签出现、讨论其贬义、引用攻击或反歧视批评就判是。一般行为/观点批评但无身份标签攻击不算；中性身份讨论、自我身份描述、引用批评和中性转述不算。'
 prompts=[]
 for r in data.itertuples():
  prompts.append(f'你是中文语用概念测量员。请严格按操作定义判断当前文本作者是否实施该概念，不判断词汇是否出现，也不判断文本是否仅讨论相关现象。\n概念：{r.concept_id}\n操作定义：{concept_def(r.concept_id)}\n输出要求：仅输出一个字符：{a.yes_token}、{a.no_token}、{a.uncertain_token}。{a.yes_token}=符合；{a.no_token}=不符合；{a.uncertain_token}=语境不足/有歧义。不要解释。\n文本：{r.text}\n回答：')
 llm=LLM(model=a.model_path,trust_remote_code=True,gpu_memory_utilization=a.gpu_memory_utilization,max_model_len=a.max_model_len,max_num_seqs=a.batch_size)
 sp=SamplingParams(temperature=0,max_tokens=1,logprobs=50)
 outputs=llm.generate(prompts,sp); result=[]; audit=[]
 for row,prompt,output in zip(data.itertuples(),prompts,outputs):
  lp=output.outputs[0].logprobs; first=output.outputs[0].token_ids[0] if output.outputs[0].token_ids else None; entries=[]
  if lp:
   for k,v in lp[0].items():
    tid=k if isinstance(k,int) else getattr(k,'token_id',None); logp=getattr(v,'logprob',None)
    if tid is not None and logp is not None: entries.append({'token':getattr(v,'decoded_token',str(k)),'token_id':int(tid),'logprob':float(logp)})
  raw={}
  for verbalizer,tid in ids.items():
   hits=[e['logprob'] for e in entries if e['token_id']==tid]; raw[verbalizer]=math.exp(hits[0]) if hits else None
  mass=sum(v for v in raw.values() if v is not None); norm={v:(x/mass if x is not None and mass>0 else None) for v,x in raw.items()}
  chosen=next((v for v,tid in ids.items() if tid==first),'OTHER'); topv=max((v for v,x in norm.items() if x is not None),key=lambda v:norm[v]) if mass>0 else None
  result.append({'sample_id':row.sample_id,'concept_id':row.concept_id,'condition':row.condition,'text':row.text,'p_yes_raw':raw[a.yes_token],'p_no_raw':raw[a.no_token],'p_uncertain_raw':raw[a.uncertain_token],'candidate_mass':mass,'p_yes_cond3':norm[a.yes_token],'p_no_cond3':norm[a.no_token],'p_uncertain_cond3':norm[a.uncertain_token],'first_token_id':first,'first_token_candidate':chosen,'candidate_argmax':topv,'argmax_matches_first':chosen==topv})
  audit.append({'sample_id':row.sample_id,'first_token_id':first,'first_token_candidate':chosen,'candidate_argmax':topv,'candidate_mass':mass,'candidate_logprobs':json.dumps(entries,ensure_ascii=False)})
 df=pd.DataFrame(result); df.to_csv(out/'pilot4_probabilities.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(audit).to_csv(out/'pilot4_firsttoken_audit.csv',index=False,encoding='utf-8-sig')
 summary=df.groupby(['concept_id','condition']).agg(n=('sample_id','count'),p_yes_mean=('p_yes_cond3','mean'),p_yes_median=('p_yes_cond3','median'),p_yes_sd=('p_yes_cond3','std'),yes_argmax_rate=('first_token_candidate',lambda s:(s==a.yes_token).mean()),uncertain_argmax_rate=('first_token_candidate',lambda s:(s==a.uncertain_token).mean()),candidate_mass_mean=('candidate_mass','mean'),argmax_consistency=('argmax_matches_first','mean')).reset_index()
 summary.to_csv(out/'pilot4_condition_summary.csv',index=False,encoding='utf-8-sig')
 meta={'model_path':a.model_path,'n_run':len(data),'verbalizers':verbalizers,'token_ids':ids,'tokenizer_audit':tokenized,'gpu_memory_utilization':a.gpu_memory_utilization,'batch_size':a.batch_size,'max_model_len':a.max_model_len}
 (out/'pilot4_run_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 print(f'Completed {len(df)} rows; outputs at {out}\n',summary.to_string(index=False))
if __name__=='__main__': main()
