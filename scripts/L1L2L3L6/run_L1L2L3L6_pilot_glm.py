#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1/L2/L3/L6 vocabulary-phenomenon Pilot with local GLM via vLLM.
Reads the provided Excel package, runs one text + one concept prompt per item,
extracts first-token probabilities for yes/no/uncertain, and writes audit files.
"""
import os
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import argparse, csv, json, math, hashlib, re, sys
from pathlib import Path
from datetime import datetime

EXPECTED_CONCEPTS = ['L1','L2','L3','L6']
CONDITION_ORDER = ['positive_use','neutral_mention','quote_critique','context_only']


def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def read_workbook(path):
    import openpyxl
    wb=openpyxl.load_workbook(path, data_only=True)
    if '概念定义与Prompt' not in wb.sheetnames or 'Pilot对照样本' not in wb.sheetnames:
        raise ValueError('Excel缺少必需工作表：概念定义与Prompt / Pilot对照样本')
    ps=wb['概念定义与Prompt']
    prompt_by_concept={}
    for row in ps.iter_rows(min_row=2, values_only=True):
        cid=row[0]
        prompt=row[5]
        if cid and prompt:
            prompt_by_concept[str(cid).strip()]=str(prompt).strip()
    ss=wb['Pilot对照样本']
    headers=[str(x.value).strip() if x.value is not None else '' for x in ss[1]]
    required=['sample_id','concept_id','condition','text']
    missing=[x for x in required if x not in headers]
    if missing: raise ValueError(f'样本表缺少列: {missing}; 当前列={headers}')
    ix={h:i for i,h in enumerate(headers)}
    samples=[]
    for row in ss.iter_rows(min_row=2, values_only=True):
        if not row[ix['sample_id']]: continue
        d={h:row[i] if i<len(row) else None for h,i in ix.items()}
        samples.append({k:(str(v).strip() if v is not None else '') for k,v in d.items()})
    if len(samples)!=32: raise ValueError(f'预期32条pilot样本，读取到{len(samples)}')
    if set(s['concept_id'] for s in samples)!=set(EXPECTED_CONCEPTS):
        raise ValueError('概念ID不符合预期，发现：'+str(sorted(set(s['concept_id'] for s in samples))))
    if len({s['sample_id'] for s in samples})!=len(samples): raise ValueError('sample_id重复')
    for cid in EXPECTED_CONCEPTS:
        ss_c=[s for s in samples if s['concept_id']==cid]
        if len(ss_c)!=8: raise ValueError(f'{cid}样本数应为8，实际{len(ss_c)}')
        counts={c:sum(x['condition']==c for x in ss_c) for c in CONDITION_ORDER}
        if any(v!=2 for v in counts.values()): raise ValueError(f'{cid}条件分布不符：{counts}')
    for cid in EXPECTED_CONCEPTS:
        if cid not in prompt_by_concept: raise ValueError(f'缺少{cid}概念Prompt')
    return samples,prompt_by_concept


def prompt_text(template, text, uncertain_word="不确定"):
    # Insert the sample before the concept question/rules; do not append after the answer instruction.
    marker = "判断问题："
    if marker not in template:
        raise ValueError("概念Prompt模板缺少‘判断问题：’标记，无法安全插入文本")
    template = template.replace("不确定", uncertain_word)
    return template.replace(marker, "给定文本：\n" + text + "\n\n" + marker, 1)


def get_token_ids(tokenizer, phrase):
    return tokenizer.encode(phrase, add_special_tokens=False)


def parse_logprob_candidates(lp_map, ids):
    mass=0.0
    for tid in ids:
        if tid in lp_map:
            mass += math.exp(lp_map[tid].logprob)
    return mass


def sampling_params_for(SamplingParams, allowed_ids):
    # Unconstrained generation: do NOT mask logits, otherwise candidate mass is biased.
    return SamplingParams(max_tokens=1, temperature=0, logprobs=50), 'unconstrained_next_token'


def main():
    ap=argparse.ArgumentParser(description='GLM local Pilot runner for L1/L2/L3/L6')
    ap.add_argument('--model_path',required=True,help='本地GLM权重目录')
    ap.add_argument('--pilot_xlsx',default=str(Path(__file__).with_name('ToxiCN_L1L2L3L6概念测量Pilot_实验包.xlsx')))
    ap.add_argument('--output_dir',default='L1L2L3L6_pilot_outputs')
    ap.add_argument('--gpu_memory_utilization',type=float,default=0.82)
    ap.add_argument('--max_model_len',type=int,default=2048)
    ap.add_argument('--max_num_seqs',type=int,default=16)
    ap.add_argument('--batch_size',type=int,default=32)
    ap.add_argument('--yes_token',default='是')
    ap.add_argument('--no_token',default='否')
    ap.add_argument('--uncertain_token',default='不确定',help='须为单token verbalizer；若 tokenizer 分词多段，改成单token词如“存疑”，并脚本会同步替换Prompt中的“不确定”')
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    if not 0<args.gpu_memory_utilization<1: raise ValueError('gpu_memory_utilization需在0和1之间')
    xlsx=Path(args.pilot_xlsx).resolve(); model=Path(args.model_path).resolve()
    if not xlsx.is_file(): raise FileNotFoundError(f'找不到Pilot Excel: {xlsx}')
    if not model.exists(): raise FileNotFoundError(f'找不到模型目录: {model}')
    samples,prompt_templates=read_workbook(xlsx)
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    files=[out/'pilot_probabilities.csv',out/'pilot_firsttoken_audit.csv',out/'pilot_condition_summary.csv',out/'pilot_run_meta.json']
    if not args.overwrite and any(p.exists() for p in files): raise FileExistsError('输出目录已有结果文件；如确认覆盖，请添加 --overwrite')

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok=AutoTokenizer.from_pretrained(str(model),trust_remote_code=True,local_files_only=True)
    verbalizers={'yes':args.yes_token,'no':args.no_token,'uncertain':args.uncertain_token}
    v_ids={k:get_token_ids(tok,v) for k,v in verbalizers.items()}
    multi={k:ids for k,ids in v_ids.items() if len(ids)!=1}
    if multi:
        print('Verbalizer tokenizer audit:', {k:{'text':verbalizers[k],'token_ids':ids,'decoded':[tok.decode([i]) for i in ids]} for k,ids in v_ids.items()},file=sys.stderr)
        raise ValueError('至少一个回答选项不是单token。请查看上面的tokenizer audit，使用 --uncertain_token 等参数换成单token verbalizer 后重试。')
    token_id_map={k:ids[0] for k,ids in v_ids.items()}
    if len(set(token_id_map.values()))!=3: raise ValueError('三种回答verbalizer对应同一token ID，需更换选项词')
    allowed=list(token_id_map.values())
    sampling,restrict_mode=sampling_params_for(SamplingParams,allowed)

    chat_template=getattr(tok,'chat_template',None) or ''
    chat_kwargs={'enable_thinking':False} if 'enable_thinking' in chat_template else {}
    llm=LLM(model=str(model),trust_remote_code=True,dtype='auto',gpu_memory_utilization=args.gpu_memory_utilization,
             enable_prefix_caching=True,max_model_len=args.max_model_len,max_num_seqs=args.max_num_seqs,
             max_num_batched_tokens=max(1024,args.max_model_len*min(args.max_num_seqs,8)),enforce_eager=True,max_logprobs=50)
    prompts=[]; metadata=[]
    for s in samples:
        raw=prompt_text(prompt_templates[s['concept_id']],s['text'],args.uncertain_token)
        msg=[{'role':'user','content':raw}]
        formatted=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,**chat_kwargs)
        if not formatted.endswith('\n'): formatted+='\n'
        prompts.append(formatted); metadata.append(s)

    # Process batches and preserve probability candidate mass (sum of three verbalizers).
    rows=[]; audit=[]
    for start in range(0,len(prompts),args.batch_size):
        end=min(start+args.batch_size,len(prompts))
        batch_out=llm.generate(prompts[start:end],sampling,use_tqdm=True)
        if len(batch_out)!=(end-start): raise RuntimeError('输出条数和输入batch不一致')
        for j,outobj in enumerate(batch_out,start=start):
            sample=metadata[j]
            if not outobj.outputs or not outobj.outputs[0].logprobs or not outobj.outputs[0].logprobs[0]:
                raise RuntimeError(f'缺失首token logprobs: {sample["sample_id"]}')
            lp=outobj.outputs[0].logprobs[0]
            rawmass={k:parse_logprob_candidates(lp,[tid]) for k,tid in token_id_map.items()}
            cand_mass=sum(rawmass.values())
            if cand_mass<=0: raise RuntimeError(f'候选token概率质量为0: {sample["sample_id"]}')
            # Conditional probability among the three allowed response tokens.
            cond={k:v/cand_mass for k,v in rawmass.items()}
            generated_ids=outobj.outputs[0].token_ids
            generated=tok.decode(generated_ids,skip_special_tokens=False) if generated_ids else ''
            row={
                'sample_id':sample['sample_id'],'concept_id':sample['concept_id'],'condition':sample['condition'],
                'p_yes_raw':rawmass['yes'],'p_no_raw':rawmass['no'],'p_uncertain_raw':rawmass['uncertain'],
                'candidate_mass_3':cand_mass,'p_yes_cond3':cond['yes'],'p_no_cond3':cond['no'],'p_uncertain_cond3':cond['uncertain'],
                'generated_first_token':generated,'text':sample['text']
            }
            rows.append(row)
            audit.append({**row,'prompt_sha256':hashlib.sha256(prompts[j].encode('utf-8')).hexdigest(),
                         'first_token_logprob_yes':lp.get(token_id_map['yes']).logprob if token_id_map['yes'] in lp else None,
                         'first_token_logprob_no':lp.get(token_id_map['no']).logprob if token_id_map['no'] in lp else None,
                         'first_token_logprob_uncertain':lp.get(token_id_map['uncertain']).logprob if token_id_map['uncertain'] in lp else None,
                         'verbalizer_yes':args.yes_token,'verbalizer_no':args.no_token,'verbalizer_uncertain':args.uncertain_token,
                         'token_id_yes':token_id_map['yes'],'token_id_no':token_id_map['no'],'token_id_uncertain':token_id_map['uncertain']})
        print(f'Completed {end}/{len(prompts)}')
    if len(rows)!=32: raise AssertionError(f'结果数不是32: {len(rows)}')
    # Condition summaries, descriptive only.
    summaries=[]
    for cid in EXPECTED_CONCEPTS:
        for cond in CONDITION_ORDER:
            subset=[r for r in rows if r['concept_id']==cid and r['condition']==cond]
            summaries.append({'concept_id':cid,'condition':cond,'n':len(subset),
                'mean_p_yes_cond3':sum(r['p_yes_cond3'] for r in subset)/len(subset),
                'mean_p_no_cond3':sum(r['p_no_cond3'] for r in subset)/len(subset),
                'mean_p_uncertain_cond3':sum(r['p_uncertain_cond3'] for r in subset)/len(subset),
                'mean_candidate_mass_3':sum(r['candidate_mass_3'] for r in subset)/len(subset)})
    def write_csv(path, data):
        if not data: return
        with open(path,'w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    write_csv(files[0],rows); write_csv(files[1],audit); write_csv(files[2],summaries)
    meta={'created_at':datetime.now().isoformat(timespec='seconds'),'model_path':str(model),'model_dir_name':model.name,
          'xlsx_path':str(xlsx),'xlsx_sha256':sha256_file(xlsx),'sample_count':len(samples),'concepts':EXPECTED_CONCEPTS,
          'condition_order':CONDITION_ORDER,'verbalizers':verbalizers,'token_ids':token_id_map,'token_restriction_mode':restrict_mode,
          'vllm_sampling':{'temperature':0,'max_tokens':1,'logprobs':50,'logits_unmasked':True},'gpu_memory_utilization':args.gpu_memory_utilization,
          'max_model_len':args.max_model_len,'max_num_seqs':args.max_num_seqs,'batch_size':args.batch_size,
          'notes':['Probabilities are first-token masses for selected verbalizer tokens.','p_*_raw are unnormalized token probabilities in full next-token distribution.',
                   'p_*_cond3 are normalized conditional probabilities over three candidate tokens only.',
                   'Candidate mass indicates how much full next-token probability is covered by these three verbalizers.',
                   'This is a constructed pilot stimulus set, not a gold-standard or corpus prevalence estimate.']}
    with open(files[3],'w',encoding='utf-8') as f: json.dump(meta,f,ensure_ascii=False,indent=2)
    print('\nPilot finished. Output files:')
    for p in files: print(' -',p)

if __name__=='__main__': main()
