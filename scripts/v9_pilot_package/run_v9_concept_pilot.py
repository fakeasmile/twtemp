#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v9 概念测量试点：GLM-4-9B-Chat + vLLM，固定 512 样本 manifest。

功能：
- 读取 v9_pilot_512_manifest.csv，严格按 train_index 从 train.json 取文本；
- 对 D1-04/B8-05 原始与修订问句分别运行 P0/P1/P2；
- 对 12 个 v9 候选概念运行 P2；
- 同一 manifest 固定复用所有条件，输出宽矩阵、首 token 审计和运行元数据。

重要：矩阵保存原始 p_yes，不对反向概念执行 1-p。p_yes 是“是/否”候选概率质量归一化值；
cand_mass 单独保存，便于审计候选 token 在完整首 token 分布中的覆盖率。
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path


PROMPTS = {
    "v7_global": (
        "你是语言学标注助手。判断文本是否符合下述问句。只依据文本本身判断，不评价内容对错，不解释。"
        "问句指出的现象以讽刺、委婉、谐音、缩写等变形出现同样算。\n\n"
        "文本：{text}\n\n问句：{question}\n\n只回答‘是’或‘否’。回答："
    ),
    "p1_no_global": (
        "你是语言学标注助手。判断文本是否符合下述问句。只依据文本本身判断，不评价内容对错，不解释。\n\n"
        "文本：{text}\n\n问句：{question}\n\n只回答‘是’或‘否’。回答："
    ),
    "p2_local_boundary": (
        "你是语言学标注助手。请判断文本是否符合问句所描述的具体现象。"
        "只依据文本及其语境线索判断，不评价内容对错，不解释。"
        "只有当文本满足该问句的核心定义时才回答‘是’；仅出现相关话题、身份标签、情绪词、相似词语或相关表达形式，不足以回答‘是’。"
        "若问句列出排除条件，必须遵守；不能把其他概念的现象自动算作本问句现象。\n\n"
        "文本：{text}\n\n问句：{question}\n\n只回答‘是’或‘否’。回答："
    ),
}
MODEL_SUFFIX = {"glm-4-9b-chat": "\n"}


def sha256_text(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def load_manifest(path, raw):
    records = []
    seen = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "train_index", "content_sha256"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"manifest 缺少列：{sorted(required-set(reader.fieldnames or []))}")
        for row in reader:
            sid = row["sample_id"].strip()
            idx = int(row["train_index"])
            if sid in seen:
                raise ValueError(f"manifest 存在重复 sample_id：{sid}")
            if idx < 0 or idx >= len(raw):
                raise IndexError(f"train_index 越界：{idx}")
            expected_id = f"TOXICN_train_{idx:04d}"
            if sid != expected_id:
                raise ValueError(f"sample_id/index 不匹配：{sid} != {expected_id}")
            content = str(raw[idx].get("content", ""))
            if sha256_text(content) != row["content_sha256"].strip():
                raise ValueError(f"文本哈希不匹配，数据版本可能不同：{sid}")
            seen.add(sid)
            records.append({
                "sample_id": sid, "train_index": idx,
                "content": content,
                "toxic": raw[idx].get("toxic"),
                "topic": raw[idx].get("topic", ""),
                "platform": raw[idx].get("platform", ""),
                "length_chars": len(content),
                "toxic_type": raw[idx].get("toxic_type", 0),
                "expression": raw[idx].get("expression", 0),
                "sample_source": row.get("sample_source", ""),
            })
    if len(records) != 512:
        raise ValueError(f"本试点预期 512 条样本，manifest 实际为 {len(records)} 条")
    return records


def load_items(path):
    items = []
    required = {"条目id", "构念id", "构念名", "区块", "题式", "方向", "条目问句", "prompt_variants"}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            raise ValueError(f"概念 CSV 缺少列：{sorted(required-fields)}")
        for row in reader:
            raw_variants = row["prompt_variants"].strip()
            variants = [x.strip() for x in raw_variants.split(";") if x.strip()] if ";" in raw_variants else ([raw_variants] if raw_variants else [])
            if not variants or any(v not in PROMPTS for v in variants):
                raise ValueError(f"{row['条目id']} 的 prompt_variants 非法：{variants}")
            if not row["条目问句"].strip():
                raise ValueError(f"{row['条目id']} 问句为空")
            items.append({
                "item_id": row["条目id"].strip(),
                "construct_id": row["构念id"].strip(),
                "construct_name": row["构念名"].strip(),
                "axis": row["区块"].strip(),
                "form": row["题式"].strip(),
                "direction": row["方向"].strip(),
                "question": row["条目问句"].strip(),
                "variants": variants,
            })
    if len(items) != 16:
        raise ValueError(f"预期 16 个条目版本（4 个旧/修订 + 12 个新候选），实际 {len(items)}")
    ids = [x["item_id"] for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError("概念 CSV 中 item_id 重复")
    return items


def candidate_ids(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"verbalizer {word!r} 不是单 token：{ids}；停止以免错误测量")
    return ids


def build_sampling_params(allowed_ids):
    from vllm import SamplingParams
    allowed = sorted(set(allowed_ids))
    def processor(*args, **kwargs):
        import torch
        logits = kwargs.get("logits")
        if logits is None:
            logits = next(a for a in args if isinstance(a, torch.Tensor))
        masked = torch.full_like(logits, float("-inf"))
        masked[allowed] = logits[allowed]
        return masked
    attempts = [
        ("logits_processors", {"logits_processors": [processor]}),
        ("allowed_token_ids", {"allowed_token_ids": allowed}),
        ("unconstrained_fallback", {}),
    ]
    errors=[]
    for name, extra in attempts:
        try:
            return SamplingParams(max_tokens=1, temperature=0, logprobs=50, **extra), name
        except TypeError as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError("无法构建首 token SamplingParams：" + " | ".join(errors))


def get_mass(logprob_map, yes_ids, no_ids):
    yes_mass=sum(math.exp(logprob_map[t].logprob) for t in yes_ids if t in logprob_map)
    no_mass=sum(math.exp(logprob_map[t].logprob) for t in no_ids if t in logprob_map)
    total=yes_mass+no_mass
    if total <= 0:
        return None, 0.0
    return yes_mass/total, total


def main():
    ap=argparse.ArgumentParser(description="v9 512样本概念测量试点（GLM/vLLM）")
    ap.add_argument("--model_path", required=True, help="本地 GLM-4-9B-Chat 权重目录")
    ap.add_argument("--train_json", required=True, help="原始 TOXICN train.json")
    ap.add_argument("--manifest_csv", required=True, help="v9_pilot_512_manifest.csv")
    ap.add_argument("--items_csv", required=True, help="cbm_concept_set_v9_pilot_14.csv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--max_num_seqs", type=int, default=32)
    ap.add_argument("--max_num_batched_tokens", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=128)
    args=ap.parse_args()
    if args.batch_size < 1: raise ValueError("batch_size 必须大于 0")
    if not 0 < args.gpu_memory_utilization < 1: raise ValueError("gpu_memory_utilization 必须在 0 和 1 之间")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    train_path=Path(args.train_json); manifest_path=Path(args.manifest_csv); items_path=Path(args.items_csv)
    for p in (train_path,manifest_path,items_path,Path(args.model_path)):
        if not p.exists(): raise FileNotFoundError(p)
    raw=json.load(open(train_path,encoding="utf-8"))
    samples=load_manifest(manifest_path,raw)
    items=load_items(items_path)
    schedule=[]
    for item in items:
        for variant in item["variants"]:
            schedule.append((item,variant,f"{item['item_id']}__{variant}"))
    col_ids=[x[2] for x in schedule]
    if len(col_ids)!=len(set(col_ids)): raise ValueError("测量 schedule 存在重复列名")
    expected=512*len(schedule)

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    matrix_path=out/'v9_pilot_matrix.csv'
    audit_path=out/'v9_pilot_firsttoken.csv'
    meta_path=out/'v9_pilot_run_meta.json'
    for p in (matrix_path,audit_path,meta_path):
        if p.exists(): raise FileExistsError(f"输出文件已存在，拒绝覆盖：{p}")

    model_path=Path(args.model_path)
    tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,padding_side='right',local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    llm=LLM(model=str(model_path),trust_remote_code=True,dtype='auto',gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,max_model_len=args.max_model_len,max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,enforce_eager=True,max_logprobs=50)
    chat_template=getattr(tokenizer,'chat_template',None) or ''
    chat_kwargs={'enable_thinking':False} if 'enable_thinking' in chat_template else {}
    yes_ids=candidate_ids(tokenizer,'是'); no_ids=candidate_ids(tokenizer,'否')
    sampling,restrict_mode=build_sampling_params(yes_ids+no_ids)
    model_name=model_path.name
    suffix=MODEL_SUFFIX.get(model_name,'\n')

    values={s['sample_id']:{c:None for c in col_ids} for s in samples}
    total_done=0; mass_sum=0.0; mass_low=0; invalid=0; firsttoken_exact=0
    with open(audit_path,'w',encoding='utf-8-sig',newline='') as af:
        aw=csv.writer(af); aw.writerow(['sample_id','item_variant','item_id','prompt_variant','first_token','p_yes','cand_mass'])
        batch_prompts=[]; batch_keys=[]
        def flush_batch():
            nonlocal total_done,mass_sum,mass_low,invalid,firsttoken_exact,batch_prompts,batch_keys
            if not batch_prompts: return
            prompts=[]
            for raw_prompt in batch_prompts:
                messages=[{'role':'user','content':raw_prompt}]
                prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**chat_kwargs)
                prompt += suffix if suffix is not None else ('' if prompt.endswith('\n') else '\n')
                prompts.append(prompt)
            outputs=llm.generate(prompts,sampling,use_tqdm=True)
            if len(outputs)!=len(batch_keys): raise RuntimeError(f"vLLM 输出数量不匹配：{len(outputs)} != {len(batch_keys)}")
            for key,output in zip(batch_keys,outputs):
                sid,col,item_id,pv=key
                if not output.outputs or not output.outputs[0].logprobs or not output.outputs[0].logprobs[0]:
                    raise RuntimeError(f"缺少首 token logprobs：{sid}/{col}")
                p_yes,cand_mass=get_mass(output.outputs[0].logprobs[0],yes_ids,no_ids)
                token_ids=output.outputs[0].token_ids
                first=tokenizer.decode(token_ids) if token_ids else ''
                values[sid][col]=p_yes
                aw.writerow([sid,col,item_id,pv,first,'' if p_yes is None else round(p_yes,6),round(cand_mass,6)])
                total_done+=1; mass_sum+=cand_mass
                if cand_mass<0.95: mass_low+=1
                if p_yes is None: invalid+=1
                if first.strip() in ('是','否'): firsttoken_exact+=1
            af.flush(); batch_prompts=[]; batch_keys=[]

        for sample in samples:
            for item,pv,col in schedule:
                raw_prompt=PROMPTS[pv].format(text=sample['content'],question=item['question'])
                batch_prompts.append(raw_prompt)
                batch_keys.append((sample['sample_id'],col,item['item_id'],pv))
                if len(batch_prompts)>=args.batch_size: flush_batch()
        flush_batch()

    if total_done!=expected: raise RuntimeError(f"测量数量不完整：{total_done}/{expected}")
    with open(matrix_path,'w',encoding='utf-8-sig',newline='') as mf:
        mw=csv.writer(mf)
        meta_cols=['sample_id','train_index','sample_source','toxic','topic','platform','length_chars','toxic_type','expression']
        mw.writerow(meta_cols+col_ids)
        for s in samples:
            mw.writerow([s[k] for k in meta_cols]+[values[s['sample_id']][c] for c in col_ids])
    meta={
        'script':'run_v9_concept_pilot.py','model_name':model_name,'model_path':str(model_path),
        'train_json':str(train_path),'train_json_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),
        'manifest_csv':str(manifest_path),'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'items_csv':str(items_path),'items_csv_sha256':hashlib.sha256(items_path.read_bytes()).hexdigest(),
        'n_samples':len(samples),'n_item_prompt_conditions':len(schedule),'expected_queries':expected,
        'completed_queries':total_done,'prompt_schedule':{item['item_id']:item['variants'] for item in items},
        'prompt_templates':PROMPTS,'verbalizer_ids':{'是':yes_ids,'否':no_ids},'restricted_decoding':restrict_mode,
        'cand_mass_mean':mass_sum/total_done,'cand_mass_below_0_95_rate':mass_low/total_done,
        'invalid_rate':invalid/total_done,'firsttoken_exact_yes_no_rate':firsttoken_exact/total_done,
        'batch_size':args.batch_size,'timestamp':datetime.now().isoformat(timespec='seconds'),
        'notes':'p_yes is normalized over yes/no candidate mass; raw p_yes retained for reverse-direction items.'
    }
    with open(meta_path,'w',encoding='utf-8') as f: json.dump(meta,f,ensure_ascii=False,indent=2)
    print(f"完成：{total_done}/{expected} queries")
    print(f"cand_mass_mean={meta['cand_mass_mean']:.6f}; below .95={meta['cand_mass_below_0_95_rate']:.3%}; invalid={invalid}")
    print(f"矩阵：{matrix_path}\n首 token 审计：{audit_path}\n运行配置：{meta_path}")

if __name__=='__main__':
    main()
