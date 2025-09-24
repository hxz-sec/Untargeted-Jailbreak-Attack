import requests
import os
import argparse
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from accelerate import Accelerator
# from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
import torch.nn as nn
import pandas as pd
from tqdm import trange, tqdm
import torch.nn.functional as F
import json
import numpy as np
import os
from utils import *
# from guard4_utils import load_guard4_model, guard4_predict
from transformers.configuration_utils import PretrainedConfig


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


os.environ["CUDA_VISIBLE_DEVICES"] = "0,2" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" 


def config_get(self, key, default=None):
    return getattr(self, key, default)

PretrainedConfig.get = config_get



if __name__ == "__main__":

    # file_name = "Untarget_DST-DNA_100_DS-suffix_MicroBZ-1_ITER-20-2025-04-30-00:39"
    method = None
    # method="gcg"
    # method="prp"
    # method="cold_suffix"
    # method="advPrefix"
    # method="TransferAttack"
    file_name = "advbench"
    # file_name = "harmBench"
    target_model_name = "llama-3-8B-Instruct"
    #target_model_name = "Llama-3.1-8B-Instruct"
    #target_model_name = "Qwen-2.5-7B-Instruct"
    #target_model_name = "Qwen-3-8B-Instruct"
    #target_model_name = "Vicuna-7B-v1.5"
    #target_model_name = "Mistral-7B-Instruct-v0.3"
    no_semantic = None
    # no_semantic = True
    guard_name = "guard3"
    # guard_name = "gptfuzz"
    # guard_name = "harmBench"
    iter = None
    model_type = "white"
    pipe,model,tokenizer = None,None,None
    if target_model_name == "Qwen-2.5-7B-Instruct":
        target_llm_path = f"/hub/huggingface/models/Qwen/Qwen2.5-7B-Instruct/"
        pipe,model,tokenizer = get_model_inference_pipeline(target_llm_path)
    elif target_model_name == "llama-3-8B-Instruct":
        target_llm_path = f"/hub/huggingface/models/meta/Llama-3-8B-Instruct/"
        pipe,model,tokenizer = get_model_inference_pipeline(target_llm_path)
    elif target_model_name == "Vicuna-7b-v1.5":
        target_llm_path = f"/hub/huggingface/models/vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d/"    
        pipe,model,tokenizer = get_model_inference_pipeline(target_llm_path)    
    elif target_model_name == "Mistral-7b":
        target_llm_path = f"/hub/huggingface/models/MistralAI/Mistral-7B-Instruct-v0.3/"
        pipe,model,tokenizer = get_model_inference_pipeline(target_llm_path)
    elif target_model_name == "gemma-2-9b-it":
        target_llm_path = f"/hub/huggingface/models/google/gemma-2-9b-it/"
        pipe,model,tokenizer = get_model_inference_pipeline(target_llm_path)
    elif target_model_name == "Qwen3-8B":
        target_llm_path = f"/hub/huggingface/models/Qwen/Qwen3-8B/"
        pipe,model,tokenizer = get_model_inference_pipeline(target_llm_path)
    elif target_model_name == "Llama-3.1-8B-Instruct":
        target_llm_path = f"/hub/huggingface/models/meta/Meta-Llama-3.1-8B-Instruct-abliterated/"
        pipe,model,tokenizer = get_model_inference_pipeline(target_llm_path)        
    else:
        model_type = "Black"
    
    # guard_model_path = "/hub/huggingface/models/meta/Llama-Guard-3-8B/"
    # _, guard_model, guard_tokenizer = get_model_inference_pipeline(guard_model_path)
    roberta_model = RoBERTaPredictor('/hub/huggingface/models/hubert233/GPTFuzz/', device='cuda:0')
    harmBench_path = "/hub/huggingface/models/cais/HarmBench-Llama-2-13b-cls"
    harmBench_model, harmBench_tokenizer = HarmBench_init()
    harmBench_tokenizer.pad_token = harmBench_tokenizer.eos_token
    input_file_path,output_file_path = None,None
    
    if model_type == "Black":
        input_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/output/{guard_name}/llama-3-8B-Instruct/{file_name}_untarget_{guard_name}_100.jsonl"
        output_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/output_score/{guard_name}/{target_model_name}/{file_name}_untarget_{guard_name}_100_evaluate.jsonl"
        if "/" in target_model_name:
            name = target_model_name.split("/")[-1]
            output_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/output_score/{guard_name}/{name}/{file_name}_untarget_{guard_name}_100_evaluate.jsonl"
    elif method:
        input_file_path = f"/data/home/Xinzhe/Untarget/data/csv/llama-3-8B-Instruct/{file_name}_{method}_100.jsonl"
        output_file_path = f"/data/home/Xinzhe/Untarget/data/evaluate/{target_model_name}/{file_name}_{method}_100_evaluate.jsonl"
    elif no_semantic:
        print("no_semantic")
        input_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/output/{guard_name}/{target_model_name}/{file_name}_untarget_{guard_name}_no_semantic_100.jsonl"
        output_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/output_score/{guard_name}/{target_model_name}/{file_name}_untarget_{guard_name}_no_semantic_100_evaluate.jsonl"
    elif iter:
        print("iter")
        input_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/output/{guard_name}/{target_model_name}/{file_name}_untarget_gptfuzz_{iter}_100.jsonl"
        output_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/output_score/{guard_name}/{target_model_name}/{file_name}_untarget_gptfuzz_{iter}_100_evaluate.jsonl"
    elif guard_name:
        input_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/output/{guard_name}/{target_model_name}/{file_name}_untarget_{guard_name}_100.jsonl"
        output_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/output_score/{guard_name}/{target_model_name}/{file_name}_untarget_{guard_name}_100_evaluate.jsonl"
    input_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/output/guard3/{target_model_name}/harmBench_untarget_guard3_early_100.jsonl"
    output_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/output_score/guard3/{target_model_name}/harmBench_untarget_guard3_early_100_evaluate.jsonl"
    
    # advbench_untarget_gptfuzz_100_no_stage1
    prompts = get_prompts(input_file_path,"prompt_with_adv")
    orginal = get_prompts(input_file_path,"harmful")
    print(f"{len(prompts)} prompt_with_adv")
    print(f"{len(orginal)} harmful")
    
    exist_ids = []
    if os.path.exists(output_file_path):
        exist_ids = get_prompts(output_file_path,"id")
    # os.makedirs(output_file_path, exist_ok=True)
    # guard3_success_amount = 0
    gptfuzz_success_amount = 0
    rej_words_success_amount = 0
    harmBench_success_amount = 0

    with open(output_file_path, 'a', encoding='utf-8') as f:
        for i in trange(0, len(prompts)):
            if i in exist_ids:
                continue
            print(f"已跳过{len(exist_ids)}条,当前id:{i}")

            rsp = None
            # if target_model_name == "gpt-4-turbo-2024-04-09" or target_model_name == "gemini-2.0-flash":
            if model_type == "Black":    
                if "/" in target_model_name:
                    rsp = response_together_response(prompts[i],target_model_name)
                else:
                    rsp = LLM(prompts[i], target_model_name)
            elif "qwen" in target_model_name or "qwq" in target_model_name:
                rsp = qwen(prompts[i], target_model_name)
            else:
                rsp = response_cold(model,tokenizer,prompts[i])
            print("response:",rsp)

            if rsp == None:
                unsafe_score=0
                jail_result=0
                keyword_result=0
                harmBench_result=0
            
            else :
                # 1. Guard3 evaluate, unsafe>0.5
                # _, unsafe_score = Guard3_moderate(rsp, guard_model, guard_tokenizer)
                # guard3_success_amount += binarize(unsafe_score)
                
                # 2. Gptfuzz roberta evaluate, 1:success, 0:fail
                jail_result = roberta_model.predict(rsp)[0]
                gptfuzz_success_amount += jail_result
                #  3. Reject keywords
                keyword_result = get_reject_words_score(rsp)
                rej_words_success_amount += keyword_result
                # 4. HarmBench evaluate, 1:success, 0:fail
                harmBench_result = HarmBench_eval(harmBench_model, harmBench_tokenizer, prompts[i], rsp)
                print("harmBench_result:",harmBench_result)
                if harmBench_result == None:
                    harmBench_result = 0
                harmBench_success_amount += harmBench_result
            # strongreject_gpt4_score = strongreject_gpt4_eval(prompts[i],rsp)
            # strongreject_finetune_score = strongreject_finetuned_eval(prompts[i],rsp,strong_finetune_model,strong_finetune_tokenizer)
                
            strongreject_gpt4_score,    strongreject_finetune_score= -1,-1 
            print(f"strongreject_finetune_score: {strongreject_finetune_score}")
            print(f"strongreject_gpt4_score: {strongreject_gpt4_score}")      
            print(f"target_model_name:{target_model_name}")
            print("id:",i,"gptfuzz_result:",jail_result,"keyword_result:",keyword_result,"harmBench_result:",harmBench_result)
            print("id:",i,"gptfuzz_success_amount:",gptfuzz_success_amount,"rej_words_success_amount:",rej_words_success_amount,"harmBench_success_amount:",harmBench_success_amount)
            json.dump({
                "id": i,
                "orginal": orginal[i],
                "jailbreak_prompt": prompts[i],
                "response": rsp,
                "gptfuzz_result":jail_result,
                "keyword_result":keyword_result,
                "harmBench_result":harmBench_result,
                "strongreject_finetune_score":strongreject_finetune_score,                
                "strongreject_gpt4_score": strongreject_gpt4_score
            }, f, ensure_ascii=False)
            f.write('\n')
            if (i + 1) % 1 == 0:
                f.flush()   

print("succesfully finished")
print(output_file_path)