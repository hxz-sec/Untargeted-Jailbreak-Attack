import requests
import os
import argparse
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from accelerate import Accelerator
from torch.optim import AdamW
import torch.nn as nn
import pandas as pd
from tqdm import trange, tqdm
import torch.nn.functional as F
import json
import numpy as np
import os
from utils import *
from transformers.configuration_utils import PretrainedConfig


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" 


def config_get(self, key, default=None):
    return getattr(self, key, default)

PretrainedConfig.get = config_get



if __name__ == "__main__":


    file_name = "advbench_untarget_gptfuzz_example_rej_100"
    # file_name = "harmBench"


    target_model_name = "llama-3-8B-Instruct"
    #target_model_name = "Llama-3.1-8B-Instruct"
    #target_model_name = "Qwen-2.5-7B-Instruct"
    #target_model_name = "Qwen-3-8B-Instruct"
    #target_model_name = "Vicuna-7B-v1.5"
    #target_model_name = "Mistral-7B-Instruct-v0.3"

    # guard_name = "guard3"
    guard_name = "gptfuzz"

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
    

    roberta_model = RoBERTaPredictor('/hub/huggingface/models/hubert233/GPTFuzz/', device='cuda:0')
    harmBench_path = "/hub/huggingface/models/cais/HarmBench-Llama-2-13b-cls"
    harmBench_model, harmBench_tokenizer = HarmBench_init()
    
    input_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/github/input/gptfuzz/{target_model_name}/{file_name}.jsonl"
    output_file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/github/output/gptfuzz/{target_model_name}/output/{file_name}_eval.jsonl"


    prompts = get_prompts(input_file_path,"prompt_with_adv")
    orginal = get_prompts(input_file_path,"harmful")
    print(f"{len(prompts)} prompt_with_adv")
    print(f"{len(orginal)} harmful")
    
    exist_ids = []
    if os.path.exists(output_file_path):
        exist_ids = get_prompts(output_file_path,"id")


    gptfuzz_success_amount = 0
    harmBench_success_amount = 0

    with open(output_file_path, 'a', encoding='utf-8') as f:
        for i in trange(0, len(prompts)):
            if i in exist_ids:
                continue
            print(f"已跳过{len(exist_ids)}条,当前id:{i}")

            rsp = None
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
                # 1. Gptfuzz roberta evaluate, 1:success, 0:fail
                jail_result = roberta_model.predict(rsp)[0]
                gptfuzz_success_amount += jail_result
                # 2. HarmBench evaluate, 1:success, 0:fail
                harmBench_result = HarmBench_eval(harmBench_model, harmBench_tokenizer, prompts[i], rsp)
                print("harmBench_result:",harmBench_result)
                if harmBench_result == None:
                    harmBench_result = 0
                harmBench_success_amount += harmBench_result


            print(f"target_model_name:{target_model_name}")
            print("id:",i,"gptfuzz_result:",jail_result,"harmBench_result:",harmBench_result)
            print("id:",i,"gptfuzz_success_amount:",gptfuzz_success_amount,"harmBench_success_amount:",harmBench_success_amount)
            json.dump({
                "id": i,
                "orginal": orginal[i],
                "jailbreak_prompt": prompts[i],
                "response": rsp,
                "gptfuzz_result":jail_result,
                "harmBench_result":harmBench_result,
            }, f, ensure_ascii=False)
            f.write('\n')
            if (i + 1) % 1 == 0:
                f.flush()   

print("succesfully finished")
print(output_file_path)