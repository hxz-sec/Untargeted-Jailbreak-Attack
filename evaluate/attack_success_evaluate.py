import json


file_name = "advbench_untarget_gptfuzz_example_rej_100"
# file_name = "harmBench"


target_model_name = "Llama-3-8B-Instruct"
#target_model_name = "Llama-3.1-8B-Instruct"
#target_model_name = "Qwen-2.5-7B-Instruct"
#target_model_name = "Qwen-3-8B-Instruct"
#target_model_name = "Vicuna-7B-v1.5"
#target_model_name = "Mistral-7B-Instruct-v0.3"



guard_name = "gptfuzz"
# guard_name = "harmBench"



gptfuzz_success = 0
harmbench_success = 0

file_path = f"/data/home/Xinzhe/Kedong/repos/Untarget/github/output/gptfuzz/{target_model_name}/output/{file_name}_eval.jsonl"
print("file_name:",file_name)
print("guard_name:",guard_name)
print("target_model_name:",target_model_name)
print("iter:",iter)

lines = 0 
with open(file_path, "r") as f:
    for line in f:
        lines += 1
        try:
            data = json.loads(line.strip())
            if data.get("gptfuzz_result", 0) == 1:
                gptfuzz_success += 1
            if data.get("harmBench_result", 0) == 1:
                harmbench_success += 1
                
        except json.JSONDecodeError:
            continue

print(f"amount: {lines}")
print(f"gptfuzz_success_amount: {gptfuzz_success}")
print(f"harmbench_success_amount: {harmbench_success}")