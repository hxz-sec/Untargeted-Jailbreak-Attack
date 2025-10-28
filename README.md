# UNTARGETED JAILBREAK ATTACK


## Abstract

Existing gradient-based jailbreak attacks on Large Language Models (LLMs),such as Greedy Coordinate Gradient (GCG) and COLD-Attack, typically optimizeadversarial sufixes to align the LLM output with a predefined target responseHowever, by restricting the optimization objective as inducing a predefined target.these methods inherently constrain the adversarial search space, which limit theiroverall attack effcacy. Furthermore, existing methods typically require a largenumber of optimization iterations to fulfll the large gap between the fixed targetand the original model response, resulting in low attack effciency.To overcome the limitations of targeted jailbreak attacks, we propose the firstgradient-based untargeted jailbreak attack (UJA), aiming to elicit an unsafe re-sponse without enforcing any predefined patterns. Specifcally, we formulatean untargeted attack objective to maximize the unsafety probability of the LLMresponse, which can be quantified using a judge model. Since the objective isnon-differentiable, we further decompose it into two differentiable sub-objectivesfor optimizing an optimal harmful response and the corresponding adversarialprompt, with a theoretical analysis to validate the decomposition. In contrast totargeted jailbreak attacks, UJA's unrestricted objective significantly expands thesearch space, enabling a more flexible and effcient exploration ofLLM vulnera-bilities. Extensive evaluations demonstrate that UJA can achieve over 80% attacksuccess rates against recent safety-aligned LLMs with only 100 optimization itera-tions, outperforming the state-of-the-art gradient-based attacks such as I-GCG andCOLD-Attack by over 20%.

**Figure 1.** Jailbreak example.  
![Jailbreak example](figs/image-3.png)

**Figure 2.** Workflow of UJA framework.  
![Workflow](figs/workflow-2_Page1_Image1.jpg)

## Quick Start

### Preparation

Befor we start, we need to download the target LLMs you want to jailbreak and the dataset you want to use. We list the six target LLMs and two benchmark datasets used in our paper below:

**Target LLMs** </br>
1. Llama-3-8B-Instruct
2. Llama-3.1-8B-Instruct
3. Qwen-2.5-7B-Instruct
4. Qwen-3-8B-Instruct
5. Vicuna-7B-v1.5
6. Mistral-7B-Instruct-v0.3

Below is the links you can download these LLMs:

---------------
| Model Name               | Link                                                       |
|--------------------------|------------------------------------------------------------|
| Llama-3-8B-Instruct      | https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct |
| Llama-3.1-8B-Instruct    | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct    |
| Qwen2.5-7B-Instruc       | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct            |
| Qwen3-8B-Instruct        | https://huggingface.co/Qwen/Qwen3-8B                       |
| Vicuna-7B-v1.5           | https://huggingface.co/lmsys/vicuna-7b-v1.5                |
| Mistral-7B-Instruct-v0.3 | https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3  |




**Benchmark Datasets** </br>
1. AdvBench
2. HarmBench 
3. StrongReject

| Dataset      | Link                                                         |
|-----------   |--------------------------------------------------------------|
| AdvBench     | https://github.com/llm-attacks/llm-attacks                   |
| HarmBench    | https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors |



After downloading three benchmark datasets, you need to put them into ``./data`` folder (if not exist, create a folder with the same name).


Additionally, in order to evaluate the harmfulness of generated responses, we employ two evaluators in our paper to score each response: </br>
1. GPTFuzzer (ASR-G)
2. HarmBench-Llama-2-13b-cls (ASR-H)

| Model Name             | Link                                                       |
|------------------------|------------------------------------------------------------|
| GPTFuzzer              | https://huggingface.co/hubert233/GPTFuzz                   |
| HarmBench-Llama-2-13b-cls       | https://huggingface.co/cais/HarmBench-Llama-2-13b-cls      |



### Code
**1) Download this GitHub**
```
git clone https://github.com/hxz-sec/Untargeted-Jailbreak-Attack
```

**2) Setup Environment**

We recommend conda for setting up a reproducible experiment environment.
We include `environment.yaml` for creating a working environment:

```bash
conda env create -f environment.yaml -n ca2
```

Then use requirements.txt or manually install pip in the project:
```
pip install -r requirements.txt
```

**3) Run Command for Untargeted-Jailbreak-Attack**

Run the main code :
```bash
conda activate ca2
```
```
python scripts/main_split_gptfuzz.py
```
Evaluate:
```
python evaluate/evaluate.py 
```

Drawing:
```
python plot/asr_comparison_methods.py 
python plot/attack_asr_comparison.py 
python plot/response_diversity.py
```




## Experiments

### Result
Here we demonstrate the effectiveness of UJA compared to the baseline. The detailed results are shown in the following table：



Table: Comparison of ASRs achieved by UJA and baseline methods across two datasets on six white-box LLMs. **The ASRs are measured after performing *only 100 iterations* for each prompt.**

| Method       | Metric | Llama-3 | Llama-3.1 | Qwen-2.5 | Qwen-3 | Vicuna | Mistral | Llama-3 | Llama-3.1 | Qwen-2.5 | Qwen-3 | Vicuna | Mistral |
|--------------|--------|---------|-----------|----------|--------|--------|---------|---------|-----------|----------|--------|--------|---------|
|              |        |                 **AdvBench**             ||     |                        |        |         |                 **HarmBench**               |   |                        |        |         |
| GCG          | ASR-G  | 50.0    | 51.0      | 31.0     | 30.0   | 28.0   | 70.0    | 22.0    | 31.0      | 24.0     | 18.0   | 22.0   | 37.0    |
|              | ASR-H  | 40.0    | 42.0      | 37.0     | 15.0   | 21.0   | 81.0    | 40.0    | 50.0      | 53.0     | 19.0   | 12.0   | 67.0    |
| COLD         | ASR-G  | 52.0    | 57.0      | 28.0     | 54.0   | 52.0   | 72.0    | 38.0    | 43.0      | 32.0     | 36.0   | 39.0   | 38.0    |
|              | ASR-H  | 44.0    | 47.0      | 35.0     | 27.0   | 40.0   | 73.0    | 41.0    | 44.0      | 51.0     | 24.0   | 27.0   | 75.0    |
| DRL          | ASR-G  | 30.0    | 25.0      | 36.0     | 24.0   | 29.0   | 34.0    | 33.0    | 35.0      | 41.0     | 22.0   | 30.0   | 34.0    |
|              | ASR-H  | 28.0    | 45.0      | 64.0     | 42.0   | 27.0   | 94.0    | 44.0    | 37.0      | 78.0     | 39.0   | 55.0   | 84.0    |
| PAP          | ASR-G  | 21.0    | 31.0      | 41.0     | 14.0   | 2.0    | 38.0    | 16.0    | 19.0      | 33.0     | 10.0   | 1.0    | 31.0    |
|              | ASR-H  | 62.0    | 77.0      | 82.0     | 74.0   | 3.0    | 84.0    | 64.0    | 71.0      | 84.0     | 76.0   | 3.0    | 77.0    |
| AdvPrefix    | ASR-G  | 40.0    | 42.0      | 28.0     | 29.0   | 41.0   | 66.0    | 43.0    | 44.0      | 29.0     | 25.0   | 36.0   | 60.0    |
|              | ASR-H  | 15.0    | 22.0      | 36.0     | 12.0   | 17.0   | 65.0    | 20.0    | 24.0      | 36.0     | 6.0    | 16.0   | 59.0    |
| I-GCG        | ASR-G  | 23.0    | 23.0      | 8.0      | 12.0   | 25.0   | 38.0    | 13.0    | 16.0      | 17.0     | 8.0    | 17.0   | 30.0    |
|              | ASR-H  | 11.0    | 13.0      | 10.0     | 2.0    | 5.0    | 38.0    | 4.0     | 9.0       | 19.0     | 5.0    | 6.0    | 41.0    |
| llm-adaptive | ASR-G  | 51.0    | 60.0      | 29.0     | 62.0   | 41.0   | 44.0    | 37.0    | 35.0      | 16.0     | 41.0   | 23.0   | 33.0    |
|              | ASR-H  | 0.0     | 1.0       | 32.0     | 6.0    | 1.0    | 46.0    | 7.0     | 3.0       | 31.0     | 2.0    | 2.0    | 56.0    |
| Ours (UJA)   | ASR-G  | 89.0    | 86.0      | 74.0     | 59.0   | 88.0   | 88.0    | 65.0    | 47.0      | 64.0     | 56.0   | 66.0   | 67.0    |
|              | ASR-H  | 67.0    | 80.0      | 55.0     | 33.0   | 59.0   | 85.0    | 73.0    | 62.0      | 66.0     | 29.0   | 64.0   | 81.0    |


**Figure 3.** t-SNE visualization of response embeddings generated by six jailbreak methods on the AdvBench dataset.  
![t-SNE visualization](figs/image.png)

**Figure 4.** Convergence of cumulative ASR of UJA on four LLMs from the AdvBench dataset: (a) ASR-G and (b) ASR-H.  
<img src="figs/iter.png" alt="UJA Untargeted Strategy" width="1070">

**Figure 5.** Example of a UJA Optimized Jailbreak Prompt across Multiple LLMs on AdvBench.
![alt text](image.png)

**Figure 6.** Example of a UJA-Optimized Jailbreak Prompt Compared with Baselines on Llama-3 across AdvBench.  
![Llama-3 Attack Result](figs/image-4.png)

**Figure 7.**  Example of a UJA Optimized Jailbreak Response Compared with Baselines on Qwen-3 across AdvBench.
<img src="figs/example- compare .png" alt="UJA Untargeted Strategy" width="1400">


## Citation
If you find this useful in your research, please consider citing:


```

arXiv Preprint:
```

@misc{huang2025UJA,
      title={Untargeted Jailbreak Attack}, 
      author={Xinzhe Huang and Wenjing Hu and Tianhang Zheng and Kedong Xiu and Xiaojun Jia and Di Wang and Zhan Qin and Kui Ren},
      year={2025},
      eprint={2510.02999},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2510.02999}, 
}



## Lisence
This project is licensed under the MIT License.